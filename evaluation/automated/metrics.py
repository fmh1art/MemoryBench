"""RGB [0,1] full-reference metrics and explicitly named artifact diagnostics."""

import math

import numpy as np
from scipy.ndimage import gaussian_filter, laplace

METRICS = {
    "mse": ("lower", "RGB squared error"),
    "mae": ("lower", "RGB absolute error"),
    "psnr_db": ("higher", "PSNR of clip MSE; identical clips are +inf"),
    "ssim": ("higher", "11x11 Gaussian, sigma 1.5, population covariance, RGB mean"),
    "ms_ssim": ("higher", "Five scales, Wang weights, nonnegative structure terms"),
    "temporal_difference_l1": ("lower", "L1 between signed adjacent-frame differences"),
    "final_mse": ("lower", "Last aligned frame MSE"),
    "final_ssim": ("higher", "Last aligned frame SSIM"),
    "gen_laplacian_variance": (
        "diagnostic",
        "Sharpness proxy; noise also increases this",
    ),
    "gt_laplacian_variance": ("diagnostic", "Reference sharpness proxy"),
    "gen_adjacent_l1": (
        "diagnostic",
        "Motion / flicker proxy; static output is not necessarily better",
    ),
    "gt_adjacent_l1": ("diagnostic", "Reference adjacent-frame change"),
    "lpips": ("lower", "AlexNet LPIPS v0.1; pretrained weights required"),
    "final_lpips": ("lower", "Last aligned frame AlexNet LPIPS"),
    "niqe": ("lower", "Generated-frame NIQE via PyIQA; domain dependent"),
}


def _ssim(a, b):
    """Return luminance*contrast-structure and contrast-structure, per channel."""
    if min(a.shape[:2]) < 11:
        raise ValueError("SSIM requires both dimensions >= 11")

    def blur(x):
        return gaussian_filter(x, sigma=(1.5, 1.5, 0), truncate=10 / 3)[5:-5, 5:-5]

    ma, mb = blur(a), blur(b)
    va = np.maximum(blur(a * a) - ma * ma, 0)
    vb = np.maximum(blur(b * b) - mb * mb, 0)
    cov = blur(a * b) - ma * mb
    cs = (2 * cov + 0.03**2) / (va + vb + 0.03**2)
    luminance = (2 * ma * mb + 0.01**2) / (ma * ma + mb * mb + 0.01**2)
    return (luminance * cs).mean(axis=(0, 1)), cs.mean(axis=(0, 1))


def ms_ssim(a, b):
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    if min(a.shape[:2]) < 161:
        return None  # Do not silently change the number of scales.
    factors = []
    for index, weight in enumerate(weights):
        score, cs = _ssim(a, b)
        factors.append(np.maximum(score if index == 4 else cs, 0) ** weight)
        if index < 4:

            def down(x):
                h, w = x.shape[:2]
                x = np.pad(x, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
                return (x[::2, ::2] + x[1::2, ::2] + x[::2, 1::2] + x[1::2, 1::2]) / 4

            a, b = down(a), down(b)
    return float(np.prod(factors, axis=0).mean())


def psnr(mse):
    return "inf" if mse == 0 else float(-10 * math.log10(mse))


def compute(gt, gen):
    """Frames are aligned arrays T,H,W,3. Values are raw, never normalized together."""
    if gt.shape != gen.shape or gt.ndim != 4 or gt.shape[-1] != 3 or not len(gt):
        raise ValueError("Expected equally shaped, nonempty T,H,W,3 videos")
    if (
        not np.isfinite(gt).all()
        or not np.isfinite(gen).all()
        or min(gt.min(), gen.min()) < 0
        or max(gt.max(), gen.max()) > 1
    ):
        raise ValueError("Expected finite RGB values in [0,1]")
    frame_rows = []
    for a, b in zip(gt, gen):
        delta = a - b
        gray_gt = a @ np.array([0.299, 0.587, 0.114])
        gray_gen = b @ np.array([0.299, 0.587, 0.114])
        frame_rows.append(
            {
                "mse": float(np.mean(delta**2)),
                "mae": float(np.mean(np.abs(delta))),
                "ssim": float(_ssim(a, b)[0].mean()),
                "ms_ssim": ms_ssim(a, b),
                "gt_laplacian_variance": float(laplace(gray_gt).var()),
                "gen_laplacian_variance": float(laplace(gray_gen).var()),
            }
        )
    scores = {
        name: float(np.mean([row[name] for row in frame_rows]))
        if all(row[name] is not None for row in frame_rows)
        else None
        for name in frame_rows[0]
    }
    scores.update(
        psnr_db=psnr(scores["mse"]),
        final_mse=frame_rows[-1]["mse"],
        final_ssim=frame_rows[-1]["ssim"],
    )
    scores.update(
        temporal_difference_l1=None, gt_adjacent_l1=None, gen_adjacent_l1=None
    )
    if len(gt) > 1:
        da, db = np.diff(gt, axis=0), np.diff(gen, axis=0)
        scores.update(
            temporal_difference_l1=float(np.abs(da - db).mean()),
            gt_adjacent_l1=float(np.abs(da).mean()),
            gen_adjacent_l1=float(np.abs(db).mean()),
        )
    return scores, frame_rows


class LearnedMetrics:
    """Optional local pretrained scorers; initialized once per worker, never random weights."""

    def __init__(self, names, device="cpu"):
        import torch

        self.torch, self.device, self.models = torch, device, {}
        if "lpips" in names:
            import lpips

            self.models["lpips"] = (
                lpips.LPIPS(net="alex", version="0.1").to(device).eval()
            )
        if "niqe" in names:
            import pyiqa

            self.models["niqe"] = pyiqa.create_metric("niqe", device=device)

    def compute(self, gt, gen):
        torch = self.torch
        values = {key: [] for key in self.models}
        with torch.inference_mode():
            for a, b in zip(gt, gen):
                a, b = [
                    torch.from_numpy(x.copy())
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(self.device)
                    for x in (a, b)
                ]
                for key, model in self.models.items():
                    value = model(2 * a - 1, 2 * b - 1) if key == "lpips" else model(b)
                    number = float(value.item())
                    if not math.isfinite(number):
                        raise ValueError(f"{key} returned a nonfinite value")
                    values[key].append(number)
        scores = {key: float(np.mean(v)) for key, v in values.items()}
        if "lpips" in values:
            scores["final_lpips"] = values["lpips"][-1]
        return scores
