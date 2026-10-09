"""Timestamp-aligned future clips; no contact sheets, stretch, or time warping."""

import av
import numpy as np
from PIL import Image, ImageOps

from evaluation.video import _nearest, _timestamps


def sample(case, fps=24.0, width=320, height=192, max_frames=0, max_skew=0.1):
    times = [_timestamps(case[key]) for key in ("gt_video", "gen_video")]
    offsets = [
        float(case.get(key, 0)) for key in ("gt_start_seconds", "gen_start_seconds")
    ]
    if any(not np.isfinite(v) or v < 0 for v in offsets):
        raise ValueError("Video offsets must be finite and nonnegative")
    end = min(t[-1] - offset for t, offset in zip(times, offsets))
    if end <= 0:
        raise ValueError("No positive shared future duration")
    targets = np.arange(0, end, 1 / fps).tolist()
    if targets and end - targets[-1] < 0.5 / fps:
        targets[-1] = end
    else:
        targets.append(end)
    if max_frames and len(targets) > max_frames:
        targets = np.linspace(0, end, max_frames).tolist()
    indices = [
        [_nearest(t, target + offset) for target in targets]
        for t, offset in zip(times, offsets)
    ]
    actual = [
        [t[index] - offset for index in idx]
        for t, idx, offset in zip(times, indices, offsets)
    ]
    if any(
        abs(at - target) > max_skew
        for seq in actual
        for at, target in zip(seq, targets)
    ) or any(abs(a - b) > max_skew for a, b in zip(*actual)):
        raise ValueError("Decoded frame alignment exceeds max-skew-seconds")
    output, size = [], None
    for role, idx in zip(("gt_video", "gen_video"), indices):
        selected = {}
        wanted = set(idx)
        with av.open(case[role]) as container:
            for number, frame in enumerate(container.decode(video=0)):
                if number in wanted:
                    image = frame.to_image().convert("RGB")
                    if size is None:
                        size = ImageOps.contain(image, (width, height)).size
                    image = ImageOps.pad(
                        image, size, method=Image.Resampling.LANCZOS, color=(0, 0, 0)
                    )
                    selected[number] = np.asarray(image, dtype=np.float32) / 255
                if number >= max(idx):
                    break
        output.append(np.stack([selected[index] for index in idx]))
    return *output, {
        "times_seconds": targets,
        "gt_actual_seconds": actual[0],
        "gen_actual_seconds": actual[1],
        "effective_size": size,
        "requested_canvas": [width, height],
        "shared_end_seconds": end,
        "source_end_seconds": [t[-1] - offset for t, offset in zip(times, offsets)],
        "unique_frames": dict(zip(("gt", "gen"), (len(set(x)) for x in indices))),
    }
