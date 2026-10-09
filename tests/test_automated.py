"""Numerical invariants, reference agreement, timestamp sampling and resume safety."""

import json

import av
import numpy as np
import pytest

pytest.importorskip("scipy")
from evaluation.automated.metrics import _ssim, compute
from evaluation.automated.run import evaluate, summarize
from evaluation.automated.video import sample


def test_identity_and_constant_offset():
    a = np.full((2, 192, 192, 3), 0.4, np.float32)
    same, _ = compute(a, a)
    assert same["mse"] == 0 and same["psnr_db"] == "inf"
    assert same["ssim"] == pytest.approx(1) and same["ms_ssim"] == pytest.approx(1)
    assert same["temporal_difference_l1"] == 0
    shifted, _ = compute(a, a + 0.1)
    assert shifted["mse"] == pytest.approx(0.01)
    assert shifted["mae"] == pytest.approx(0.1)
    assert shifted["psnr_db"] == pytest.approx(20)
    assert shifted["temporal_difference_l1"] == 0


def test_ssim_matches_independent_reference():
    skimage = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(7)
    a = rng.random((192, 192, 3), dtype=np.float32)
    b = np.clip(a + rng.normal(0, 0.05, a.shape), 0, 1).astype(np.float32)
    expected = skimage.structural_similarity(
        a,
        b,
        data_range=1,
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    assert _ssim(a, b)[0].mean() == pytest.approx(expected, abs=1e-6)


def test_static_and_temporal_failures_are_not_conflated():
    a = np.stack([np.zeros((32, 32, 3)), np.ones((32, 32, 3))]).astype(np.float32)
    b = np.zeros_like(a)
    metrics, _ = compute(a, b)
    assert metrics["temporal_difference_l1"] == 1
    assert metrics["gen_adjacent_l1"] == 0 and metrics["gt_adjacent_l1"] == 1
    assert metrics["ms_ssim"] is None
    one, _ = compute(a[:1], b[:1])
    assert one["temporal_difference_l1"] is None


def video(path, fps=10, count=10):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for i in range(count):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), i * 10, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_sample_offsets_identity_and_resume(tmp_path):
    path = tmp_path / "clip.mp4"
    video(path)
    case = {
        "gt_video": str(path),
        "gen_video": str(path),
        "gt_start_seconds": 0.1,
        "gen_start_seconds": 0.1,
    }
    a, b, info = sample(case, fps=10, width=320, height=192, max_frames=4)
    assert np.array_equal(a, b)
    assert info["effective_size"] == (256, 192)
    assert info["times_seconds"][0] == 0 and info["times_seconds"][-1] == pytest.approx(
        0.8
    )
    result = tmp_path / "input" / "case1" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps({"case_id": "case1", "case": case, "status": "gate_failed"})
    )
    settings = {
        "sampling": {
            "fps": 10,
            "width": 320,
            "height": 192,
            "max_frames": 4,
            "max_skew": 0.1,
        },
        "learned": [],
        "device": "cpu",
    }
    job = (result, tmp_path / "output", settings, True)
    first = evaluate(job)
    assert first["status"] == "complete"
    output = tmp_path / "output/case1/automated.json"
    stamp = output.stat().st_mtime_ns
    assert evaluate(job)["signature"] == first["signature"]
    assert output.stat().st_mtime_ns == stamp
    settings["sampling"]["max_frames"] = 3
    assert evaluate(job)["signature"] != first["signature"]
    summary = summarize(
        [
            first,
            {"case_id": "bad", "status": "error", "metrics": {}, "error": "missing"},
        ]
    )
    assert summary["metrics"]["ssim"]["n"] == 1
    assert summary["metrics"]["psnr_db"]["mean"] == "inf"
    assert len(summary["errors"]) == 1
