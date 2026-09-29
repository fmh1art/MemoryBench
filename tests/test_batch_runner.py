import json
from pathlib import Path
from types import SimpleNamespace

import av
import imageio.v2 as imageio
import numpy as np
import pytest

from evaluation.pipeline.infer_evaluation_cases import load_cases
from evaluation.pipeline.run_memory_bench import (
    Run,
    comparison_video,
)


def test_dataset_relocation_resolves_current_paths(tmp_path):
    directory = tmp_path / "eval_001"
    (directory / "packed").mkdir(parents=True)
    (directory / "packed/index.json").write_text("{}")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "evaluation_case_id": "eval_001",
                        "directory": "/removed/old/location",
                        "packed_index": "/removed/index.json",
                    }
                ]
            }
        )
    )
    case = load_cases(tmp_path)[0]
    assert case["directory"] == str(directory)
    assert case["packed_index"] == str(directory / "packed/index.json")
    (directory / "packed/index.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path)


def test_error_and_unfinished_cases_are_not_zero_scores(tmp_path):
    cases = [
        {
            "evaluation_case_id": f"eval_{i}",
            "source_case_id": "source",
            "sensor_duration_seconds": {"prefix": 1},
            "generation_frame_count": 10,
            "evaluation_frame_count": 5,
        }
        for i in range(3)
    ]
    run = Run(SimpleNamespace(output=tmp_path, resume=False), cases)
    run.update("eval_0", status="completed", gate=0, memory_score=0.0)
    run.update("eval_1", status="error", error="API error")
    result = json.loads((tmp_path / "status.json").read_text())
    assert result["aggregate_denominator"] == 1 and result["mean_memory_score"] == 0
    assert result["status"] == "running"
    assert result["cases"][1]["memory_score"] is None
    run.update("eval_2", status="completed", gate=1, memory_score=0.8)
    result = json.loads((tmp_path / "status.json").read_text())
    assert result["mean_memory_score"] == 0.4 and result["gate_pass_rate"] == 0.5
    assert result["status"] == "completed_with_errors"


def test_comparison_keeps_gt_left_prediction_right_and_exact_frames(tmp_path):
    for name, channel in [("gt.mp4", 0), ("generated.mp4", 2)]:
        with imageio.get_writer(
            str(tmp_path / name), fps=30, codec="libx264", macro_block_size=1
        ) as writer:
            for i in range(6):
                frame = np.zeros((48, 64, 3), np.uint8)
                frame[:, :, channel] = 100 + 20 * i
                writer.append_data(frame)
    comparison_video(tmp_path / "gt.mp4", tmp_path / "generated.mp4", tmp_path, 6)
    with av.open(str(tmp_path / "comparison.mp4")) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    assert len(frames) == 6
    assert frames[0][40:, :64, 0].mean() > 90
    assert frames[0][40:, 64:, 2].mean() > 90
    assert frames[-1][40:, :64, 0].mean() > frames[0][40:, :64, 0].mean() + 90
