"""Audit physical videos against annotated frame ranges and original MP4 PTS."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.case_definition"
import argparse
from fractions import Fraction
import json

import av
import numpy as np


def at_time(path, target):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(target / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            stamp = frame.pts * stream.time_base
            if stamp == target:
                return frame.to_ndarray(format="rgb24")
            if stamp > target:
                break
    raise ValueError(f"No exact frame at PTS {target}: {path}")


def validate(root):
    case = json.loads((root / "case.json").read_text())
    annotation = json.loads((root / "annotation.json").read_text())
    mapping = np.load(root / "frame_mapping.npz")
    conditions = np.load(root / "input/aligned_conditions.npz")
    p, t, es, ee = (
        case["marks"][key] for key in ("prefix_start", "t", "eval_start", "eval_end")
    )
    pe = case["marks"].get("prefix_end", t)
    selected = np.concatenate([np.arange(p, pe), np.arange(t, ee)])
    n = case["prefix_frame_count"] + case["generation_frame_count"]
    assert np.array_equal(conditions["source_frame_indices"], selected)
    assert np.all(np.diff(conditions["sensor_timestamp_ns"]) > 0)
    assert np.array_equal(
        mapping["source_frame"][~mapping["is_padding"]], selected
    )
    for side in ("left", "right"):
        points = conditions[f"{side}_hand_positions_ref"]
        assert points.shape == (n, 21, 3)
        assert np.isfinite(points[conditions[f"{side}_hand_valid"]]).all()
    assert conditions["c2w_memworld"].shape == (n, 4, 4)
    assert np.isfinite(conditions["c2w_memworld"]).all()
    counts = {
        "input/rgb_prefix.mp4": pe - p,
        "input/past_hands.mp4": pe - p,
        "input/future_hands.mp4": ee - t,
        "ground_truth/future.mp4": ee - t,
        "ground_truth/evaluation.mp4": ee - es,
        f"context/videos/{case['evaluation_case_id']}.mp4": len(
            mapping["source_frame"]
        ),
        f"context/keypoint_videos/{case['evaluation_case_id']}.mp4": len(
            mapping["source_frame"]
        ),
    }
    for name, count in counts.items():
        with av.open(str(root / name)) as container:
            stream = container.streams.video[0]
            assert stream.frames == count, (name, stream.frames, count)
            assert (stream.width, stream.height) == (case["width"], case["height"])
            assert stream.average_rate == 30
    tb = Fraction(*annotation["source"]["time_base"])
    comparisons = []
    for video, local, source in [
        ("input/rgb_prefix.mp4", 0, p),
        ("input/rgb_prefix.mp4", pe - p - 1, pe - 1),
        ("ground_truth/future.mp4", 0, t),
        ("ground_truth/evaluation.mp4", 0, es),
        ("ground_truth/evaluation.mp4", ee - es - 1, ee - 1),
    ]:
        pos = np.flatnonzero(mapping["source_frame"] == source)[0]
        original = at_time(
            annotation["source"]["video_path"],
            int(mapping["source_media_pts"][pos]) * tb,
        )
        exported = at_time(root / video, Fraction(local, 30))
        mae = float(np.abs(original.astype(float) - exported.astype(float)).mean())
        if mae > 8:
            raise ValueError(
                f"Possible RGB frame mismatch: {root.name}/{video}/{local}: MAE {mae}"
            )
        comparisons.append(
            {
                "file": video,
                "frame": local,
                "source_frame": source,
                "reencoding_mae": mae,
            }
        )
    result = {
        "status": "passed",
        "video_frame_counts": counts,
        "boundary_rgb_comparisons": comparisons,
        "conditions_aligned_to_original_sensor_timestamps": True,
        "real_frames_not_dropped": True,
    }
    (root / "asset_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return {
        "case": root.name,
        "status": "passed",
        "max_boundary_mae": max(r["reencoding_mae"] for r in comparisons),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/evaluation_cases",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            [
                validate(path.parent)
                for path in sorted(args.root.glob("eval_*/case.json"))
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
