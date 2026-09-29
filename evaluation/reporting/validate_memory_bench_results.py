#!/usr/bin/env python3
"""Validate completed physical videos, rollout settings, copied assets and score math."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.reporting"
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
import json

import av
import numpy as np

from ..common.vlm import write_json


from .. import __version__
from ..metric.schemas import PairJudgment
from ..metric.scoring import aggregate, score_pair
from ..common.video import Video, uniform_frames


def validate_run(root):
    root = Path(root)
    status = json.loads((root / "status.json").read_text())
    if status["status"] != "completed":
        raise ValueError("Cannot validate an incomplete/error run")

    def check(row):
        d = root / row["evaluation_case_id"]
        c = json.loads((d / "case.json").read_text())
        p = json.loads((d / "input_provenance.json").read_text())
        r = json.loads((d / "evaluation/result.json").read_text())
        assert r["status"] == "completed" and r["framework_version"] == __version__
        recomputed = aggregate(r["frame_results"], r["gate"], r["weights"])
        for key in ("metrics", "memory_score", "effective_weights"):
            assert r[key] == recomputed[key], (d.name, key)
        assert row["memory_score"] == r["memory_score"]
        for name, item in p["copied_assets"].items():
            assert digest(d / name) == item["sha256"] == digest(item["source"]), (
                d.name,
                name,
                "copy mismatch",
            )
        for pair in r["frame_results"]:
            judgment = PairJudgment.model_validate(pair["judgment"])
            assert pair["scores"] == score_pair(judgment)[0]
        counts = {
            "prefix.mp4": c["prefix_frame_count"],
            "past_hands.mp4": c["prefix_frame_count"],
            "future_hands.mp4": c["generation_frame_count"],
            "gt_future.mp4": c["generation_frame_count"],
            "generated_future.mp4": c["generation_frame_count"],
            "comparison.mp4": c["generation_frame_count"],
            "gt_evaluation.mp4": c["evaluation_frame_count"],
            "generated_evaluation.mp4": c["evaluation_frame_count"],
            "gt_action.mp4": p["action_source_interval"][1],
            "generated_action.mp4": p["action_source_interval"][1],
        }
        for name, count in counts.items():
            with av.open(str(d / name)) as container:
                assert float(container.streams.video[0].average_rate) == 30
                frames = 0
                last = None
                for f in container.decode(video=0):
                    assert last is None or f.pts > last
                    last = f.pts
                    frames += 1
                assert frames == count, (d.name, name, frames, count)
        sel = r["selected_frames"]
        for role, name, key, n in [
            ("gt", "gt_evaluation.mp4", "gt_images", 8),
            ("gen", "generated_evaluation.mp4", "gen_images", 8),
            ("action", "generated_action.mp4", "gen_action_images", 6),
        ]:
            v = Video(d / name, role, d / "evaluation", 768)
            assert r["videos"][role]["sha256"] == digest(d / name)
            assert sel[key] == [x.as_dict() for x in uniform_frames(v.frames, n)]
        calls = [
            json.loads(f.read_text())
            for f in sorted((d / "evaluation/vlm_calls").glob("*.json"))
        ]
        assert (
            len(calls)
            == (1 + len(sel["pairs"]) if r["gate"] else 1)
            == r["vlm_call_count"]
        )
        for i, call in enumerate(calls):
            assert (
                len(call["images"]) <= 8 and call["attempts"][-1]["http_status"] == 200
            )
            assert call["attempts"][-1].get("parsed") is not None
            if i == 0:
                assert call["task"] == "action_gate" and len(call["images"]) == len(
                    sel["gen_action_images"]
                )
                assert all("/frames/action/" in x["path"] for x in call["images"])
            else:
                assert call["task"] == "pair_judge" and len(call["images"]) == 2
                pair = sel["pairs"][i - 1]
                assert [Path(x["path"]).stem for x in call["images"]] == [
                    pair["gt"]["frame_id"],
                    pair["generated"]["frame_id"],
                ]
        return {
            "evaluation_case_id": d.name,
            "status": "passed",
            "video_frame_counts": counts,
            "vlm_calls": len(calls),
            "max_images_per_query": max(len(x["images"]) for x in calls),
            "scores_recomputed": True,
            "sampled_frames_verified": True,
            "copied_assets_verified": True,
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        cases = list(pool.map(check, status["cases"]))
    result = {
        "status": "passed",
        "framework_version": __version__,
        "case_count": len(cases),
        "cases": cases,
        "scope": "Protocol, assets, sampling, schemas and arithmetic; not a certificate of VLM semantic correctness",
    }
    write_json(root / "validation.json", result)
    return result


def legacy_aggregate(root, frame_results, gate):
    """Use the historical formula frozen with the historical experiment."""
    import importlib
    import importlib.util
    import sys

    snapshot = root / "code_snapshot/memory_eval"
    if not (snapshot / "scoring.py").is_file():
        raise ValueError(f"Historical scoring snapshot missing: {snapshot}")
    package = (
        "_memory_eval_legacy_"
        + sha256(str(snapshot.resolve()).encode()).hexdigest()[:12]
    )
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package,
            snapshot / "__init__.py",
            submodule_search_locations=[str(snapshot)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return importlib.import_module(package + ".scoring").aggregate(frame_results, gate)


def digest(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_case(output):
    case = json.loads((output / "case.json").read_text())
    cid = case["evaluation_case_id"]
    counts = {
        "prefix.mp4": case["prefix_frame_count"],
        "past_hands.mp4": case["prefix_frame_count"],
        "future_hands.mp4": case["generation_frame_count"],
        "gt_future.mp4": case["generation_frame_count"],
        "generated_future.mp4": case["generation_frame_count"],
        "comparison.mp4": case["generation_frame_count"],
        "gt_evaluation.mp4": case["evaluation_frame_count"],
        "generated_evaluation.mp4": case["evaluation_frame_count"],
    }
    videos = {}
    for name, expected in counts.items():
        with av.open(str(output / name)) as c:
            stream = c.streams.video[0]
            assert float(stream.average_rate) == 30, (cid, name, "fps")
            previous = None
            n = 0
            for frame in c.decode(video=0):
                assert frame.pts is not None and (
                    previous is None or frame.pts > previous
                )
                previous = frame.pts
                if name != "comparison.mp4":
                    assert (frame.width, frame.height) == (
                        case["width"],
                        case["height"],
                    )
                n += 1
            assert n == expected, (cid, name, n, expected)
            videos[name] = {
                "frames": n,
                "fps": 30,
                "width": stream.width,
                "height": stream.height,
            }
    source = Path(case["directory"])
    for copied, original in [
        ("prefix.mp4", "input/rgb_prefix.mp4"),
        ("gt_future.mp4", "ground_truth/future.mp4"),
        ("gt_evaluation.mp4", "ground_truth/evaluation.mp4"),
        ("frame_mapping.npz", "frame_mapping.npz"),
    ]:
        assert digest(output / copied) == digest(source / original), (
            cid,
            copied,
            "copy checksum",
        )
    assert digest(output / "gt_future.mp4") != digest(output / "generated_future.mp4")
    meta = json.loads((output / "raw" / f"{cid}.json").read_text())
    for key, expected in {
        "teacher_forced_rgb": False,
        "use_keypoint_condition": True,
        "zero_future_keypoints": False,
        "use_long_context": True,
        "use_dummy_pose": False,
        "overlay_keypoints": False,
        "num_inference_steps": 20,
        "cfg_scale": 5.0,
    }.items():
        assert meta[key] == expected, (cid, key, meta[key])
    start = case["inference_start_clip"]
    assert meta["clip_indices"] == list(range(start, start + case["generation_clips"]))
    assert meta["clip_start_frames"][0] == case["model_anchor_frame"]
    trace = json.loads(
        (output / "raw/memory_traces" / cid / f"clip{start:04d}.json").read_text()
    )
    mapping = np.load(output / "frame_mapping.npz")["source_frame"]
    checked = 0
    for pool in trace["pools"].values():
        for slot in pool["slots"]:
            index = slot.get("source_frame")
            if index is not None:
                assert 0 <= index <= case["model_anchor_frame"], (cid, slot)
                original = int(mapping[index])
                assert case["marks"]["prefix_start"] <= original < case["marks"]["t"], (
                    cid,
                    slot,
                    original,
                )
                checked += 1
    result = json.loads((output / "evaluation/result.json").read_text())
    assert result["status"] == "completed", (cid, result.get("error"))
    assert result["framework_version"] == "1.0.1"
    recomputed = legacy_aggregate(
        output.parent, result["frame_results"], result["gate"]
    )
    assert result["metrics"] == recomputed["metrics"]
    assert result["memory_score"] == recomputed["memory_score"]
    assert digest(output / "result.json") == digest(output / "evaluation/result.json")
    for role, name in [("gt", "gt_evaluation.mp4"), ("gen", "generated_future.mp4")]:
        assert result["videos"][role]["sha256"] == digest(output / name)
    calls = list((output / "evaluation/vlm_calls").glob("*.json"))
    assert len(calls) == result["vlm_call_count"] > 0
    for p in calls:
        call = json.loads(p.read_text())
        last = call["attempts"][-1]
        assert last["http_status"] == 200 and last.get("parsed") is not None, (
            cid,
            p.name,
        )
        if call["task"] in ("localize", "inventory"):
            requested = call["input"]["requested_frames"]
            assert len(call["images"]) == len(requested)
            assert set(call["input"]["target"]) == {
                "description",
                "read_action",
                "visibility_criterion",
            }
            for frame, image in zip(requested, call["images"], strict=True):
                assert Path(image["path"]).stem == frame["frame_id"]
                assert image["label"].startswith("FRAME " + frame["frame_id"] + " ")
    return {
        "evaluation_case_id": cid,
        "status": "passed",
        "videos": videos,
        "checked_initial_memory_slots": checked,
        "gate": result["gate"],
        "memory_score": result["memory_score"],
        "vlm_call_count": result["vlm_call_count"],
        "vlm_request_count": result["vlm_request_count"],
        "copied_assets_sha256_match": True,
        "score_recomputed": True,
        "offline_history_seed": meta["offline_history_seed"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "evaluation_results",
    )
    args = p.parse_args()
    status = json.loads((args.root / "status.json").read_text())
    assert (
        status["status"] == "completed"
        and status["completed_count"] == status["case_count"]
    )
    if str(status.get("framework_version", "")).startswith("2."):
        result = validate_run(args.root)
        print(
            json.dumps({"status": result["status"], "case_count": result["case_count"]})
        )
        return
    with ThreadPoolExecutor(max_workers=5) as pool:
        cases = list(pool.map(check_case, sorted(args.root.glob("eval_*"))))
    latencies = {
        r["evaluation_case_id"]: (
            datetime.fromisoformat(r["evaluation_started_at"])
            - datetime.fromisoformat(r["video_ready_at"])
        ).total_seconds()
        for r in status["cases"]
    }
    # A restarted evaluation retains the original inference readiness timestamps.
    assert all(t >= 0 for t in latencies.values()), latencies
    report = {
        "status": "passed",
        "case_count": len(cases),
        "cases": cases,
        "video_ready_to_evaluation_worker_seconds": latencies,
        "evaluation_concurrency": status["case_count"],
        "checks": [
            "physical frame counts and increasing PTS",
            "copied asset checksums",
            "exact annotated rollout start and non-teacher-forced conditions",
            "initial memory source frames within observed prefix",
            "independent generated video",
            "real VLM responses and score arithmetic",
            "automatic evaluation after video creation",
        ],
    }
    write_json(args.root / "validation.json", report)
    print(
        json.dumps(
            {"status": "passed", "case_count": len(cases), "latencies": latencies}
        )
    )


if __name__ == "__main__":
    main()
