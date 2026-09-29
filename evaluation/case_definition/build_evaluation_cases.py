"""Turn saved keyframe annotations into aligned MemWorld evaluation assets."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.case_definition"

import argparse
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import math
import os
import time

import av
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("MEMWORLD_ROOT", "external/MemWorld")).resolve()
STANDARD = Path(os.environ.get("WMDATA_ROOT", "external/evophys-wmdata")).resolve()
ANNOTATIONS = Path(os.environ.get("MEMORYBENCH_ANNOTATIONS", "annotations/episodes"))
PROMPT = "A first-person view of a person interacting with the environment."
sys.path[:0] = [
    str(REPO),
    str(STANDARD / "integrations/reviewers/hand_comparison/src"),
    str(HERE / "support"),
]

def _load_build_dependencies():
    os.environ.setdefault("MEMWORLD_ROOT", str(REPO))
    global convert_c2w_opencv_to_memworld, choose_stream, read_component, PackageReader, geometry, HandTimeline, render_hands
    from dataset.poses import convert_c2w_opencv_to_memworld
    from preprocess.egoquest_wmdata.prepare_smoke import choose_stream, read_component
    from wmdata_hand_review.io import PackageReader
    from wmdata_hand_review import geometry
    from ep_data_to_train_data import HandTimeline, render_hands


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    os.replace(tmp, path)


def layout(marks):
    if "prefix_end" in marks:
        p, pe, t, ee = (marks[k] for k in ("prefix_start", "prefix_end", "t", "eval_end"))
        if not 0 <= p < pe <= t < ee:
            raise ValueError(f"Prefix/generation must be disjoint and ordered: {marks}")
        prefix_count, future_count = pe - p, ee - t
        left, right = (-(prefix_count - 1)) % 32, (-future_count) % 32
        source = [p] * left + list(range(p, pe)) + list(range(t, ee)) + [ee - 1] * right
        anchor = left + prefix_count - 1
        assert anchor % 32 == 0 and (len(source) - 1) % 32 == 0
        assert source[anchor] == pe - 1 and source[anchor + 1] == t
        return {
            "prefix_frame_count": prefix_count, "generation_frame_count": future_count,
            "evaluation_frame_count": future_count, "padding_left_frames": left,
            "padding_right_frames": right, "model_anchor_frame": anchor,
            "inference_start_clip": anchor // 32, "generation_clips": math.ceil(future_count / 32),
            "total_clips": (len(source) - 1) // 32, "model_source_frame_indices": source,
            "prediction_future_slice": [1, 1 + future_count],
            "prediction_evaluation_slice": [1, 1 + future_count],
        }
    p, t, ae, es, ee = (
        marks[k] for k in ("prefix_start", "t", "action_end", "eval_start", "eval_end")
    )
    if not 0 <= p < t < ae <= es < ee:
        raise ValueError(f"Invalid annotation boundaries: {marks}")
    prefix_count = t - p
    future_count = ee - t
    left = (-(prefix_count - 1)) % 32
    right = (-future_count) % 32
    source = [p] * left + list(range(p, ee)) + [ee - 1] * right
    anchor = left + prefix_count - 1
    assert anchor % 32 == 0 and (len(source) - 1) % 32 == 0
    assert source[anchor] == t - 1 and source[anchor + 1] == t
    return {
        "prefix_frame_count": prefix_count,
        "generation_frame_count": future_count,
        "evaluation_frame_count": ee - es,
        "padding_left_frames": left,
        "padding_right_frames": right,
        "model_anchor_frame": anchor,
        "inference_start_clip": anchor // 32,
        "generation_clips": math.ceil(future_count / 32),
        "total_clips": (len(source) - 1) // 32,
        "model_source_frame_indices": source,
        "prediction_future_slice": [1, 1 + future_count],
        "prediction_evaluation_slice": [1 + es - t, 1 + ee - t],
    }


def video_writer(stack, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return stack.enter_context(
        imageio.get_writer(
            str(path),
            fps=30,
            codec="libx264",
            quality=None,
            macro_block_size=1,
            ffmpeg_log_level="error",
            ffmpeg_params=["-preset", "fast", "-crf", "16", "-threads", "2"],
        )
    )


def prepare_one(job):
    annotation, destination, ordinal = job
    began = time.monotonic()
    m = dict(annotation["marks"])
    independent = "prefix_end" in m
    if independent:
        # Evaluation covers the annotated generation; no invented stable/action boundary.
        m.update(action_end=None, eval_start=m["t"])
    pe = m.get("prefix_end", m["t"])
    hand_source = annotation.get("hand_source", "egoquest/legacy-world-landmarks-basis@v1")
    if hand_source not in ("none", "source", "egoquest/legacy-world-landmarks-basis@v1"):
        raise ValueError(f"Unsupported hand_source: {hand_source}")
    prompt = annotation.get("prompt", PROMPT)
    plan = layout(m)
    scene = f"eval_{ordinal:03d}_{annotation['id'][:8]}"
    root = Path(destination) / scene
    if (root / "case.json").exists():
        old = json.loads((root / "annotation.json").read_text())
        if old != annotation:
            raise ValueError(
                f"Annotation changed; use a fresh output directory: {root}"
            )
        return json.loads((root / "case.json").read_text())
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "annotation.json", annotation)
    reader = PackageReader(Path(annotation["source"]["case_path"]))
    # Extent reader accepts legacy string IDs and current numeric WMData IDs.
    reader.files_by_id.update({row["file_id"]: row for row in reader.files})
    rgb = next(
        s for s in reader.streams if s["stream_id"] == annotation["source"]["stream_id"]
    )
    camera, trajectory = rgb["camera_frame_id"], rgb["trajectory_index"]
    pose_stream = choose_stream(reader, "camera_pose/pose9@v1", camera, trajectory)
    calibration_stream = choose_stream(
        reader, "camera_calibration/pinhole@v1", camera, trajectory
    )
    pose_index = geometry.PoseIndex.from_rows(
        read_component(reader, pose_stream, "camera_pose")
    )
    calibrations = geometry.parse_calibrations(
        calibration_stream["schema_id"],
        read_component(reader, calibration_stream, "camera_calibration"),
    )
    cal_times = [c.timestamp_ns for c in calibrations]
    provenance = {
        r["provenance_id"]: r
        for r in pd.read_parquet(reader.root / "meta/provenance.parquet").to_dict(
            "records"
        )
    }
    hands, hand_streams = {}, {}
    for side in ("left", "right"):
        if hand_source == "none":
            hands[side] = None
            continue
        matches = [
            s
            for s in reader.streams
            if s["schema_id"] == "hand/landmarks21@v1"
            and s["trajectory_index"] == trajectory
            and s["side"] == side
            and (
                provenance[s["provenance_id"]]["origin"] in ("native", "exact_conversion")
                if hand_source == "source" else
                provenance[s["provenance_id"]]["method_id"] == hand_source
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one original {side} hand stream, got {len(matches)}"
            )
        hand_streams[side] = matches[0]["stream_id"]
        hands[side] = HandTimeline(
            read_component(reader, matches[0], "hand_landmarks"),
            max_gap_ns=100_000_000,
            max_nearest_ns=20_000_000,
        )
    rgb_rows = reader.rgb_metadata(rgb["stream_id"])
    selected = list(range(m["prefix_start"], pe)) + list(range(m["t"], m["eval_end"]))
    times = np.array([rgb_rows[i]["timestamp_ns"] for i in selected], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Invalid capture timeline")
    poses = []
    points = {
        side: np.full((len(selected), 21, 3), np.nan, dtype=np.float32)
        for side in hands
    }
    valid = {side: np.zeros(len(selected), dtype=bool) for side in hands}
    alignment = Counter()
    for k, timestamp in enumerate(times):
        pose = geometry.resolve_pose(
            pose_index, int(timestamp), max_delta_ns=50_000_000
        )
        if pose is None:
            raise ValueError(
                f"Missing head pose at original frame {selected[k]}; refusing to move the annotation"
            )
        poses.append(pose)
        for side, timeline in hands.items():
            value, method = (None, "absent_by_design") if timeline is None else timeline.at(int(timestamp))
            alignment[
                f'{side}:{method}:{"valid" if value is not None else "missing"}'
            ] += 1
            if value is not None:
                points[side][k] = value
                valid[side][k] = True
    poses = np.stack(poses)
    c2ws = np.stack([convert_c2w_opencv_to_memworld(pose) for pose in poses])
    if not np.isfinite(c2ws).all():
        raise ValueError("Nonfinite head pose")
    width, height = annotation["source"]["width"], annotation["source"]["height"]
    if width % 32 or height % 32:
        raise ValueError(
            f"Keep native resolution only when divisible by 32: {width}x{height}"
        )
    context = root / "context"
    for subdir in ("jsons", "texts", "static_views"):
        (context / subdir).mkdir(parents=True, exist_ok=True)
    local_by_source = {source: i for i, source in enumerate(selected)}
    model_local = np.array([local_by_source[i] for i in plan["model_source_frame_indices"]])
    model_times = times[model_local]
    write_json(
        context / "jsons" / f"{scene}.json",
        {
            "c2ws": c2ws[model_local].tolist(),
            "camera_basis": "memworld",
            "translation_unit": "meter",
            "source_timestamps_ns": model_times.tolist(),
            "source_frame_indices": plan["model_source_frame_indices"],
            "fps": 30,
            "time_policy": "native source frame order; repeat boundary frames only for 32-frame alignment",
        },
    )
    (context / "texts" / f"{scene}.txt").write_text(prompt + "\n")
    write_json(
        context / "static_views" / f"{scene}.json",
        {"selected_source_indices": [plan["padding_left_frames"]]},
    )
    (root / "input").mkdir(exist_ok=True)
    np.savez_compressed(
        root / "input" / "aligned_conditions.npz",
        source_frame_indices=np.array(selected),
        sensor_timestamp_ns=times,
        c2w_opencv=poses,
        c2w_memworld=c2ws,
        **{f"{side}_hand_positions_ref": value for side, value in points.items()},
        **{f"{side}_hand_valid": value for side, value in valid.items()},
        model_source_frame_indices=np.array(plan["model_source_frame_indices"]),
        model_sensor_timestamp_ns=model_times,
    )
    preview_frames = {}
    preview_targets = {
        m["prefix_start"],
        pe - 1,
        m["t"],
        m["eval_start"],
        m["eval_end"] - 1,
    }
    visible_counts = Counter()
    rgb_iter = reader.iter_video_frames(rgb["stream_id"], set(selected))
    try:
        with ExitStack() as stack:
            combined_rgb = video_writer(stack, context / "videos" / f"{scene}.mp4")
            combined_hand = video_writer(
                stack, context / "keypoint_videos" / f"{scene}.mp4"
            )
            prefix_rgb = video_writer(stack, root / "input/rgb_prefix.mp4")
            prefix_hand = video_writer(stack, root / "input/past_hands.mp4")
            future_hand = video_writer(stack, root / "input/future_hands.mp4")
            future_gt = video_writer(stack, root / "ground_truth/future.mp4")
            evaluation_gt = video_writer(stack, root / "ground_truth/evaluation.mp4")
            for k, source_index in enumerate(selected):
                actual, frame = next(rgb_iter)
                if actual != rgb_rows[source_index]["frame_index"]:
                    raise ValueError(f"RGB decoder mismatch {actual} != {source_index}")
                rgb_array = frame.to_ndarray(format="rgb24")
                if rgb_array.shape != (height, width, 3):
                    raise ValueError("Source resolution mismatch")
                ci = bisect_right(cal_times, int(times[k])) - 1
                if ci < 0 or calibrations[ci].status != "valid":
                    raise ValueError("Missing camera calibration")
                calibration = calibrations[ci]
                if (calibration.width, calibration.height) != (width, height):
                    raise ValueError("Calibration / source resolution mismatch")
                hand_array = np.asarray(
                    render_hands(
                        (width, height),
                        poses[k],
                        calibration,
                        {
                            side: points[side][k] if valid[side][k] else None
                            for side in hands
                        },
                        geometry,
                    )
                )
                phase = "prefix" if k < plan["prefix_frame_count"] else "future"
                visible_counts[f"{phase}_frames"] += 1
                visible_counts[f"{phase}_nonempty_hand_frames"] += bool(
                    np.any(hand_array)
                )
                copies = (
                    1
                    + (plan["padding_left_frames"] if k == 0 else 0)
                    + (plan["padding_right_frames"] if k == len(selected) - 1 else 0)
                )
                for _ in range(copies):
                    combined_rgb.append_data(rgb_array)
                    combined_hand.append_data(hand_array)
                if k < plan["prefix_frame_count"]:
                    prefix_rgb.append_data(rgb_array)
                    prefix_hand.append_data(hand_array)
                else:
                    future_hand.append_data(hand_array)
                    future_gt.append_data(rgb_array)
                if source_index >= m["eval_start"]:
                    evaluation_gt.append_data(rgb_array)
                if source_index == pe - 1:
                    Image.fromarray(rgb_array).save(root / "input/anchor.png")
                if source_index in preview_targets:
                    preview_frames[source_index] = (
                        Image.fromarray(rgb_array),
                        Image.fromarray(hand_array),
                    )
    finally:
        rgb_iter.close()
    contact = Image.new("RGB", (320 * len(preview_frames), 260 * 2), "#17212b")
    draw = ImageDraw.Draw(contact)
    for col, (source_index, images) in enumerate(sorted(preview_frames.items())):
        for row, image in enumerate(images):
            image.thumbnail((320, 240))
            contact.paste(image, (col * 320, row * 260 + 20))
            draw.text(
                (col * 320 + 5, row * 260 + 3),
                f'{"RGB" if row == 0 else "hands"} source frame {source_index}',
                fill="white",
            )
    contact.save(root / "preview.jpg")
    np.savez_compressed(
        root / "frame_mapping.npz",
        model_frame=np.arange(len(model_local)),
        source_frame=np.array(plan["model_source_frame_indices"]),
        source_media_pts=np.array(
            [rgb_rows[i]["video_pts"] for i in plan["model_source_frame_indices"]]
        ),
        sensor_timestamp_ns=model_times,
        is_padding=np.array(
            [True] * plan["padding_left_frames"]
            + [False] * len(selected)
            + [True] * plan["padding_right_frames"]
        ),
    )
    durations = {}
    for name, start, end in [
        ("prefix", m["prefix_start"], pe),
        ("generation", m["t"], m["eval_end"]),
        ("evaluation", m["eval_start"], m["eval_end"]),
    ]:
        durations[name] = (
            (rgb_rows[end]["timestamp_ns"] if end < len(rgb_rows) else rgb_rows[-1]["timestamp_ns"] + int(np.median(np.diff([r["timestamp_ns"] for r in rgb_rows])))) - rgb_rows[start]["timestamp_ns"]
        ) / 1e9
    plan.pop("model_source_frame_indices")
    result = {
        "evaluation_case_id": scene,
        "annotation_id": annotation["id"],
        "annotation_revision": annotation["revision"],
        "source_case_id": annotation["case_id"],
        "directory": str(root),
        "marks": m,
        "annotation_scope": annotation.get("annotation_scope", "keyframes"),
        "evaluation_scope": "full_annotated_generation" if independent else "annotated_stable_interval",
        "source_marks": annotation["marks"],
        "assembly": {
            "prefix_source_range": [m["prefix_start"], pe],
            "generation_source_range": [m["t"], m["eval_end"]],
            "case_generation_start_frame": plan["prefix_frame_count"],
            "case_fps": 30,
            "source_gap_frames_excluded": m["t"] - pe,
        },
        "rgb_stream_id": rgb["stream_id"],
        "pose_stream_id": pose_stream["stream_id"],
        "hand_stream_ids": hand_streams,
        "width": width,
        "height": height,
        "fps": 30,
        "sensor_duration_seconds": durations,
        **plan,
        "input_contract": {
            "rgb_prefix": "[prefix_start,prefix_end)" if independent else "[prefix_start,t)",
            "rgb_anchor": "prefix_end-1" if independent else "t-1",
            "future_actions": "[t,eval_end)",
            "future_head_pose": True,
            "rgb_and_hands_separate": True,
            "prompt": prompt,
            "prompt_source": "annotation" if "prompt" in annotation else "fixed generic description; no object answer",
            "hand_alignment": "absent by explicit annotation policy; empty condition raster" if hand_source == "none" else "source sensor timestamps; valid brackets <=100ms, boundary nearest <=20ms",
            "hand_source": hand_source,
            "padding": "repeat first prefix frame on the left and last allowed future condition on the right",
            "future_gt_rgb_is_model_condition": False,
        },
        "hand_alignment_counts": dict(alignment),
        "hand_visibility": dict(visible_counts),
        "packed_index": str(root / "packed/index.json"),
        "preparation_seconds": time.monotonic() - began,
        "status": "assets_ready",
    }
    write_json(root / "case.json", result)
    print(
        json.dumps(
            {
                "case": scene,
                "status": "assets_ready",
                "seconds": result["preparation_seconds"],
            }
        ),
        flush=True,
    )
    return result


def main():
    _load_build_dependencies()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=HERE.parents[1] / "data/evaluation_cases")
    parser.add_argument("--annotation-ids", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-drafts", action="store_true", help="Explicitly include valid user-selected drafts without changing saved status")
    args = parser.parse_args()
    records = {
        r["id"]: r
        for p in args.annotations.glob("*.json")
        if (r := json.loads(p.read_text()))["status"] == "complete" or args.include_drafts
    }
    chosen = [records[key] for key in args.annotation_ids]
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "annotations_snapshot.json", chosen)
    jobs = [(r, str(args.output.resolve()), i + 1) for i, r in enumerate(chosen)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(prepare_one, jobs))
    write_json(
        args.output / "manifest.json",
        {
            "format": "acwm_memory_evaluation_cases_v1",
            "cases": results,
            "case_count": len(results),
            "source_case_count": len({r["case_id"] for r in chosen}),
        },
    )


if __name__ == "__main__":
    main()
