"""Create portable review assets from an existing generation without rerunning ACWM."""

from hashlib import sha256
from pathlib import Path
import json
import shutil
import av
import imageio.v2 as imageio
import numpy as np
from ..ground_truth.schema import MemoryAnnotation
from ..common.vlm import write_json


def digest(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def crop_video(source, destination, start, stop):
    if not 0 <= start < stop:
        raise ValueError("Nonempty half-open action interval required")
    destination = Path(destination)
    temporary = destination.with_name(destination.stem + ".pending.mp4")
    count = 0
    try:
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            with imageio.get_writer(
                str(temporary),
                fps=float(stream.average_rate),
                codec="libx264",
                macro_block_size=1,
                quality=None,
                ffmpeg_log_level="error",
                ffmpeg_params=["-crf", "16", "-threads", "2"],
            ) as writer:
                for i, frame in enumerate(container.decode(video=0)):
                    if i >= stop:
                        break
                    if i >= start:
                        writer.append_data(np.asarray(frame.to_image()))
                        count += 1
        if count != stop - start:
            raise ValueError(f"Action crop got {count} frames, expected {stop-start}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return count



def action_frame_count(case):
    """Use the explicit action interval, or review all generation for four-boundary cases."""
    end = case["marks"].get("action_end")
    if end is None:
        if case.get("evaluation_scope") != "full_annotated_generation":
            raise ValueError("Missing action boundary without an explicit full-generation policy")
        return case["generation_frame_count"]
    return end - case["marks"]["t"]


def prepare_bundle(source, case_directory, output):
    source, case_directory, output = map(Path, (source, case_directory, output))
    case = json.loads((source / "case.json").read_text())
    annotation = MemoryAnnotation.model_validate(
        json.loads((case_directory / "memory_annotation.json").read_text())
    )
    if annotation.evaluation_case_id != case["evaluation_case_id"]:
        raise ValueError("Manual annotation and video case ID disagree")
    output.mkdir(parents=True, exist_ok=True)
    names = [
        "prefix.mp4",
        "anchor.png",
        "past_hands.mp4",
        "future_hands.mp4",
        "gt_future.mp4",
        "generated_future.mp4",
        "gt_evaluation.mp4",
        "generated_evaluation.mp4",
        "comparison.mp4",
        "comparison.jpg",
        "annotation.json",
        "frame_mapping.npz",
        "inference_command.json",
        "inference.log",
    ]
    provenance = {}
    for name in names:
        origin = source / name
        if not origin.exists():
            if name in ("inference_command.json", "inference.log"):
                continue
            raise FileNotFoundError(origin)
        shutil.copy2(origin, output / name)
        provenance[name] = {"source": str(origin.resolve()), "sha256": digest(origin)}
    write_json(output / "case.json", case)
    write_json(output / "memory_annotation.json", annotation.model_dump())
    count = action_frame_count(case)
    for role in ("gt", "generated"):
        crop_video(
            output / f"{role}_future.mp4", output / f"{role}_action.mp4", 0, count
        )
    write_json(
        output / "input_provenance.json",
        {
            "source_run": str(source.parent.resolve()),
            "copied_assets": provenance,
            "action_source_interval": [0, count],
            "action_interval_policy": case.get("evaluation_scope", "annotated_stable_interval"),
            "memory_source_interval": [
                case["marks"]["eval_start"] - case["marks"]["t"],
                case["marks"]["eval_end"] - case["marks"]["t"],
            ],
            "boundary_convention": "zero-based [start,end)",
            "annotation_source": str(case_directory.resolve()),
        },
    )
    return case, annotation
