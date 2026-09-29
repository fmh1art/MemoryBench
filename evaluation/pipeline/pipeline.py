"""v2: explicit annotation -> six-frame action Gate -> direct two-image judgments."""

import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from .. import __version__
from ..metric.schemas import GateJudgment, PairJudgment, METRICS
from ..ground_truth.schema import MemoryAnnotation
from ..metric.scoring import aggregate, metric_table, score_pair
from ..common.video import Video, uniform_frames
from ..common.vlm import VLM, EvaluationError, write_json


def pair_frames(gt_frames, gen_frames):
    """Use every sampled image; unequal short clips reuse the nearest progress partner."""
    count = max(len(gt_frames), len(gen_frames))
    if not count or not gt_frames or not gen_frames:
        raise ValueError("Both videos must contain at least one frame")
    pairs = []
    for i in range(count):
        progress = i / (count - 1) if count > 1 else 0.0

        def nearest(frames):
            if len(frames) == 1:
                return frames[0]
            span = frames[-1].time - frames[0].time
            return min(
                frames, key=lambda f: abs((f.time - frames[0].time) / span - progress)
            )

        # Use the longer side exactly once even for irregular timestamps.
        gt = gt_frames[i] if len(gt_frames) == count else nearest(gt_frames)
        gen = gen_frames[i] if len(gen_frames) == count else nearest(gen_frames)
        pairs.append({"progress": progress, "gt": gt, "generated": gen})
    # Nearest-time selection can skip short-side frames on irregular clips.
    # Reserve one distinct progress slot for each short-side image, preserving order.
    for role, frames in [("gt", gt_frames), ("generated", gen_frames)]:
        if len(frames) < count:
            slots = [
                round(i * (count - 1) / (len(frames) - 1)) if len(frames) > 1 else 0
                for i in range(len(frames))
            ]
            for slot, frame in zip(slots, frames):
                pairs[slot][role] = frame
            indices = sorted(p[role].index for p in pairs)
            by_index = {f.index: f for f in frames}
            for p, index in zip(pairs, indices):
                p[role] = by_index[index]
    return pairs


def evaluate(
    gt_path,
    gen_path,
    config,
    output,
    *,
    action_path,
    annotation_path,
    progress=print,
    transport=None,
):
    annotation = MemoryAnnotation.model_validate(
        json.loads(Path(annotation_path).read_text())
    )
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {
        "schema_version": "acwm.memory_evaluation.v2",
        "framework_version": __version__,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "memory_score": None,
        "config": config.public(),
        "annotation": annotation.model_dump(),
        "weights": config.weights.model_dump(),
        "warnings": [],
    }
    write_json(output / "config.redacted.json", config.public())
    write_json(output / "memory_annotation.json", annotation.model_dump())
    rubric_dir = Path(__file__).resolve().parents[1] / "prompt"
    (output / "rubrics").mkdir()
    for name in ("common.md", "action_gate.md", "pair_judge.md"):
        shutil.copy2(rubric_dir / name, output / "rubrics" / name)
    client = VLM(config.api, output, transport=transport)
    try:
        gt = Video(Path(gt_path), "gt", output, config.sampling.image_max_side)
        gen = Video(Path(gen_path), "gen", output, config.sampling.image_max_side)
        action = Video(
            Path(action_path), "action", output, config.sampling.image_max_side
        )
        result["videos"] = {v.role: v.metadata() for v in (gt, gen, action)}
        gt_frames = uniform_frames(gt.frames, config.sampling.frames_per_video)
        gen_frames = uniform_frames(gen.frames, config.sampling.frames_per_video)
        action_frames = uniform_frames(action.frames, config.sampling.action_frames)
        pairs = pair_frames(gt_frames, gen_frames)
        selection = {
            "gt_images": [f.as_dict() for f in gt_frames],
            "gen_images": [f.as_dict() for f in gen_frames],
            "gen_action_images": [f.as_dict() for f in action_frames],
            "pairs": [
                {
                    "progress": p["progress"],
                    "gt": p["gt"].as_dict(),
                    "generated": p["generated"].as_dict(),
                }
                for p in pairs
            ],
            "pairing_policy": "chronological normalized progress; all unique samples used; shorter side may be reused when counts differ",
        }
        if len(gt_frames) != len(gen_frames):
            result["warnings"].append(
                "Unequal sampled frame counts: the shorter side is reused for pairing; this does not add independent visual evidence."
            )
        for video, frames in [
            (gt, gt_frames),
            (gen, gen_frames),
            (action, action_frames),
        ]:
            for frame in frames:
                video.image(frame)
        write_json(output / "selected_frames.json", selection)
        result["selected_frames"] = selection
        progress(
            f"Action Gate: {annotation.read_action}, {len(action_frames)} ordered generated action frames"
        )

        def validate_gate(value):
            if not set(value.evidence_frame_ids) <= {f.id for f in action_frames}:
                raise ValueError(
                    "Gate evidence must reference supplied action frame IDs"
                )

        gate_judgment = client.ask(
            "action_gate",
            {
                "annotation": annotation.model_dump(),
                "requested_frames": [f.as_dict() for f in action_frames],
            },
            [
                (
                    f"GENERATED ACTION FRAME {i+1}/{len(action_frames)}: {f.id}, {f.time:.6f}s",
                    action.image(f),
                )
                for i, f in enumerate(action_frames)
            ],
            GateJudgment,
            validate_gate,
        )
        gate = int(gate_judgment.action_correct and gate_judgment.target_visible)
        write_json(output / "gate.json", gate_judgment.model_dump() | {"gate": gate})
        result["gate_evidence"] = gate_judgment.model_dump()
        rows = []
        if gate:
            for i, p in enumerate(pairs):
                progress(f"Direct image-pair judge {i+1}/{len(pairs)}")
                judgment = client.ask(
                    "pair_judge",
                    {
                        "annotation": annotation.model_dump(),
                        "pair_index": i + 1,
                        "pair_count": len(pairs),
                        "progress": p["progress"],
                        "gt_frame": p["gt"].as_dict(),
                        "generated_frame": p["generated"].as_dict(),
                    },
                    [
                        (
                            "IMAGE 1 — GROUND TRUTH stable evaluation frame",
                            gt.image(p["gt"]),
                        ),
                        (
                            "IMAGE 2 — GENERATED stable evaluation frame",
                            gen.image(p["generated"]),
                        ),
                    ],
                    PairJudgment,
                )
                if not judgment.gt_readable:
                    raise EvaluationError(
                        f"Pair {i+1}: GT reference unreadable; fix the annotation/reference rather than assign model zero"
                    )
                scores, adjustments = score_pair(judgment)
                row = {
                    "pair_index": i + 1,
                    "progress": p["progress"],
                    "gt_frame_id": p["gt"].id,
                    "gen_frame_id": p["generated"].id,
                    "scores": scores,
                    "arithmetic_adjustments": adjustments,
                    "judgment": judgment.model_dump(),
                }
                rows.append(row)
                write_json(output / "pairs" / f"pair_{i+1:02d}.json", row)
        result.update(aggregate(rows, gate, config.weights.model_dump()))
        result.update(
            status="completed",
            frame_results=rows,
            requested_frame_pairs=len(pairs),
            evaluated_frame_pairs=len(rows),
        )
        (output / "metrics.md").write_text(metric_table(result))
        with (output / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["evaluation_case_id", "gate", *METRICS, "memory_score"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "evaluation_case_id": annotation.evaluation_case_id,
                    "gate": gate,
                    **result["metrics"],
                    "memory_score": result["memory_score"],
                }
            )
        return result
    except Exception as exc:
        detail = str(exc)
        if config.api.key():
            detail = detail.replace(config.api.key(), "[REDACTED]")
        result.update(status="error", error=detail, memory_score=None)
        raise
    finally:
        client.close()
        result["elapsed_seconds"] = time.monotonic() - started
        result["vlm_call_count"] = len(client.calls)
        result["vlm_request_count"] = sum(
            len(call["attempts"]) for call in client.calls
        )
        write_json(output / "result.json", result)
