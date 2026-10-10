"""CLI: python -m evaluation.run --cases cases.json --endpoint ... --model ... --output ..."""

import argparse
import asyncio
import hashlib
import json
import os
from math import isfinite
from pathlib import Path

from .api_config import add_api_arguments, resolve_api_arguments
from .appearance.judge import judge_appearance
from .judge import Judge, write_json
from .schema import Case
from .scoring import (
    MEMORY_DIMENSIONS,
    METRIC_NAMES,
    SCORING_VERSION,
    WEIGHTS,
    average_memory_frames,
    score_memory,
    select_metrics,
    summarize,
)
from .video import MEMORY_FRACTIONS, make_memory_pairs, make_sheets


def load_cases(path: Path) -> list[Case]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("--cases must contain a non-empty JSON array")
    cases = [
        Case.model_validate(value).resolve_videos(path.resolve().parent)
        for value in data
    ]
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("case_id values must be unique")
    return cases


async def evaluate_case(
    case,
    judge,
    output,
    *,
    tile_width=480,
    tile_height=320,
    max_skew_seconds=0.1,
    resume=False,
    selected_metrics=None,
):
    selected_metrics = (
        select_metrics(metrics=selected_metrics)
        if selected_metrics is not None
        else select_metrics()
    )
    dimensions = tuple(
        dim
        for dim in MEMORY_DIMENSIONS
        if any(key.startswith(dim + ".") for key in selected_metrics)
    )
    output.mkdir(parents=True, exist_ok=resume)
    calls = output / "judges"
    calls.mkdir(exist_ok=resume)
    result = {
        "case_id": case.case_id,
        "case": case.model_dump(),
        "status": "error",
        "gate": None,
        "memory_score": None,
        "protocol": "memory_evaluation.frame_pairs.v3",
        "aggregation_policy": "skip_reference_unavailable_score_generated_failure_zero",
        "case_aggregation_policy": "all_cases_gate_failed_zero",
        "scoring_version": SCORING_VERSION,
        "selected_metrics": list(selected_metrics),
    }
    try:
        gt_image, gen_image, sampling = await asyncio.to_thread(
            make_sheets, case, output, tile_width, tile_height, max_skew_seconds
        )
        result["sampling"] = sampling
        write_json(output / "sampling.json", sampling)
        context = {"case": case.model_dump(), "sampling": sampling}
        images = (gt_image, gen_image)
        gate = await judge.ask("gate", context, images, calls)
        result["gate"] = gate.model_dump()
        passed = gate.observability.label == 1 and gate.action_following.label == 1
        if not passed:
            result["status"] = "gate_failed"
            result["memory_score"] = 0.0
        else:
            memory_sampling = await asyncio.to_thread(
                make_memory_pairs, case, output, sampling
            )
            result["memory_sampling"] = memory_sampling
            write_json(output / "memory_sampling.json", memory_sampling)

            async def evaluate_frame(frame):
                index = frame["frame_index"]
                frame_calls = calls / f"frame_{index}"
                frame_calls.mkdir(exist_ok=resume)
                frame_context = {
                    "case": case.model_dump(),
                    "selected_metrics": list(selected_metrics),
                    "memory_frame": frame,
                    "sampling": {"read_frame_indices": [index], "frames": [frame]},
                }
                images = tuple(
                    output / "memory_frames" / f"{index}_{role}.jpg"
                    for role in ("gt", "gen")
                )
                inventory_task = (
                    asyncio.create_task(
                        judge.ask("inventory", frame_context, images, frame_calls)
                    )
                    if "inventory" in dimensions or "appearance" in dimensions
                    else None
                )

                async def ask_dimension(dimension):
                    if dimension == "inventory":
                        return await inventory_task
                    if dimension != "appearance":
                        return await judge.ask(
                            dimension, frame_context, images, frame_calls
                        )
                    return await judge_appearance(
                        judge, await inventory_task, frame_context, images, frame_calls
                    )

                replies = await asyncio.gather(
                    *(ask_dimension(dim) for dim in dimensions),
                    return_exceptions=True,
                )
                errors = {
                    dim: str(reply)
                    for dim, reply in zip(dimensions, replies)
                    if isinstance(reply, Exception)
                }
                frame_result = {
                    **frame,
                    "status": "error",
                    "memory_score": None,
                    "judgments": {
                        dim: reply.model_dump()
                        for dim, reply in zip(dimensions, replies)
                        if not isinstance(reply, Exception)
                    },
                }
                if errors:
                    frame_result["errors"] = errors
                else:
                    scores = score_memory(
                        dict(zip(dimensions, replies)), selected_metrics
                    )
                    frame_result.update(
                        status=scores["status"],
                        memory_score=scores["memory_score"],
                        scores=scores,
                    )
                return frame_result

            frames = await asyncio.gather(
                *(evaluate_frame(frame) for frame in memory_sampling["frames"])
            )
            result["memory_frames"] = frames
            errors = {
                f"frame_{frame['frame_index']}": frame["errors"]
                for frame in frames
                if frame["status"] == "error"
            }
            if errors:
                result["errors"] = errors
            else:
                scores = average_memory_frames(frames)
                result.update(
                    status=scores["status"],
                    memory_score=scores["memory_score"],
                    scores=scores,
                )

    except Exception as exc:  # noqa: BLE001 -- persist isolated case failures; keep batch running
        result["error"] = str(exc)
    write_json(output / "result.json", result)
    print(
        f"{case.case_id}: {result['status']} memory_score={result['memory_score']}",
        flush=True,
    )
    return result


async def run(args):
    cases = load_cases(args.cases)
    selected_metrics = select_metrics(args.dimensions, args.metrics)
    extra_body = json.loads(args.extra_body.read_text()) if args.extra_body else {}
    if not isinstance(extra_body, dict):
        raise ValueError("--extra-body must contain a JSON object")  # noqa: TRY004 -- CLI input error
    extra_body = {**getattr(args, "api_extra_body", {}), **extra_body}
    judge = Judge(
        args.endpoint,
        args.model,
        os.environ.get(args.api_key_env, ""),
        args.concurrency,
        args.timeout,
        args.retries,
        args.max_tokens,
        extra_body,
        temperature=None if args.use_default_temperature else 0,
        qpm=args.qpm,
        tpm=args.tpm,
        tokens_per_request=args.tokens_per_request,
    )
    try:
        # Refuse overwrites to keep runs auditable. Create only after validating all case inputs.
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(
            args.output / "run.json",
            {
                "protocol": "memory_evaluation.frame_pairs.v3",
                "aggregation_policy": "skip_reference_unavailable_score_generated_failure_zero",
                "case_aggregation_policy": "all_cases_gate_failed_zero",
                "scoring_version": SCORING_VERSION,
                "selected_metrics": list(selected_metrics),
                "weights": WEIGHTS,
                "rubric_sha256": {
                    name: hashlib.sha256(text.encode()).hexdigest()
                    for name, text in {
                        "common": judge.common,
                        "memory_common": judge.memory_common,
                        **judge.rubrics,
                    }.items()
                },
                "geometric_zero_policy": "exact_zero; positive values evaluated with log/exp",
                "missing_score_policy": "skip frame nulls; generated failures zero; whole-dimension failure zero; unresolved enabled dimensions block positive case scores",
                "whole_dimension_null_policy": "zero only for confirmed generated failure or absent matches; reference-unavailable/unknown stays unresolved",
                "benchmark_completion_policy": "publish headline only when every case has a determined score; rejected Gates contribute zero",
                "case_formula": "geometric_mean_of_temporally_averaged_core_dimensions",
                "appearance_scope": "inventory_matched_instances_only",
                "memory_sampling_fractions": list(MEMORY_FRACTIONS),
                "memory_image_max_side": 1024,
                "model": args.model,
                "concurrency": args.concurrency,
                "case_workers": args.case_workers,
                "timeout": args.timeout,
                "retries": args.retries,
                "max_tokens": args.max_tokens,
                "temperature": None if args.use_default_temperature else 0,
                "qpm": args.qpm,
                "tpm": args.tpm,
                "tokens_per_request": args.tokens_per_request,
                "tile_width": args.tile_width,
                "tile_height": args.tile_height,
                "max_skew_seconds": args.max_skew_seconds,
                "cases": [case.model_dump() for case in cases],
            },
        )
        case_limit = asyncio.Semaphore(args.case_workers)

        async def one(case):
            async with case_limit:
                return await evaluate_case(
                    case,
                    judge,
                    args.output / case.case_id,
                    tile_width=args.tile_width,
                    tile_height=args.tile_height,
                    max_skew_seconds=args.max_skew_seconds,
                    selected_metrics=selected_metrics,
                )

        results = await asyncio.gather(*(one(case) for case in cases))
        summary = summarize(results)
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if summary["error_cases"] else 0
    finally:
        await judge.close()


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="JSON array; paths relative to this file",
    )
    add_api_arguments(parser)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory; cannot exist"
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        default=4,
        help="Global limit on in-flight VLM calls",
    )
    parser.add_argument(
        "--case-workers",
        type=positive_int,
        default=2,
        help="Concurrent cases/video decoders",
    )
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Additional attempts after transport/JSON/schema failures",
    )
    parser.add_argument("--max-tokens", type=positive_int, default=4096)
    parser.add_argument(
        "--use-default-temperature",
        action="store_true",
        help="Omit temperature for models that reject temperature=0",
    )
    parser.add_argument(
        "--qpm", type=positive_int, help="Global requests per minute budget"
    )
    parser.add_argument(
        "--tpm", type=positive_int, help="Global reserved tokens per minute budget"
    )
    parser.add_argument(
        "--tokens-per-request",
        type=positive_int,
        default=20000,
        help="Conservative input+output token reservation for TPM pacing",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--dimensions",
        nargs="+",
        choices=MEMORY_DIMENSIONS,
        help="Enabled memory dimensions; default: inventory appearance spatial",
    )
    selection.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_NAMES,
        help="Only these submetrics; overrides dimension defaults",
    )
    parser.add_argument("--tile-width", type=positive_int, default=480)
    parser.add_argument("--tile-height", type=positive_int, default=320)
    parser.add_argument("--max-skew-seconds", type=float, default=0.1)
    parser.add_argument(
        "--extra-body",
        type=Path,
        help="Optional provider-specific request fields as JSON",
    )
    args = parser.parse_args()
    if (
        args.tile_width < 420
        or args.tile_height < 64
        or not isfinite(args.max_skew_seconds)
        or args.max_skew_seconds < 0
    ):
        parser.error(
            "tile-width>=420, tile-height>=64, max-skew-seconds>=0 are required"
        )
    try:
        resolve_api_arguments(args)
        return asyncio.run(run(args))
    except (ValueError, OSError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
