"""CLI: python -m evaluation.run --cases cases.json --endpoint ... --model ... --output ..."""

import argparse
import asyncio
import json
import os
from math import isfinite
from pathlib import Path

from .judge import Judge, write_json
from .schema import Case
from .scoring import MEMORY_DIMENSIONS, score_memory, summarize
from .video import make_sheets


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
    case, judge, output, *, tile_width=480, tile_height=320, max_skew_seconds=0.1
):
    output.mkdir(parents=True, exist_ok=False)
    calls = output / "judges"
    calls.mkdir()
    result = {
        "case_id": case.case_id,
        "case": case.model_dump(),
        "status": "error",
        "gate": None,
        "memory_score": None,
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
        else:
            # All four finish/log even if one fails; no partial result becomes a valid score.
            replies = await asyncio.gather(
                *(judge.ask(dim, context, images, calls) for dim in MEMORY_DIMENSIONS),
                return_exceptions=True,
            )
            errors = {
                dim: str(reply)
                for dim, reply in zip(MEMORY_DIMENSIONS, replies)
                if isinstance(reply, Exception)
            }
            result["judgments"] = {
                dim: reply.model_dump()
                for dim, reply in zip(MEMORY_DIMENSIONS, replies)
                if not isinstance(reply, Exception)
            }
            if errors:
                result["errors"] = errors
            else:
                scores = score_memory(dict(zip(MEMORY_DIMENSIONS, replies)))
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
    extra_body = json.loads(args.extra_body.read_text()) if args.extra_body else {}
    if not isinstance(extra_body, dict):
        raise ValueError("--extra-body must contain a JSON object")  # noqa: TRY004 -- CLI input error
    judge = Judge(
        args.endpoint,
        args.model,
        os.environ.get(args.api_key_env, ""),
        args.concurrency,
        args.timeout,
        args.retries,
        args.max_tokens,
        extra_body,
    )
    try:
        # Refuse overwrites to keep runs auditable. Create only after validating all case inputs.
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(
            args.output / "run.json",
            {
                "protocol": "memory_evaluation.contact_sheets.v1",
                "model": args.model,
                "concurrency": args.concurrency,
                "case_workers": args.case_workers,
                "timeout": args.timeout,
                "retries": args.retries,
                "max_tokens": args.max_tokens,
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
    parser.add_argument(
        "--endpoint", required=True, help="Full compatible /v1/chat/completions URL"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory; cannot exist"
    )
    parser.add_argument("--api-key-env", default="MEM_BENCH_API_KEY")
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
        return asyncio.run(run(args))
    except (ValueError, OSError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
