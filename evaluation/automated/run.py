"""Run without VLM calls: python -m evaluation.automated.run --help."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .metrics import METRICS, LearnedMetrics, compute
from .video import sample

VERSION = "memorybench.automated.v1"
LEARNED = None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def evaluate(job):
    source, output, settings, resume = job
    start = time.monotonic()
    record = json.loads(source.read_text())
    cid, case = record["case_id"], record["case"]
    if cid != source.parent.name:
        raise ValueError(f"Case identity mismatch: {source}")
    destination = output / cid / "automated.json"
    fingerprints = {
        key: {
            "path": str(Path(case[key]).resolve()),
            "size": Path(case[key]).stat().st_size,
            "mtime_ns": Path(case[key]).stat().st_mtime_ns,
        }
        for key in ("gt_video", "gen_video")
        if Path(case[key]).is_file()
    }
    signature = hashlib.sha256(
        json.dumps([VERSION, case, settings, fingerprints], sort_keys=True).encode()
    ).hexdigest()
    if resume and destination.exists():
        existing = json.loads(destination.read_text())
        if (
            existing.get("signature") == signature
            and existing.get("status") == "complete"
        ):
            return existing
    result = {
        "case_id": cid,
        "version": VERSION,
        "signature": signature,
        "source_result": str(source.resolve()),
        "source_status": record.get("status"),
        "settings": settings,
        "inputs": fingerprints,
        "status": "error",
        "metrics": {},
    }
    try:
        gt, gen, sampling = sample(case, **settings["sampling"])
        scores, frames = compute(gt, gen)
        if settings["learned"]:
            global LEARNED
            if LEARNED is None:
                LEARNED = LearnedMetrics(settings["learned"], settings["device"])
            scores.update(LEARNED.compute(gt, gen))
        result.update(
            status="complete", metrics=scores, frames=frames, sampling=sampling
        )
        result["unavailable"] = {
            key: "Requires dimensions >=161 at five scales"
            if key == "ms_ssim"
            else "Requires two sampled frames"
            for key, value in scores.items()
            if value is None
        }
    except Exception as error:  # noqa: BLE001 - isolate and record each video/backend failure
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = time.monotonic() - start
    write(destination, result)
    return result


def summarize(records):
    metrics = {}
    for key in METRICS:
        values = [
            r["metrics"].get(key)
            for r in records
            if r["status"] == "complete" and r["metrics"].get(key) is not None
        ]
        if not values:
            continue
        infinite = sum(v == "inf" for v in values)
        finite = [v for v in values if isinstance(v, (float, int))]
        metrics[key] = {
            "mean": "inf" if infinite else sum(finite) / len(finite),
            "n": len(values),
            "infinite_cases": infinite,
            "direction": METRICS[key][0],
        }
    return {
        "total": len(records),
        "complete": sum(r["status"] == "complete" for r in records),
        "errors": [
            {"case_id": r["case_id"], "error": r.get("error")}
            for r in records
            if r["status"] != "complete"
        ],
        "metrics": metrics,
        "aggregation": "Equal weight per case, all Gates included. Missing/error values excluded with counts. No combined score.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Existing VLM result root; authoritative GT/Gen paths are reused",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Separate automatic result directory"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 = full shared future; nonzero is a documented sampling cap",
    )
    parser.add_argument("--max-skew-seconds", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--learned", nargs="*", choices=["lpips", "niqe"], default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.workers < 1
        or not math.isfinite(args.fps)
        or args.fps <= 0
        or min(args.width, args.height) < 11
        or args.max_frames not in (0,)
        and args.max_frames < 2
        or args.limit < 0
        or not math.isfinite(args.max_skew_seconds)
        or args.max_skew_seconds < 0
    ):
        parser.error(
            "Invalid worker count, frame rate, dimensions, frame cap, limit or alignment tolerance"
        )
    if args.learned and args.workers != 1:
        parser.error(
            "Use --workers 1 for pretrained metrics to share one model instance"
        )
    if args.results.resolve() == args.output.resolve():
        parser.error("Use a separate --output directory to preserve VLM artifacts")
    paths = sorted(args.results.glob("*/result.json"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        parser.error("No */result.json inputs")
    settings = {
        "sampling": {
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "max_frames": args.max_frames,
            "max_skew": args.max_skew_seconds,
        },
        "learned": args.learned,
        "device": args.device,
    }
    versions = {}
    for name in ["numpy", "scipy", "av", "Pillow", "torch", "lpips", "pyiqa"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    write(
        args.output / "run.json",
        {
            "version": VERSION,
            "settings": settings,
            "packages": versions,
            "source_results": str(args.results.resolve()),
            "cases": [p.parent.name for p in paths],
            "metric_definitions": METRICS,
            "paper": "https://arxiv.org/html/2609.28654v1#S4.SS5",
            "upstream_inspected_commit": "824baddcdad087810d35638bf8f273b884f74975",
        },
    )
    jobs = [(path, args.output, settings, args.resume) for path in paths]
    records = []

    def collect(record):
        records.append(record)
        print(
            f"[{len(records)}/{len(jobs)}] {record['case_id']}: {record['status']}",
            flush=True,
        )

    if args.workers == 1:
        for job in jobs:
            collect(evaluate(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(evaluate, job) for job in jobs]):
                collect(future.result())
    records.sort(key=lambda r: r["case_id"])
    report = summarize(records)
    write(args.output / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
