#!/usr/bin/env python3
"""Rejudge an existing ACWM generation using v2 annotations, action clips and stable pairs."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.pipeline"
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
import shutil
import threading
import time
import av
from ..reporting.report import write_report
from ..reporting.validate_memory_bench_results import validate_run
from .. import __version__
from ..case_definition.assets import prepare_bundle, digest, action_frame_count
from ..common.config import load_config
from ..pipeline.pipeline import evaluate
from ..metric.schemas import PairJudgment, GateJudgment, METRICS
from ..metric.scoring import aggregate, score_pair
from ..common.video import Video, uniform_frames
from ..common.vlm import write_json

BASE = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--root", type=Path, default=BASE.parent / "data/evaluation_cases")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=BASE.parent / "configs/local.toml")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--vlm-gpus", default="4,5,6,7", help="Actual VLM GPU IDs for run provenance")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.root = args.root.resolve()
    config = load_config(args.config)
    source_status = json.loads((args.source / "status.json").read_text())
    if source_status["status"] != "completed":
        raise ValueError("Source generation run must be completed")
    ids = [r["evaluation_case_id"] for r in source_status["cases"]]
    settings = {
        "source": str(args.source),
        "root": str(args.root),
        "framework_version": __version__,
        "config_sha256": digest(args.config),
        "annotations_sha256": {
            cid: digest(args.root / cid / "memory_annotation.json") for cid in ids
        },
    }
    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume:
            raise ValueError("Use a fresh output or --resume for this exact experiment")
        if (
            json.loads((args.output / "run_config.json").read_text())["settings"]
            != settings
        ):
            raise ValueError("Resume settings changed")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output / "run_config.json",
            {
                "created_at": now(),
                "settings": settings,
                "config": config.public(),
                "generation_run": json.loads(
                    (args.source / "run_config.json").read_text()
                ),
                "generation_reused": True,
                "vlm_gpus": [int(gpu) for gpu in args.vlm_gpus.split(",")],
                "max_images_per_query": 8,
            },
        )
    shutil.copy2(
        BASE / "reporting/static/results_viewer.html", args.output / "index.html"
    )
    for name in ("results_viewer.css", "results_viewer.js"):
        shutil.copy2(BASE / "reporting/static" / name, args.output / name)
    shutil.copy2(BASE / "pipeline/PROTOCOL.md", args.output / "PROTOCOL.md")
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for directory in (
        "common",
        "case_definition",
        "pipeline",
        "ground_truth",
        "metric",
        "prompt",
        "reporting",
    ):
        shutil.copytree(
            BASE / directory,
            snapshot / directory,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "tests"),
        )
    shutil.copy2(BASE / "__init__.py", snapshot / "__init__.py")
    deployment = Path(__import__("os").environ.get("VLM_DEPLOYMENT_MANIFEST", "configs/deployment-manifest.json"))
    if deployment.is_file():
        shutil.copy2(deployment, args.output / "vlm_deployment.json")
    rows = []
    for cid in ids:
        d = args.output / cid
        if not (d / "input_provenance.json").exists():
            c, annotation = prepare_bundle(args.source / cid, args.root / cid, d)
        else:
            c = json.loads((d / "case.json").read_text())
            annotation = json.loads((d / "memory_annotation.json").read_text())
        annotation = (
            annotation.model_dump() if hasattr(annotation, "model_dump") else annotation
        )
        rows.append(
            {
                "evaluation_case_id": cid,
                "source_case_id": c["source_case_id"],
                "status": "queued",
                "memory_score": None,
                "annotation": annotation,
                "prefix_seconds": c["sensor_duration_seconds"]["prefix"],
                "future_frames": c["generation_frame_count"],
                "evaluation_frames": c["evaluation_frame_count"],
                "action_frames": action_frame_count(c),
            }
        )
    lock = threading.Lock()
    began = time.monotonic()

    def update(row=None, **changes):
        with lock:
            if row is not None:
                row.update(changes, updated_at=now())
            complete = [r for r in rows if r["status"] == "completed"]
            errors = [r for r in rows if r["status"] == "error"]
            terminal = len(complete) + len(errors) == len(rows)
            s = {
                "framework_version": __version__,
                "updated_at": now(),
                "status": (
                    ("completed_with_errors" if errors else "completed")
                    if terminal
                    else "running"
                ),
                "case_count": len(rows),
                "completed_count": len(complete),
                "error_count": len(errors),
                "aggregate_denominator": len(complete),
                "mean_memory_score": (
                    sum(r["memory_score"] for r in complete) / len(complete)
                    if complete
                    else None
                ),
                "gate_pass_rate": (
                    sum(r["gate"] for r in complete) / len(complete)
                    if complete
                    else None
                ),
                "weights": config.weights.model_dump(),
                "elapsed_seconds": time.monotonic() - began,
                "cases": rows,
            }
            write_json(args.output / "status.json", s)
            write_report(args.output, s)
            if row is not None:
                print(now(), row["evaluation_case_id"], changes, flush=True)

    update()

    def worker(row):
        d = args.output / row["evaluation_case_id"]
        dest = d / "evaluation"
        update(row, status="evaluating")
        started = time.monotonic()
        try:
            if (
                args.resume
                and (dest / "result.json").exists()
                and json.loads((dest / "result.json").read_text())["status"]
                == "completed"
            ):
                result = json.loads((dest / "result.json").read_text())
            else:
                if dest.exists():
                    dest.rename(d / f"evaluation_previous_{time.time_ns()}")
                result = evaluate(
                    d / "gt_evaluation.mp4",
                    d / "generated_evaluation.mp4",
                    config,
                    dest,
                    action_path=d / "generated_action.mp4",
                    annotation_path=d / "memory_annotation.json",
                    progress=lambda s: print(
                        now(), row["evaluation_case_id"], s, flush=True
                    ),
                )
            for name in ("result.json", "metrics.csv", "metrics.md"):
                shutil.copy2(dest / name, d / name)
            update(
                row,
                status="completed",
                gate=result["gate"],
                memory_score=result["memory_score"],
                **result["metrics"],
                evaluation_seconds=result["elapsed_seconds"],
                vlm_call_count=result["vlm_call_count"],
                error=None,
            )
        except Exception as exc:
            update(
                row,
                status="error",
                error=str(exc),
                evaluation_seconds=time.monotonic() - started,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(worker, row) for row in rows]):
            f.result()
    update()
    if any(r["status"] != "completed" for r in rows):
        return 1
    print(json.dumps(validate_run(args.output), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
