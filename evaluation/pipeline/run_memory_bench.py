#!/usr/bin/env python3
"""Run independent ACWM cases on GPU workers and immediately queue VLM scoring."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.pipeline"
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from queue import Empty, Queue
import shutil
import subprocess
import threading
import time

import httpx

from ..pipeline.infer_evaluation_cases import (
    DEFAULT_CHECKPOINT,
    REPO,
    command_for,
    load_cases,
    trim_prediction,
)
from ..common.config import load_config
from .. import __version__
from ..case_definition.assets import crop_video, digest
from ..ground_truth.schema import MemoryAnnotation
from ..metric.scoring import METRICS
from ..common.vlm import write_json
from ..reporting.media import comparison_video
from ..reporting.report import write_generation_report

BASE = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


class Run:
    def __init__(self, args, cases):
        self.args, self.cases = args, cases
        self.lock = threading.Lock()
        self.states = {
            c["evaluation_case_id"]: {
                "evaluation_case_id": c["evaluation_case_id"],
                "source_case_id": c["source_case_id"],
                "status": "queued",
                "memory_score": None,
                "prefix_seconds": c["sensor_duration_seconds"]["prefix"],
                "future_frames": c["generation_frame_count"],
                "evaluation_frames": c["evaluation_frame_count"],
            }
            for c in cases
        }
        previous = args.output / "status.json"
        if args.resume and previous.exists():
            for row in json.loads(previous.read_text())["cases"]:
                if row["evaluation_case_id"] in self.states:
                    self.states[row["evaluation_case_id"]].update(row)

    def update(self, case_id=None, **changes):
        with self.lock:
            if case_id:
                self.states[case_id].update(changes, updated_at=now())
            rows = list(self.states.values())
            finished = [r for r in rows if r["status"] == "completed"]
            errors = [r for r in rows if r["status"] == "error"]
            terminal = len(finished) + len(errors) == len(rows)
            status = {
                "framework_version": __version__,
                "weights": getattr(self.args, "weights", dict.fromkeys(METRICS, 1.0)),
                "updated_at": now(),
                "status": (
                    ("completed" if not errors else "completed_with_errors")
                    if terminal
                    else "running"
                ),
                "case_count": len(rows),
                "completed_count": len(finished),
                "error_count": len(errors),
                "mean_memory_score": (
                    sum(r["memory_score"] for r in finished) / len(finished)
                    if finished
                    else None
                ),
                "gate_pass_rate": (
                    sum(r["gate"] for r in finished) / len(finished)
                    if finished
                    else None
                ),
                "aggregate_denominator": len(finished),
                "cases": rows,
            }
            write_json(self.args.output / "status.json", status)
            write_generation_report(self.args.output, status)
        if case_id:
            print(f"{now()} {case_id}: {changes}", flush=True)

    def prepare(self, case):
        cid = case["evaluation_case_id"]
        output, source = self.args.output / cid, Path(case["directory"])
        output.mkdir(exist_ok=True)
        # Copy assets into the result bundle: viewers do not need the source dataset.
        copies = {
            "input/rgb_prefix.mp4": "prefix.mp4",
            "input/anchor.png": "anchor.png",
            "input/past_hands.mp4": "past_hands.mp4",
            "input/future_hands.mp4": "future_hands.mp4",
            "ground_truth/future.mp4": "gt_future.mp4",
            "ground_truth/evaluation.mp4": "gt_evaluation.mp4",
            "annotation.json": "annotation.json",
            "frame_mapping.npz": "frame_mapping.npz",
            "preview.jpg": "source_preview.jpg",
        }
        for old, new in copies.items():
            if not (output / new).exists():
                shutil.copy2(source / old, output / new)
        annotation = MemoryAnnotation.model_validate(
            json.loads((source / "memory_annotation.json").read_text())
        )
        if annotation.evaluation_case_id != cid:
            raise ValueError("Memory annotation belongs to a different case")
        write_json(output / "memory_annotation.json", annotation.model_dump())
        self.states[cid].update(
            annotation=annotation.model_dump(),
            action_frames=case["marks"]["action_end"] - case["marks"]["t"],
        )
        write_json(
            output / "input_provenance.json",
            {
                "source_run": "new_acwm_inference",
                "copied_assets": {
                    new: {"source": str(source / old), "sha256": digest(source / old)}
                    for old, new in copies.items()
                },
                "action_source_interval": [
                    0,
                    case["marks"]["action_end"] - case["marks"]["t"],
                ],
                "memory_source_interval": [
                    case["marks"]["eval_start"] - case["marks"]["t"],
                    case["marks"]["eval_end"] - case["marks"]["t"],
                ],
                "boundary_convention": "zero-based [start,end)",
            },
        )
        write_json(output / "case.json", case)
        return output

    def evaluate(self, case):
        cid = case["evaluation_case_id"]
        output = self.args.output / cid
        started = time.monotonic()
        try:
            self.update(
                cid, status="evaluating", evaluation_started_at=now(), error=None
            )
            evaluation = output / "evaluation"
            if evaluation.exists() and any(evaluation.iterdir()):
                archived = output / f"evaluation_previous_{time.time_ns()}"
                evaluation.rename(archived)
            for role in ("gt", "generated"):
                crop_video(
                    output / f"{role}_future.mp4",
                    output / f"{role}_action.mp4",
                    0,
                    case["marks"]["action_end"] - case["marks"]["t"],
                )
            cmd = [
                sys.executable,
                "-u",
                str(BASE / "pipeline/evaluate.py"),
                "--config",
                str(self.args.config),
                "--gt",
                str(output / "gt_evaluation.mp4"),
                "--generated",
                str(output / "generated_evaluation.mp4"),
                "--action",
                str(output / "generated_action.mp4"),
                "--annotation",
                str(output / "memory_annotation.json"),
                "--output-dir",
                str(evaluation),
            ]
            write_json(output / "evaluation_command.json", cmd)
            with (output / "evaluation.log").open("w") as log:
                subprocess.run(
                    cmd, cwd=BASE, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            result = json.loads((evaluation / "result.json").read_text())
            if result["status"] != "completed":
                raise RuntimeError(result.get("error", "Evaluation incomplete"))
            for name in ("result.json", "metrics.csv", "metrics.md"):
                shutil.copy2(evaluation / name, output / name)
            self.update(
                cid,
                status="completed",
                gate=result["gate"],
                memory_score=result["memory_score"],
                **result["metrics"],
                evaluation_seconds=time.monotonic() - started,
                content_evaluated=result["content_evaluated"],
                annotation=result["annotation"],
                vlm_call_count=result["vlm_call_count"],
                completed_at=now(),
            )
        except Exception as exc:
            detail = str(exc)
            result_path = output / "evaluation/result.json"
            if result_path.exists():
                detail = json.loads(result_path.read_text()).get("error", detail)
            self.update(
                cid,
                status="error",
                error_stage="evaluation",
                error=detail,
                memory_score=None,
                evaluation_seconds=time.monotonic() - started,
            )

    def worker(self, gpu, queue, evaluations, pending, pending_lock):
        while True:
            try:
                case = queue.get_nowait()
            except Empty:
                return
            cid = case["evaluation_case_id"]
            output = self.args.output / cid
            try:
                if self.states[cid]["status"] == "completed":
                    continue
                if not (output / "generated_future.mp4").exists():
                    started = time.monotonic()
                    self.update(
                        cid,
                        status="inferring",
                        gpu=gpu,
                        inference_started_at=now(),
                        error=None,
                    )
                    raw = output / "raw"
                    if raw.exists():
                        raw.rename(output / f"raw_previous_{time.time_ns()}")
                    cmd = command_for(case, self.args, raw)
                    write_json(output / "inference_command.json", cmd)
                    env = {
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "OMP_NUM_THREADS": "4",
                        "TOKENIZERS_PARALLELISM": "false",
                        "PYTHONUNBUFFERED": "1",
                    }
                    with (output / "inference.log").open("w") as log:
                        subprocess.run(
                            cmd,
                            cwd=REPO,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    counts = trim_prediction(
                        raw / "videos" / f"{cid}.mp4", output, case
                    )
                    (output / "future.mp4").rename(output / "generated_future.mp4")
                    (output / "evaluation.mp4").rename(
                        output / "generated_evaluation.mp4"
                    )
                    self.update(
                        cid,
                        status="video_ready",
                        inference_seconds=time.monotonic() - started,
                        generated_frames=counts["future.mp4"],
                        video_ready_at=now(),
                    )
                # Dispatch scoring immediately. The GPU is free to take the next case.
                with pending_lock:
                    pending.append(evaluations.submit(self.evaluate, case))
                if not (output / "comparison.mp4").exists():
                    comparison_video(
                        output / "gt_future.mp4",
                        output / "generated_future.mp4",
                        output,
                        case["generation_frame_count"],
                    )
                self.update(cid, comparison_ready=True)
            except Exception as exc:
                self.update(
                    cid,
                    status="error",
                    error_stage="inference_or_review",
                    error=str(exc),
                    memory_score=None,
                )
            finally:
                queue.task_done()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=BASE.parent / "data/evaluation_cases")
    p.add_argument("--output", type=Path, default=BASE.parent / "evaluation_results")
    p.add_argument("--config", type=Path, default=BASE.parent / "configs/local.toml")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument(
        "--model-dir", type=Path, default=Path(__import__("os").environ.get("WAN_MODEL_DIR", "models/Wan2.2-TI2V-5B"))
    )
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.dry_run = False
    for key in ("root", "output", "config", "checkpoint", "model_dir"):
        setattr(args, key, getattr(args, key).resolve())
    gpus = [int(x) for x in args.gpus.split(",")]
    if len(set(gpus)) != len(gpus) or args.steps < 1 or args.cfg < 1:
        p.error("Unique GPUs, steps >= 1 and cfg >= 1 required")
    config = load_config(args.config)
    args.weights = config.weights.model_dump()
    with httpx.Client(trust_env=False, timeout=10) as client:
        models = client.get(
            config.api.endpoint.rsplit("/chat/completions", 1)[0] + "/models"
        )
        models.raise_for_status()
        if config.api.model not in {m["id"] for m in models.json()["data"]}:
            raise ValueError("Configured VLM model is not served")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        p.error("Output must be empty; use --resume only for this same experiment")
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {
        k: str(getattr(args, k))
        for k in ("root", "config", "checkpoint", "model_dir", "steps", "cfg", "seed")
    }
    settings.update(framework_version=__version__, config_sha256=digest(args.config))
    previous = args.output / "run_config.json"
    if args.resume and (
        not previous.exists()
        or json.loads(previous.read_text())["settings"] != settings
    ):
        p.error(
            "Resume requires the same model, data, config path and sampling settings"
        )
    if not args.resume:
        write_json(
            previous,
            {
                "created_at": now(),
                "settings": settings,
                "acwm_gpus": gpus,
                "vlm_gpus": [4, 5, 6, 7],
                "vlm_config": config.public(),
                "evaluation_input": "GT and generated annotated stable clips; separate generated action clip; explicit manual annotation",
                "checkpoint_bytes": args.checkpoint.stat().st_size,
            },
        )
        shutil.copy2(
            BASE / "reporting/static/results_viewer.html", args.output / "index.html"
        )
        for name in ("results_viewer.css", "results_viewer.js"):
            shutil.copy2(BASE / "reporting/static" / name, args.output / name)
        shutil.copy2(BASE / "pipeline/PROTOCOL.md", args.output / "PROTOCOL.md")
        deployment = Path(__import__("os").environ.get("VLM_DEPLOYMENT_MANIFEST", "configs/deployment-manifest.json"))
        if deployment.exists():
            shutil.copy2(deployment, args.output / "vlm_deployment.json")
    cases = load_cases(args.root)
    run = Run(args, cases)
    for case in cases:
        run.prepare(case)
    run.update()
    queue = Queue()
    for case in cases:
        queue.put(case)
    pending, pending_lock = [], threading.Lock()
    # vLLM schedules requests across its four replicas. Keep one evaluation worker
    # per case so a ready fifth video can start without waiting for a whole case.
    with ThreadPoolExecutor(max_workers=len(cases)) as evaluations:
        with ThreadPoolExecutor(max_workers=len(gpus)) as workers:
            futures = [
                workers.submit(
                    run.worker, gpu, queue, evaluations, pending, pending_lock
                )
                for gpu in gpus
            ]
            for future in as_completed(futures):
                future.result()
        for future in as_completed(pending):
            future.result()
    run.update()
    return int(any(row["status"] != "completed" for row in run.states.values()))


if __name__ == "__main__":
    raise SystemExit(main())
