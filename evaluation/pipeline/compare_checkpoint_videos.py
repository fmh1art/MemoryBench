#!/usr/bin/env python3
"""Generate the same held-out episode with eight saved checkpoints on eight GPUs."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.pipeline"
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

from ..pipeline.infer_evaluation_cases import (
    command_for,
    load_cases,
    trim_prediction,
    REPO,
)

BASE = Path(__file__).resolve().parents[1]


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--case", default="eval_002_10891f27")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_config.json").exists():
        raise ValueError("Use a new output folder to preserve previous comparisons")
    case = next(
        c
        for c in load_cases(BASE.parent / "data/evaluation_cases")
        if c["evaluation_case_id"] == args.case
    )
    plan = json.loads((args.training / "training_plan.json").read_text())
    assert (
        case["source_case_id"]
        in plan["source_exclusion_audit"]["excluded_source_cases"]
    )
    checkpoint_dir = (
        args.training / "training" / plan["training_env"]["RUN_NAME"] / "checkpoints"
    )
    checkpoints = sorted(
        checkpoint_dir.glob("dit_step*.safetensors"),
        key=lambda p: int(p.stem.removeprefix("dit_step")),
    )[-8:]
    assert len(checkpoints) == 8
    casesource = Path(case["directory"])
    common = output / "reference"
    common.mkdir(exist_ok=True)
    for source, destination in {
        "input/rgb_prefix.mp4": "prefix.mp4",
        "input/anchor.png": "anchor.png",
        "input/past_hands.mp4": "past_hands.mp4",
        "input/future_hands.mp4": "future_hands.mp4",
        "ground_truth/future.mp4": "gt_future.mp4",
        "ground_truth/evaluation.mp4": "gt_evaluation.mp4",
        "annotation.json": "annotation.json",
        "frame_mapping.npz": "frame_mapping.npz",
    }.items():
        shutil.copy2(casesource / source, common / destination)
    baseline = BASE.parent / "evaluation_results" / args.case / "generated_future.mp4"
    if baseline.exists():
        shutil.copy2(baseline, common / "original_step4529.mp4")
    write_json(output / "case.json", case)
    runs = []
    for gpu, checkpoint in enumerate(checkpoints):
        step = int(checkpoint.stem.removeprefix("dit_step"))
        epoch = step / plan["optimizer_steps_per_epoch"]
        name = f"epoch_{epoch:04.1f}".replace(".", "p") + f"_step{step}"
        (output / name).mkdir()
        runs.append(
            {
                "id": name,
                "gpu": gpu,
                "step": step,
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "status": "queued",
            }
        )
    settings = {
        "created_at": now(),
        "evaluation_case_id": args.case,
        "source_case_held_out": True,
        "model_dir": __import__("os").environ.get("WAN_MODEL_DIR", "models/Wan2.2-TI2V-5B"),
        "steps": 20,
        "cfg": 5.0,
        "seed": 42,
        "context_rope_mode": "legacy_76",
        "rgb_and_hands_separate": True,
        "future_hand_condition": True,
        "future_head_pose_condition": True,
        "teacher_forced_rgb": False,
        "history_memory": "observed prefix followed by autoregressive generated RGB",
        "generated_frame_count": case["generation_frame_count"],
        "fps": 30,
        "selection": "latest eight retained training checkpoints; same case and sampler settings",
        "runs": runs,
    }
    write_json(output / "run_config.json", settings)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>Checkpoint 视频对比</title><h1>8 个 checkpoint 正在生成视频</h1><p>完成后此页面会自动显示同步视频对比。</p>'
    )
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for script in (
        Path(__file__),
        BASE / "pipeline/infer_evaluation_cases.py",
        REPO / "script/infer/infer_sample_pt_eval.py",
    ):
        shutil.copy2(script, snapshot / script.name)
    lock = threading.Lock()
    started = time.monotonic()

    def update(run, **changes):
        with lock:
            run.update(changes)
            write_json(
                output / "status.json",
                {
                    "updated_at": now(),
                    "completed": sum(r["status"] == "video_ready" for r in runs),
                    "count": len(runs),
                    "elapsed_seconds": time.monotonic() - started,
                    "runs": runs,
                },
            )

    def worker(run):
        destination = output / run["id"]
        began = time.monotonic()
        options = SimpleNamespace(
            model_dir=settings["model_dir"],
            checkpoint=run["checkpoint"],
            steps=20,
            cfg=5.0,
            seed=42,
            dry_run=False,
        )
        command = command_for(case, options, destination / "raw")
        write_json(destination / "command.json", command)
        update(run, status="inferring", started_at=now())
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(run["gpu"]),
            "OMP_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
        try:
            with (destination / "inference.log").open("w") as log:
                subprocess.run(
                    command,
                    cwd=REPO,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            counts = trim_prediction(
                destination / "raw/videos" / f"{args.case}.mp4", destination, case
            )
            (destination / "future.mp4").rename(destination / "generated_future.mp4")
            (destination / "evaluation.mp4").rename(
                destination / "generated_evaluation.mp4"
            )
            checksum = hashlib.sha256(
                (destination / "generated_future.mp4").read_bytes()
            ).hexdigest()
            update(
                run,
                status="video_ready",
                frames=counts["future.mp4"],
                sha256=checksum,
                inference_seconds=time.monotonic() - began,
                finished_at=now(),
            )
            print(f"{now()} {run['id']}: video ready", flush=True)
        except BaseException as exc:
            update(run, status="failed", error=repr(exc))
            raise

    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(worker, run) for run in runs]):
            future.result()
    write_json(
        output / "run_config.json",
        {**settings, "inference_wall_seconds": time.monotonic() - started},
    )
    subprocess.run(
        [
            sys.executable,
            str(BASE / "reporting/build_checkpoint_demo.py"),
            "--output",
            str(output),
        ],
        cwd=BASE,
        check=True,
    )
    print(f"{now()} COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
