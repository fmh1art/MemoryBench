"""Run the actual MemWorld sample_pt evaluator at each annotated prediction start."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.pipeline"
import argparse
import json
import os
import subprocess
from datetime import datetime

import av
import imageio.v2 as imageio

REPO = Path(os.environ.get("MEMWORLD_ROOT", "external/MemWorld")).resolve()
DEFAULT_CHECKPOINT = Path(
    os.environ.get("MEMWORLD_CHECKPOINT", "models/dit.safetensors")
)


def load_cases(root):
    """Resolve operational paths from the current root, so moved datasets work."""
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    cases = []
    for original in manifest["cases"]:
        case = dict(original)
        directory = root / case["evaluation_case_id"]
        case["directory"] = str(directory)
        case["packed_index"] = str(directory / "packed/index.json")
        if not Path(case["packed_index"]).is_file():
            raise FileNotFoundError(case["packed_index"])
        cases.append(case)
    return cases


def command_for(case, args, output):
    command = [
        sys.executable,
        "-u",
        str(REPO / "script/infer/infer_sample_pt_eval.py"),
        "--index_path",
        case["packed_index"],
        "--output_dir",
        str(output),
        "--model_dir",
        str(args.model_dir),
        "--dit_ckpt_path",
        str(args.checkpoint),
        "--episode_start_clip",
        str(case["inference_start_clip"]),
        "--max_episode_clips",
        str(case["generation_clips"]),
        "--device",
        "cuda",
        "--num_inference_steps",
        str(args.steps),
        "--cfg_scale",
        str(args.cfg),
        "--seed",
        str(args.seed),
        "--context_rope_mode",
        "legacy_76",
        "--use_keypoint_condition",
        "--use_long_context",
        "--no-overlay_keypoints",
        "--fps",
        "30",
        "--save_metadata",
        "--save_memory_trace",
    ]
    if args.dry_run:
        command.append("--dry_run")
    return command


def trim_prediction(source, output, case):
    """Drop the observed anchor and alignment tail, then expose the scoring segment."""
    spans = {
        "future.mp4": case["prediction_future_slice"],
        "evaluation.mp4": case["prediction_evaluation_slice"],
    }
    writers = {
        name: imageio.get_writer(
            str(output / name),
            fps=30,
            codec="libx264",
            quality=None,
            macro_block_size=1,
            ffmpeg_log_level="error",
            ffmpeg_params=["-crf", "16", "-threads", "2"],
        )
        for name in spans
    }
    counts = {name: 0 for name in spans}
    try:
        with av.open(str(source)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                for name, (start, end) in spans.items():
                    if start <= i < end:
                        writers[name].append_data(frame.to_ndarray(format="rgb24"))
                        counts[name] += 1
                if i >= max(end for _, end in spans.values()) - 1:
                    break
    finally:
        for writer in writers.values():
            writer.close()
    for name, (start, end) in spans.items():
        if counts[name] != end - start:
            raise ValueError(
                f"Prediction has too few frames for {name}: {counts[name]} != {end-start}"
            )
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/evaluation_cases",
    )
    parser.add_argument(
        "--case", help="Evaluation case ID, e.g. eval_002_10891f27; omit for all five"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--model-dir", type=Path, default=Path(os.environ.get("WAN_MODEL_DIR", "models/Wan2.2-TI2V-5B"))
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cases = [
        case
        for case in load_cases(args.root)
        if args.case is None or case["evaluation_case_id"] == args.case
    ]
    if not cases:
        parser.error(f"No matching case: {args.case}")
    if args.steps < 1 or args.cfg < 1:
        parser.error("--steps and --cfg must be >= 1")
    if args.output is None:
        args.output = (
            Path("evaluation_results")
            / f"memory_office_eval_{datetime.now():%Y%m%d_%H%M%S}"
        )
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(
            "Choose a new / empty output directory to preserve previous results"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    }
    results = []
    for case in cases:
        if case["status"] != "ready":
            raise ValueError(f"Case is not ready: {case['evaluation_case_id']}")
        output = args.output / case["evaluation_case_id"]
        output.mkdir()
        command = command_for(case, args, output / "raw")
        (output / "command.json").write_text(
            json.dumps(command, ensure_ascii=False, indent=2)
        )
        print(
            f"Running {case['evaluation_case_id']}; log: {output / 'inference.log'}",
            flush=True,
        )
        with (output / "inference.log").open("w") as log:
            subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if args.dry_run:
            continue
        original = output / "raw/videos" / f"{case['evaluation_case_id']}.mp4"
        counts = trim_prediction(original, output, case)
        row = {
            "evaluation_case_id": case["evaluation_case_id"],
            "annotation_id": case["annotation_id"],
            "generated_future": str(output / "future.mp4"),
            "generated_evaluation": str(output / "evaluation.mp4"),
            "ground_truth_future": str(
                Path(case["directory"]) / "ground_truth/future.mp4"
            ),
            "ground_truth_evaluation": str(
                Path(case["directory"]) / "ground_truth/evaluation.mp4"
            ),
            "evaluation_frame_count": counts["evaluation.mp4"],
            "gate_pass": None,
            "object_match_score": None,
            "memory_score": None,
            "scoring_rule": "gate_pass ? object_match_score : 0",
        }
        results.append(row)
        (args.output / "evaluation_inputs.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
        )
    print(f"Finished. Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
