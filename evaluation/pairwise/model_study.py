"""Prepare a reproducible sampled study from the PredictionsV1 evaluation jobs."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import random

from .manifest import digest, load_study, split
from .sheets import duration, sha256, write_json


def sample_plan(models, cases, count, seed):
    """Balance unordered model pairs, split proportions and case exposure."""
    rng = random.Random(seed)
    pairs = list(itertools.combinations(sorted(models), 2))
    rng.shuffle(pairs)
    if not 0 < count <= len(cases) * len(pairs):
        raise ValueError("Requested task count is outside the common cohort")
    groups = {s: sorted(c for c in cases if split(c) == s) for s in ("IID", "OOD")}
    quotas = {pair: count // len(pairs) + (i < count % len(pairs)) for i, pair in enumerate(pairs)}
    target_iid = round(count * len(groups["IID"]) / len(cases))
    iid = {p: int(quotas[p] * len(groups["IID"]) / len(cases)) for p in pairs}
    for p in sorted(pairs, key=lambda p: -(quotas[p] * len(groups["IID"]) / len(cases) - iid[p]))[:target_iid - sum(iid.values())]:
        iid[p] += 1
    exposure = Counter()
    plan = []
    for a, b in pairs:
        for group, n in (("IID", iid[a, b]), ("OOD", quotas[a, b] - iid[a, b])):
            available = groups[group].copy()
            for _ in range(n):
                # Random tie breaking is deterministic and independent of outcomes.
                rng.shuffle(available)
                cid = min(available, key=lambda c: exposure[c])
                available.remove(cid)
                exposure[cid] += 1
                plan.append({"id": digest([cid, a, b])[:24], "case_id": cid,
                             "left": a, "right": b, "split": group})
    rng.shuffle(plan)
    return plan


def remap(value, root):
    marker = f"/{root.name}/"
    if marker not in value:
        raise ValueError(f"Unexpected dataset root: {value}")
    relative = Path(value.split(marker, 1)[1])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Dataset path escapes its root")
    # Exported benchmark assets may legitimately be symlinked to another disk.
    return (root / relative).resolve()


def recorded_prompt(directory, model, cid):
    """Use saved generator inputs only, never infer text from GT annotations."""
    for name in ("adapter.json", "prediction.json", "metadata.json", "result.json"):
        path = directory / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        identity = data.get("case_id", data.get("case_name", data.get("case", data.get("id"))))
        if identity is None or Path(str(identity)).name != cid:
            continue
        if "prompt" in data and (data["prompt"] is None or isinstance(data["prompt"], str)):
            return data["prompt"], True, str(path)
    # Seedance's generation record stores the exact submitted action text.
    path = directory / "generation.json"
    if path.exists() and model.startswith("Seedance"):
        data = json.loads(path.read_text())
        text = data.get("future_action")
        if (data.get("evaluation_case_id") == cid and isinstance(text, str)
                and data.get("future_action_sha256") in {
                    hashlib.sha256((text + suffix).encode()).hexdigest() for suffix in ("", "\n")}):
            return text, True, str(path)
    return None, False, None


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "study.json").exists():
        raise ValueError("Study already exists; use model_judge to resume it")
    jobs = json.loads(args.jobs.read_text())
    models = sorted({j["model"] for j in jobs} - set(args.exclude))
    lookup = {m: {} for m in models}
    for job in jobs:
        if job["model"] in lookup:
            cid = job["case"]["case_id"]
            if cid in lookup[job["model"]]:
                raise ValueError("Duplicate model/case job")
            lookup[job["model"]][cid] = job
    common = sorted(set.intersection(*(set(lookup[m]) for m in models)))
    if not common:
        raise ValueError("No shared cases")
    for cid in common:
        if len({lookup[m][cid]["gt_sha256"] for m in models}) != 1:
            raise ValueError(f"Different reference GTs for {cid}; use separate studies")
    plan = sample_plan(models, common, args.count, args.seed)
    manifest = json.loads((args.predictions / ".metadata/manifest.json").read_text())
    metadata = {(p["method"], p["case_name"]): p for p in manifest["predictions"]}
    paths = set()
    case_paths = {}
    for cid in common:
        base = lookup[models[0]][cid]["case"]
        gt = remap(base["gt_video"], args.data)
        history = gt.parent.parent / "input/rgb_prefix.mp4"
        predictions = {m: remap(lookup[m][cid]["case"]["gen_video"], args.predictions) for m in models}
        case_paths[cid] = (history, gt, predictions)
        paths.update([history, gt, *predictions.values()])
    with ThreadPoolExecutor(max_workers=8) as pool:
        durations = dict(zip(sorted(paths), pool.map(duration, sorted(paths))))
    verify = {}
    for task in plan:
        for model in (task["left"], task["right"]):
            job = lookup[model][task["case_id"]]
            verify[remap(job["case"]["gt_video"], args.data)] = job["gt_sha256"]
            verify[remap(job["case"]["gen_video"], args.predictions)] = job["prediction_sha256"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for path, actual in zip(verify, pool.map(sha256, verify)):
            if actual != verify[path]:
                raise ValueError(f"Source checksum mismatch: {path}")
    cases, prompt_sources = [], {}
    for cid in common:
        history, gt, predictions = case_paths[cid]
        base = lookup[models[0]][cid]["case"]
        entries = {}
        for model, path in predictions.items():
            job = lookup[model][cid]["case"]
            directory = args.predictions / metadata[model, cid]["other_files"]
            prompt, recorded, source = recorded_prompt(directory, model, cid)
            entries[model] = {"path": str(path), "start_seconds": job.get("gen_start_seconds", 0),
                              "duration_seconds": durations[path], "prompt": prompt, "prompt_recorded": recorded}
            prompt_sources[f"{cid}/{model}"] = source
        cases.append({
            "case_id": cid, "history": {"path": str(history), "duration_seconds": durations[history]},
            "gt": {"path": str(gt), "duration_seconds": durations[gt], "start_seconds": base.get("gt_start_seconds", 0)},
            "predictions": entries,
            **{k: base[k] for k in ("action_description", "memory_target", "success_criterion", "read_start_seconds", "read_end_seconds") if k in base},
        })
    study = {
        "title": "MemoryBench · GPT5.6 自动 pairwise 评估（GT/A/B 各8帧，无 History）",
        "seed": args.seed, "raters_per_pair": 1, "models": models, "cases": cases,
        "provenance": {
            "selected_task_ids": [t["id"] for t in plan],
            "evaluator": {"kind": "model", "rater_id": "gpt5.6", "model": "gpt-5.6-sol"},
            "input_protocol": "GT/A/B only; 8 uniformly sampled real frames per available future, 2x4, actual timestamps; no History; crop at GT horizon",
            "sampling": "Seeded, balanced model pairs; IID/OOD proportional to common cohort; balanced case exposure; no outcome-based selection",
            "seed": args.seed, "excluded_models": args.exclude,
            "exclusion_reason": "Simulation models use different GTs and cannot share the real-video reference",
            "jobs": str(args.jobs.resolve()), "jobs_sha256": sha256(args.jobs),
            "verified_source_files": len(verify),
        },
    }
    write_json(output / "study.json", study)
    load_study(output / "study.json")
    write_json(output / "plan.json", plan)
    write_json(output / "prompt_sources.json", prompt_sources)
    write_json(output / "verified_sources.json", {str(k): v for k, v in verify.items()})
    print(json.dumps({"tasks": len(plan), "models": len(models), "cases": len(common),
                      "splits": dict(Counter(t["split"] for t in plan)),
                      "case_exposure": dict(Counter(Counter(t["case_id"] for t in plan).values())),
                      "verified_files": len(verify)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", nargs="+", default=["GE-SIM", "DreamDojo"])
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
