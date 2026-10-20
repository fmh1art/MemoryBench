"""Build a study from existing evaluation input lists; no recursive project scan."""

import argparse
import json
import random
import subprocess
from pathlib import Path

from .manifest import Study, split


def probe(path, start=0):
    path = Path(path).resolve()
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=duration,codec_name:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError(f"No video stream: {path}")
    duration = data["streams"][0].get("duration") or data.get("format", {}).get("duration")
    if not duration or duration == "N/A":
        raise ValueError(f"No usable duration: {path}")
    return {"path": str(path), "start_seconds": float(start), "duration_seconds": float(duration)}


def named(value):
    name, sep, path = value.partition("=")
    if not sep or not name.strip() or not path:
        raise argparse.ArgumentTypeError("Expected ModelName=/path/to/cases.json")
    return name, Path(path).resolve()


def build(args):
    models, sources = {}, {}
    for name, path in args.model:
        if name in models:
            raise ValueError(f"Duplicate model: {name}")
        entries = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError(f"Expected evaluation JSON list: {path}")
        indexed = {}
        for raw in entries:
            entry = dict(raw)
            cid = entry["case_id"]
            if cid in indexed:
                raise ValueError(f"Duplicate case_id in {path}: {cid}")
            for key in ("gt_video", "gen_video", "history_video"):
                if entry.get(key):
                    entry[key] = str((path.parent / entry[key]).resolve())
            indexed[cid] = entry
        models[name], sources[name] = indexed, str(path)
    if len(models) < 2:
        raise ValueError("At least two --model lists are required")
    common = set.intersection(*(set(items) for items in models.values()))
    if args.case_id:
        missing = set(args.case_id) - common
        if missing:
            raise ValueError(f"Requested cases missing from a model: {sorted(missing)}")
        common = set(args.case_id)
    candidates = sorted(common)
    if not candidates:
        raise ValueError("No common cases")
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    if args.limit and args.limit < len(candidates):
        # Stratified, proportional allocation; include both strata if limit >= 2.
        groups = {s: [c for c in candidates if split(c) == s] for s in ("IID", "OOD")}
        if args.limit >= 2 and all(groups.values()):
            n = max(1, min(len(groups["IID"]), args.limit - 1,
                           round(args.limit * len(groups["IID"]) / len(candidates))))
            n = max(n, args.limit - len(groups["OOD"]))
            candidates = groups["IID"][:n] + groups["OOD"][:args.limit - n]
        else:
            candidates = candidates[:args.limit]
    overrides = json.loads(args.prompts.read_text()) if args.prompts else {}
    prompt_sources = []
    cases = []
    first = next(iter(models))
    for cid in sorted(candidates):
        ref = models[first][cid]
        gt = Path(ref["gt_video"])
        history = Path(ref["history_video"]) if ref.get("history_video") else gt.parent.parent / "input/rgb_prefix.mp4"
        if not history.is_file():
            raise ValueError(f"Missing history for {cid}; supply history_video in input list: {history}")
        comparisons = ("action_description", "memory_target", "success_criterion",
                       "read_start_seconds", "read_end_seconds")
        predictions = {}
        for name, indexed in models.items():
            entry = indexed[cid]
            if Path(entry["gt_video"]).resolve() != gt.resolve() or float(entry.get("gt_start_seconds", 0)) != float(ref.get("gt_start_seconds", 0)):
                raise ValueError(f"Different GT/timeline for {cid}: {name}; build separate studies for real/simulation GT")
            if any(entry.get(k) != ref.get(k) for k in comparisons):
                raise ValueError(f"Different evaluation scope for {cid}: {name}")
            if entry.get("history_video") and Path(entry["history_video"]).resolve() != history.resolve():
                raise ValueError(f"Different history for {cid}: {name}")
            pred = probe(entry["gen_video"], entry.get("gen_start_seconds", 0))
            if cid in overrides.get(name, {}):
                pred.update(prompt=overrides[name][cid], prompt_recorded=True)
            elif "text_prompt" in entry:
                pred.update(prompt=entry["text_prompt"], prompt_recorded=True)
            elif getattr(args, "adapter_prompts", False):
                adapter_path = Path(pred["path"]).parent / "other_files/adapter.json"
                if adapter_path.is_file():
                    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
                    if adapter.get("case_id") != cid or adapter.get("model") != name:
                        raise ValueError(f"Adapter identity mismatch: {adapter_path}")
                    if "prompt" in adapter:
                        pred.update(prompt=adapter["prompt"], prompt_recorded=True)
                        prompt_sources.append({"model": name, "case_id": cid, "path": str(adapter_path)})
            predictions[name] = pred
        cases.append(dict(
            case_id=cid, history=probe(history), gt=probe(gt, ref.get("gt_start_seconds", 0)),
            predictions=predictions, **{k: ref[k] for k in comparisons if k in ref},
        ))
    study = Study.model_validate(dict(
        title=args.title, seed=args.seed, raters_per_pair=args.raters_per_pair,
        baseline=args.baseline, models=list(models), cases=cases,
        provenance={"input_lists": sources, "common_cohort_n": len(common),
                    "selected_n": len(cases), "selection": "seeded proportional IID/OOD sampling" if args.limit else "all common cases",
                    "prompt_policy": "Explicit text_prompt, --prompts, or identity-checked adapter.json with --adapter-prompts; never infer from GT annotations",
                    "adapter_prompt_sources": prompt_sources,
                    "excluded_not_common": {m: len(v.keys() - common) for m, v in models.items()}},
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("Output exists; use a new study filename to preserve active assignments")
    args.output.write_text(study.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=named, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, help='JSON: {"model": {"case_id": "actual prompt or null"}}')
    parser.add_argument("--adapter-prompts", action="store_true", help="Read identity-checked prediction/other_files/adapter.json prompts")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raters-per-pair", type=int, default=3)
    parser.add_argument("--baseline")
    parser.add_argument("--title", default="MemoryBench · 人工记忆评估")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    study = build(args)
    print(json.dumps({"study": str(args.output.resolve()), "cases": len(study.cases),
                      "models": study.models, "splits": {s: sum(split(c.case_id) == s for c in study.cases) for s in ("IID", "OOD")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
