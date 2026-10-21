"""Freeze simulation-reference and matched Seedance additions before judging."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path
import random
import shutil
import sqlite3
import time

from .manifest import digest, load_study, split
from .model_study import recorded_prompt, remap, sample_plan
from .sheets import duration, sha256, write_json
from .store import Store

SIM_MODELS = ("DreamDojo", "GE-SIM")
SEEDANCE = ("Seedance2.0", "Seedance2.0-Mini", "Seedance2.5")


def matched_seedance_plan(cases, old_tasks, per_pair, seed):
    pairs = list(itertools.combinations(sorted(SEEDANCE), 2))
    # Exclude an entire case if any of its three pairs was already judged.
    used = {t["case_id"] for t in old_tasks if (t["left"], t["right"]) in pairs}
    eligible = sorted(set(cases) - used)
    if per_pair < 1 or per_pair > len(eligible):
        raise ValueError("Not enough previously unused matched Seedance cases")
    rng = random.Random(seed)
    iid = [cid for cid in eligible if split(cid) == "IID"]
    ood = [cid for cid in eligible if split(cid) == "OOD"]
    n_iid = round(per_pair * len(iid) / len(eligible))
    chosen = sorted(rng.sample(iid, n_iid) + rng.sample(ood, per_pair - n_iid))
    plan = [{"id": digest([cid, a, b])[:24], "case_id": cid, "left": a, "right": b,
             "split": split(cid), "phase": "seedance_matched"} for cid in chosen for a, b in pairs]
    rng.shuffle(plan)
    return plan, eligible


def import_votes(source, output, store, selected):
    """Copy selected immutable rows, retaining identities, votes and A/B sides."""
    with sqlite3.connect(f"file:{source / 'votes.sqlite3'}?mode=ro", uri=True) as old:
        old.row_factory = sqlite3.Row
        rows = old.execute("SELECT * FROM assignments WHERE state='voted'").fetchall()
        raters = old.execute("SELECT * FROM raters").fetchall()
        secret = old.execute("SELECT value FROM metadata WHERE key='secret'").fetchone()[0]
    kept = [r for r in rows if r["task"] in selected]
    if len(kept) != len(selected) or {r["rater"] for r in kept} != {"gpt5.6"}:
        raise ValueError("Expected exactly one gpt5.6 vote per imported task")
    with store.transaction() as db:
        db.execute("UPDATE metadata SET value=? WHERE key='secret'", (secret,))
        for row in raters:
            if row["id"] == "gpt5.6":
                db.execute("INSERT INTO raters VALUES(?,?,?,?)", tuple(row))
        for row in kept:
            columns = list(row.keys())
            db.execute(f"INSERT INTO assignments({','.join(columns)}) VALUES({','.join('?' for _ in columns)})", tuple(row))
    store.secret = secret
    session = source / "rater_session.local.json"
    if session.exists():
        shutil.copy2(session, output / session.name)
        (output / session.name).chmod(0o600)
    for name in ("inputs", "responses"):
        (output / name).mkdir(exist_ok=True)
        for row in kept:
            original = source / name / f"{row['task']}.json"
            if not original.exists():
                raise ValueError(f"Missing imported audit file: {original}")
            (output / name / original.name).symlink_to(original.resolve())
    return len(kept)


def prepare(args):
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory; resume with model_judge")
    old, old_signature, old_tasks = load_study(source / "study.json")
    with sqlite3.connect(f"file:{source / 'votes.sqlite3'}?mode=ro", uri=True) as db:
        saved_signature = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        if saved_signature is None or saved_signature[0] != old_signature:
            raise ValueError("Source database no longer matches its study/input fingerprints")
        voted = {r[0] for r in db.execute("SELECT task FROM assignments WHERE state='voted' AND rater='gpt5.6'")}
    retained = {key: t for key, t in old_tasks.items() if key in voted
                and not any(t[s].endswith("_tail") for s in ("left", "right"))}
    old_plan = [{"id": key, "case_id": t["case"].case_id, "left": t["left"], "right": t["right"],
                 "split": split(t["case"].case_id), "phase": "imported"} for key, t in retained.items()]
    models = sorted((set(old.models) | set(SIM_MODELS)) - {m for m in old.models if m.endswith("_tail")})
    jobs_path = Path(old.provenance["jobs"])
    if sha256(jobs_path) != old.provenance["jobs_sha256"]:
        raise ValueError("Evaluation jobs changed since the source study")
    jobs = json.loads(jobs_path.read_text())
    lookup = {m: {j["case"]["case_id"]: j for j in jobs if j["model"] == m} for m in models}
    metadata = {(p["method"], p["case_name"]): p for p in
                json.loads((args.predictions / ".metadata/manifest.json").read_text())["predictions"]}
    old_cases = {c.case_id: c for c in old.cases}
    common = sorted(set(old_cases).intersection(*(set(lookup[m]) for m in models)))
    paths, sim_info, unavailable = set(), {}, []
    for cid in common:
        entries = {}
        for model in SIM_MODELS:
            job = lookup[model][cid]
            gt = remap(job["case"]["gt_video"], args.sim_data)
            directory = args.predictions / metadata[model, cid]["other_files"]
            if not gt.is_file():
                # DreamDojo publishes a byte-identical GT beside its prediction.
                fallback = directory / "gt.mp4"
                if fallback.is_file() and sha256(fallback) == job["gt_sha256"]:
                    gt = fallback.resolve()
            prediction = remap(job["case"]["gen_video"], args.predictions)
            if not gt.is_file() or not prediction.is_file():
                unavailable.append({"case_id": cid, "model": model, "reason": "missing verified simulation GT/prediction"})
                break
            prompt, recorded, prompt_source = recorded_prompt(directory, model, cid)
            entries[model] = {"gt": gt, "prediction": prediction, "job": job,
                              "prompt": prompt, "prompt_recorded": recorded, "prompt_source": prompt_source}
        if len(entries) == len(SIM_MODELS):
            sim_info[cid] = entries
            paths.update(p for e in entries.values() for p in (e["gt"], e["prediction"]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        durations = dict(zip(sorted(paths), pool.map(duration, sorted(paths))))
    sim_pairs = [pair for pair in itertools.combinations(models, 2) if set(pair) & set(SIM_MODELS)]
    sim_plan = sample_plan(models, sorted(sim_info), len(sim_pairs) * args.sim_per_pair, args.seed, sim_pairs)
    for task in sim_plan:
        task["phase"] = "simulation_reference"
    seed_plan, seed_eligible = matched_seedance_plan(old_cases, old_plan, args.seedance_per_pair, args.seed + 1)
    additions = sim_plan + seed_plan
    random.Random(args.seed + 2).shuffle(additions)
    cases, references = [], {}
    for cid, case in old_cases.items():
        value = case.model_dump()
        value["predictions"] = {m: p for m, p in value["predictions"].items() if m in models}
        for model, info in sim_info.get(cid, {}).items():
            job = info["job"]["case"]
            value["predictions"][model] = {
                "path": str(info["prediction"]), "duration_seconds": durations[info["prediction"]],
                "start_seconds": job.get("gen_start_seconds", 0),
                "prompt": info["prompt"], "prompt_recorded": info["prompt_recorded"],
            }
            references.setdefault(cid, {})[model] = {
                "clip": {"path": str(info["gt"]), "duration_seconds": durations[info["gt"]],
                         "start_seconds": job.get("gt_start_seconds", 0)},
                **{k: job.get(k, "") for k in ("action_description", "memory_target", "success_criterion")},
            }
        cases.append(value)
    verification = {}
    for task in additions:
        cid = task["case_id"]
        gt = old_cases[cid].gt.path
        verification[gt] = lookup[SEEDANCE[0]][cid]["gt_sha256"]
        for model in (task["left"], task["right"]):
            job = lookup[model][cid]
            verification[str(remap(job["case"]["gen_video"], args.predictions))] = job["prediction_sha256"]
            if model in SIM_MODELS:
                verification[str(sim_info[cid][model]["gt"])] = job["gt_sha256"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for path, actual in zip(verification, pool.map(sha256, verification)):
            if actual != verification[path]:
                raise ValueError(f"Source checksum mismatch: {path}")
    plan = old_plan + additions
    if len({t["id"] for t in plan}) != len(plan):
        raise ValueError("Extension would duplicate an existing task")
    study = {
        "title": f"MemoryBench · GPT5.6（真实与仿真 GT；无 tail；Seedance 追加{args.seedance_per_pair * 3}条）",
        "seed": args.seed, "raters_per_pair": 1, "models": models, "cases": cases,
        "provenance": {
            "allow_partial_models": True, "reference_models": list(SIM_MODELS), "model_references": references,
            "selected_task_ids": [t["id"] for t in plan], "new_task_ids": [t["id"] for t in additions],
            "evaluator": old.provenance["evaluator"],
            "input_protocol": "Real GT plus applicable per-model SIM GT(s), A/B; each 8 timestamped frames, no History; real-only comparisons retain the original prompt",
            "source_run": str(source), "source_signature": old_signature, "imported_tasks": len(old_plan),
            "excluded_models": [m for m in old.models if m.endswith("_tail")],
            "jobs": str(jobs_path), "jobs_sha256": sha256(jobs_path),
            "sampling": "Fixed quotas before API calls, no outcome-based selection or stopping; matched 200-case Seedance trio; balanced SIM pairs",
        },
    }
    output.mkdir(parents=True)
    write_json(output / "study.json", study)
    loaded, signature, tasks = load_study(output / "study.json")
    write_json(output / "plan.json", plan)
    write_json(output / "verified_sources.json", verification)
    preregistration = {
        "created": time.time(), "seed": args.seed, "plan_sha256": digest(plan), "study_signature": signature,
        "hypothesis_to_test": "Seedance2.5 > Seedance2.0 > Seedance2.0-Mini; no assumed effect size",
        "hypothesis_sent_to_judge": False, "selection_uses_votes": False,
        "seedance_per_pair": args.seedance_per_pair, "seedance_matched_cases": args.seedance_per_pair,
        "seedance_eligible_cases": seed_eligible,
        "seedance_case_split": dict(Counter(split(cid) for cid in {t["case_id"] for t in seed_plan})),
        "simulation_cases_available": len(sim_info), "simulation_pairs": len(sim_pairs),
        "simulation_per_pair": args.sim_per_pair, "unavailable_simulation_assets": unavailable,
        "counts": dict(Counter(t["phase"] for t in plan)),
        "stopping_rule": "Complete the frozen task list regardless of observed model order; transport/schema failures retried, never converted to ties",
    }
    write_json(output / "preregistration.json", preregistration)
    store = Store(output / "votes.sqlite3", loaded, signature, tasks)
    imported = import_votes(source, output, store, retained)
    print(json.dumps({"models": len(models), "cases": len(cases), "imported": imported,
                      "new": len(additions), "total": len(plan), "counts": preregistration["counts"],
                      "simulation_cases": len(sim_info), "seedance_case_split": preregistration["seedance_case_split"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sim-data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--sim-per-pair", type=int, default=8)
    parser.add_argument("--seedance-per-pair", type=int, default=200)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
