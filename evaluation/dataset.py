"""Frozen, resumable full-dataset evaluation with a separate namespace per VLM."""

import argparse
import asyncio
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .judge import Judge, SCHEMAS, write_json
from .run import evaluate_case
from .schema import Case
from .scoring import SCORING_VERSION, select_metrics, summarize

GROUPS = ("DirectAction", "TextControl", "VisualSignal")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ino]


def now():
    return datetime.now(timezone.utc).isoformat()


def provider(path):
    import yaml

    config = yaml.safe_load(Path(path).read_text())
    for name in ("openai_base_url", "llm_name", "key"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError(f"Provider config requires {name}")
    limit = config.get("max_parrel_requests", 8)
    if not isinstance(limit, int) or not 1 <= limit <= 256:
        raise ValueError("Invalid max_parrel_requests")
    config["max_parrel_requests"] = limit
    return config


def remap_reference(path, real_root, sim_root):
    for marker, root in (("/MemoryBenchDataV1/", real_root), ("/MemoryDataV1-Sim/", sim_root)):
        if marker in path:
            return root / path.split(marker, 1)[1]
    return Path(path)


def prepare(args):
    root = args.dataset.resolve()
    prior_jobs = read(args.baseline_jobs)
    by_key = {(j["group"], j["model"], j["case"]["case_id"]): j for j in prior_jobs}
    real_templates = {j["case"]["case_id"]: j for j in prior_jobs if j["group"] != "DirectAction"}
    manifest = {}
    with (root / ".metadata/file_manifest.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["path"].endswith("/prediction/prediction.mp4"):
                manifest[row["path"]] = row
    paths = []
    for group in GROUPS:
        for method in sorted((root / group).iterdir()):
            if method.is_dir():
                paths.extend(sorted(method.glob("*/prediction/prediction.mp4")))
    # Only known canonical prediction paths are evaluated; alternate views are not duplicate cases.
    jobs = []
    for prediction in paths:
        group, model, case_id = prediction.relative_to(root).parts[:3]
        original = by_key.get((group, model, case_id))
        template = original or (real_templates.get(case_id) if group != "DirectAction" else None)
        if template is None:
            raise ValueError(f"Missing case annotation for {group}/{model}/{case_id}")
        case = copy.deepcopy(template["case"])
        gt = remap_reference(case["gt_video"], args.real_gt_root, args.sim_gt_root)
        case["gt_video"] = str(gt)
        case["gen_video"] = str(prediction)
        if case.get("memory_objects_source") and "/MemoryBenchDataV1/" in case["memory_objects_source"]:
            case["memory_objects_source"] = str(remap_reference(case["memory_objects_source"], args.real_gt_root, args.sim_gt_root))
        case = Case.model_validate(case).resolve_videos(root).model_dump()
        rel = prediction.relative_to(root).as_posix()
        row = manifest.get(rel)
        size = row.get("size", row.get("bytes")) if row else None
        if not row or prediction.stat().st_size != size:
            raise ValueError(f"Prediction differs from published manifest: {rel}")
        jobs.append({"group": group, "model": model, "case": case,
            "prediction_sha256": row["sha256"], "prediction_fingerprint": fingerprint(prediction),
            "gt_sha256": template["gt_sha256"], "gt_candidates": [str(gt), str(prediction.parent / "other_files/gt.mp4")],
            "dest": str(prediction.parent.parent / "eval_results" / args.namespace),
            "prior_qwen_result": str(prediction.parent.parent / "eval_results/vlm/result.json"),
            "annotation_source": "original_qwen_job" if original else "same_real_gt_case_annotation"})
    candidates = sorted({name for j in jobs for name in j["gt_candidates"] if Path(name).is_file()})
    with ThreadPoolExecutor(max_workers=12) as pool:
        hashes = dict(zip(candidates, pool.map(digest, candidates)))
    for job in jobs:
        selected = next((p for p in job.pop("gt_candidates") if hashes.get(p) == job["gt_sha256"]), None)
        if selected is None:
            job["preflight_error"] = "No local reference matches the original evaluation GT SHA256"
        else:
            job["case"]["gt_video"] = selected
            job["gt_fingerprint"] = fingerprint(selected)
        # Freeze source hashes, scope and paths; retries must never change an evaluation input.
        job["input_signature"] = hashlib.sha256(json.dumps({k: job[k] for k in
            ("group", "model", "case", "prediction_sha256", "gt_sha256")}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    args.run.mkdir(parents=True, exist_ok=False)
    excluded = [j for j in jobs if "preflight_error" in j] if args.skip_missing_references else []
    scheduled = [j for j in jobs if "preflight_error" not in j] if args.skip_missing_references else jobs
    write_json(args.run / "jobs.json", scheduled)
    write_json(args.run / "excluded_jobs.json", excluded)
    report = {"created_at": now(), "dataset": str(root), "namespace": args.namespace,
        "baseline_jobs": str(args.baseline_jobs), "total": len(jobs),
        "ready": sum("preflight_error" not in j for j in jobs),
        "scheduled": len(scheduled), "excluded": len(excluded),
        "exclusion_policy": "Missing original GT explicitly excluded by the user; not scored as zero" if excluded else None,
        "blocked_references": [{"group": j["group"], "model": j["model"], "case_id": j["case"]["case_id"],
            "expected_gt_sha256": j["gt_sha256"], "original_gt": j["case"]["gt_video"]} for j in jobs if "preflight_error" in j],
        "models": dict(Counter(j["model"] for j in jobs)),
        "prediction_identity": "Size checked against the just-synchronized SHA256 manifest; stat fingerprint is checked again before each case",
        "gt_identity": "Every available GT hashed and compared to the original Qwen evaluation",
        "tail_models": "Included in this full VLM evaluation; pairwise exclusion is a separate protocol",
        "rubric_policy": "Original Gate and eight-pair Memory evaluation; no old VLM judgments reused"}
    write_json(args.run / "preflight.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "blocked_references"}, ensure_ascii=False), flush=True)


def model_settings(config, args):
    base = Path(__file__).parent
    sources = [base / "common.md", base / "memory_common.md", *(base / d / "rubric.md" for d in SCHEMAS)]
    return {"model": config["llm_name"], "namespace": args.namespace,
        "protocol": "memory_evaluation.frame_pairs.v3", "scoring_version": SCORING_VERSION,
        "rubric_sha256": {str(p.relative_to(base)): digest(p) for p in sources},
        "schema_sha256": {k: hashlib.sha256(json.dumps(v.model_json_schema(), sort_keys=True).encode()).hexdigest() for k, v in SCHEMAS.items()},
        "selected_metrics": list(select_metrics()), "max_completion_tokens": args.max_tokens,
        "temperature": None, "tile_size": [480, 320], "max_skew_seconds": 0.1,
        "extra_body": {}, "api_version": str(config.get("api_version", "")),
        "request_concurrency": config["max_parrel_requests"], "case_workers": args.case_workers,
        "config_path": str(args.provider_config.resolve()), "prices_cny_per_million": config.get("price_yuan_per_million_token", {})}


def completed_result(job, signature):
    path = Path(job["dest"]) / "result.json"
    if not path.exists():
        return None
    result = read(path)
    if result.get("evaluator_signature") != signature or result.get("input_signature") != job["input_signature"]:
        raise ValueError(f"Result belongs to different inputs/evaluator: {path}")
    return result if result.get("status") != "error" else None


def register_evaluator(root, run, namespace, identity):
    """Name both the legacy Qwen layout and new VLM layouts without moving old data."""
    path = root / ".metadata/vlm_evaluators.json"
    lock = path.with_suffix(".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = read(path) if path.exists() else {"format": "memorybench.vlm.registry.v1", "evaluators": {}}
        entries = registry["evaluators"]
        entries.setdefault("qwen3.8-27b", {"model": "qwen3.8-27b", "namespace": "vlm",
            "case_result": "{group}/{method}/{case}/eval_results/vlm/result.json",
            "legacy": True, "run": ".metadata/evaluation_runs/qwen38_20261018"})
        for model, entry in entries.items():
            if entry["namespace"] == namespace and model != identity["model"]:
                raise ValueError("Evaluation namespace already belongs to a different VLM")
        entries[identity["model"]] = {"model": identity["model"], "namespace": namespace,
            "case_result": "{group}/{method}/{case}/eval_results/" + namespace + "/result.json",
            "model_summary": "{group}/{method}/eval_results/" + namespace + "/summary.json",
            "run": run.resolve().relative_to(root).as_posix(), "scoring_version": SCORING_VERSION,
            "legacy": False}
        registry["updated_at"] = now()
        write_json(path, registry)
    finally:
        lock.close()


def summaries(root, run, jobs, results, namespace, identity, started):
    by_model = defaultdict(list)
    expected = Counter((j["group"], j["model"]) for j in jobs)
    for key, result in results.items():
        by_model[key[:2]].append(result)
    model_summaries = {}
    for key, total in expected.items():
        summary = summarize(by_model[key]) if by_model[key] else {"total_cases": 0, "benchmark_complete": False}
        summary.update(evaluator=identity["model"], evaluation_namespace=namespace, expected_cases=total,
                       pending_cases=total - len(by_model[key]))
        if summary["pending_cases"]:
            summary.update(benchmark_complete=False, memory_score=None, memory_score_100=None)
            if isinstance(summary.get("report"), dict):
                summary["report"]["memory_score"] = None
        dest = root / key[0] / key[1] / "eval_results" / namespace
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / "run.json").exists():
            write_json(dest / "run.json", {**identity, "run": str(run), "expected_cases": total})
        write_json(dest / "summary.json", summary)
        model_summaries[key[1]] = summary
    statuses = Counter(r["status"] for r in results.values())
    progress = {"updated_at": now(), "pid": os.getpid(), "model": identity["model"], "namespace": namespace,
        "total": len(jobs), "completed": sum(v for k, v in statuses.items() if k != "error"),
        "errors": statuses.get("error", 0), "pending": len(jobs) - len(results),
        "blocked_references": sum("preflight_error" in j for j in jobs), "statuses": dict(statuses),
        "excluded_missing_references": read(run / "preflight.json").get("excluded", 0),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "benchmark_complete": len(results) == len(jobs) and not statuses.get("error"),
        "models": {m: {k: s.get(k) for k in ("expected_cases", "total_cases", "pending_cases", "gate_passed", "error_cases", "memory_score_100")} for m, s in model_summaries.items()}}
    write_json(run / "progress.json", progress)
    write_json(run / "summaries.json", model_summaries)
    return progress


async def execute(args):
    config = provider(args.provider_config)
    root = args.dataset.resolve()
    jobs = read(args.run / "jobs.json")
    if any(Path(j["dest"]).name != args.namespace for j in jobs):
        raise ValueError("Namespace differs from the frozen plan")
    identity = model_settings(config, args)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    identity_path = args.run / "evaluator.json"
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError("Evaluator settings changed; use a separate run")
    write_json(identity_path, identity)
    register_evaluator(root, args.run, args.namespace, identity)
    endpoint = config["openai_base_url"].rstrip("/") + "/chat/completions"
    params = {"api-version": str(config["api_version"])} if config.get("api_version") else {}
    judge = Judge(endpoint, config["llm_name"], config["key"], concurrency=config["max_parrel_requests"],
        timeout=args.timeout, retries=args.retries, max_tokens=args.max_tokens, temperature=None,
        token_parameter="max_completion_tokens", request_headers={"api-key": config["key"]}, request_params=params, resume=True)
    started = time.monotonic()
    results = {}
    for job in jobs:
        existing = completed_result(job, signature)
        if existing:
            results[(job["group"], job["model"], job["case"]["case_id"])] = existing
    pending = [j for j in jobs if (j["group"], j["model"], j["case"]["case_id"]) not in results and "preflight_error" not in j]
    if args.pilot:
        # Deterministic examples from different generation models; their new GPT labels remain reusable.
        candidates = []
        for name in ("Seedance2.5", "DreamDojo", "MiniMax-H3", "GE-SIM"):
            candidates.extend(next(([j] for j in pending if j["model"] == name), []))
        pending = candidates[:args.pilot]
    elif args.limit is not None:
        pending = pending[:args.limit]
    stop = asyncio.Event()
    def report():
        return summaries(root, args.run, jobs, results, args.namespace, identity, started)
    async def heartbeat():
        while not stop.is_set():
            report()
            try:
                await asyncio.wait_for(stop.wait(), timeout=30)
            except TimeoutError:
                pass
    heartbeat_task = asyncio.create_task(heartbeat())
    limit = asyncio.Semaphore(args.case_workers)
    async def one(job):
        async with limit:
            key = (job["group"], job["model"], job["case"]["case_id"])
            for field, saved in (("gen_video", "prediction_fingerprint"), ("gt_video", "gt_fingerprint")):
                if fingerprint(job["case"][field]) != job[saved]:
                    raise ValueError("Input changed after preparation: " + job["case"][field])
            output = Path(job["dest"])
            prior = output / "result.json"
            if prior.exists():
                previous = read(prior)
                if previous.get("input_signature") != job["input_signature"]:
                    raise ValueError("Refusing to overwrite a result from different inputs")
            result = await evaluate_case(Case.model_validate(job["case"]), judge, output,
                resume=output.exists(), max_skew_seconds=0.1, result_metadata={
                    "evaluator": config["llm_name"], "evaluator_signature": signature,
                    "input_signature": job["input_signature"], "source_hashes": {
                        "prediction": job["prediction_sha256"], "gt": job["gt_sha256"]}})
            results[key] = result
            with (args.run / "events.jsonl").open("a") as stream:
                stream.write(json.dumps({"time": now(), "group": key[0], "model": key[1], "case_id": key[2],
                    "status": result["status"], "memory_score": result.get("memory_score")}, ensure_ascii=False) + "\n")
            if result["status"] == "error" and any(code in json.dumps(result.get("error", result.get("errors", {})))
                                                     for code in ("VLM HTTP 401", "VLM HTTP 403")):
                raise RuntimeError("Provider authentication failed; remaining requests were stopped for safe resume")
    tasks = []
    try:
        print(json.dumps({"scheduled": len(pending), "resumed": len(results), "blocked": sum("preflight_error" in j for j in jobs),
            "concurrency": config["max_parrel_requests"]}), flush=True)
        tasks = [asyncio.create_task(one(j)) for j in pending]
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        stop.set()
        await heartbeat_task
        await judge.close()
        progress = report()
        print(json.dumps(progress, ensure_ascii=False), flush=True)
    return 0 if not progress["errors"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--baseline-jobs", type=Path)
    parser.add_argument("--real-gt-root", type=Path)
    parser.add_argument("--sim-gt-root", type=Path)
    parser.add_argument("--skip-missing-references", action="store_true",
                        help="Explicitly exclude unavailable original GTs and record them; never score as zero")
    parser.add_argument("--provider-config", type=Path)
    parser.add_argument("--case-workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--pilot", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-rounds", type=int, default=3,
                        help="Resume unresolved cases after a full pass; successful calls remain cached")
    args = parser.parse_args()
    if not re.fullmatch(r"vlm_[a-z0-9_]+", args.namespace):
        parser.error("namespace must have the form vlm_model_name")
    if args.mode == "prepare":
        if not all((args.baseline_jobs, args.real_gt_root, args.sim_gt_root)):
            parser.error("prepare requires baseline jobs and both GT roots")
        prepare(args)
        return 0
    if not args.provider_config or args.case_workers < 1:
        parser.error("run requires a provider config and positive case-workers")
    lock = (args.run / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.retry_rounds < 1:
        parser.error("retry-rounds must be positive")
    for retry_round in range(args.retry_rounds):
        status = asyncio.run(execute(args))
        if status == 0 or args.pilot or args.limit is not None:
            return status
        if retry_round + 1 < args.retry_rounds:
            print(json.dumps({"event": "retry_unresolved_cases", "next_round": retry_round + 2}), flush=True)
            time.sleep(60)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
