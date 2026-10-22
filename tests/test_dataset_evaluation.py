"""Isolation and coverage safeguards for long-running multi-VLM batches."""

import time
import asyncio
from types import SimpleNamespace

import pytest

from evaluation.dataset import completed_result, execute, prepare_sample, read, register_evaluator, sample_jobs, summaries
from evaluation.judge import write_json


def test_registry_preserves_legacy_results_and_rejects_namespace_collision(tmp_path):
    run = tmp_path / ".metadata/evaluation_runs/new-vlm"
    run.mkdir(parents=True)
    old_result = tmp_path / "TextControl/method/case/eval_results/vlm/result.json"
    old_result.parent.mkdir(parents=True)
    old_result.write_text('{"evaluator": "qwen", "memory_score": 0.5}\n')
    original = old_result.read_bytes()
    register_evaluator(tmp_path, run, "vlm_gpt", {"model": "gpt"})
    registry = read(tmp_path / ".metadata/vlm_evaluators.json")
    assert registry["evaluators"]["qwen3.8-27b"]["namespace"] == "vlm"
    assert registry["evaluators"]["gpt"]["namespace"] == "vlm_gpt"
    with pytest.raises(ValueError, match="different VLM"):
        register_evaluator(tmp_path, run, "vlm_gpt", {"model": "different"})
    assert read(tmp_path / ".metadata/vlm_evaluators.json") == registry
    assert old_result.read_bytes() == original


def test_completed_results_require_matching_evaluator_and_inputs(tmp_path):
    job = {"dest": str(tmp_path), "input_signature": "input-one"}
    result = {"evaluator_signature": "model-one", "input_signature": "input-one", "status": "scored"}
    write_json(tmp_path / "result.json", result)
    assert completed_result(job, "model-one") == result
    with pytest.raises(ValueError, match="different inputs/evaluator"):
        completed_result(job, "model-two")
    with pytest.raises(ValueError, match="different inputs/evaluator"):
        completed_result({**job, "input_signature": "changed-input"}, "model-one")
    write_json(tmp_path / "result.json", {**result, "status": "error"})
    assert completed_result(job, "model-one") is None


def test_partial_batch_has_no_headline_and_exclusions_are_not_zeros(tmp_path):
    run = tmp_path / ".metadata/evaluation_runs/new-vlm"
    run.mkdir(parents=True)
    write_json(run / "preflight.json", {"excluded": 78})
    jobs = [{"group": "TextControl", "model": "method", "case": {"case_id": name}}
            for name in ("one", "two")]
    results = {("TextControl", "method", "one"): {"status": "gate_failed", "memory_score": 0.0}}
    started = time.monotonic()
    progress = summaries(tmp_path, run, jobs, results, "vlm_gpt", {"model": "gpt"}, started)
    assert progress["total"] == 2 and progress["completed"] == 1 and progress["pending"] == 1
    assert progress["excluded_missing_references"] == 78
    summary_path = tmp_path / "TextControl/method/eval_results/vlm_gpt/summary.json"
    assert read(summary_path)["memory_score"] is None
    assert not read(summary_path)["benchmark_complete"]
    results[("TextControl", "method", "two")] = {"status": "gate_failed", "memory_score": 0.0}
    progress = summaries(tmp_path, run, jobs, results, "vlm_gpt", {"model": "gpt"}, started)
    assert progress["benchmark_complete"] and progress["total"] == 2
    assert read(summary_path)["total_cases"] == 2
    assert read(summary_path)["memory_score"] == 0.0


def test_sampling_is_order_independent_and_shares_identical_pools():
    jobs = [{"group": "TextControl", "model": model, "case": {"case_id": f"case_{i}"},
             "input_signature": f"{model}-{i}"}
            for model, count in (("one", 150), ("two", 150), ("small", 110))
            for i in range(count)]
    jobs.append({"group": "TextControl", "model": "small", "case": {"case_id": "unavailable"},
                 "preflight_error": "Missing reference"})
    selected = sample_jobs(jobs, 100, 42)
    assert selected == sample_jobs(list(reversed(jobs)), 100, 42)
    assert selected != sample_jobs(jobs, 100, 43)
    ids = {model: {j["case"]["case_id"] for j in selected if j["model"] == model}
           for model in ("one", "two", "small")}
    assert all(len(cases) == 100 for cases in ids.values())
    assert ids["one"] == ids["two"] and "unavailable" not in ids["small"]
    assert all(j in jobs for j in selected)  # Keep exact input signatures and result paths for reuse.
    with pytest.raises(ValueError, match="Only 110 eligible"):
        sample_jobs(jobs, 120, 42)


def test_sample_plan_and_summaries_keep_full_run_intact(tmp_path):
    source = tmp_path / ".metadata/evaluation_runs/full"
    source.mkdir(parents=True)
    jobs = [{"group": "TextControl", "model": "method", "case": {"case_id": f"case_{i}"},
             "input_signature": str(i), "dest": str(tmp_path / f"case_{i}/eval_results/vlm_gpt")}
            for i in range(4)]
    write_json(source / "jobs.json", jobs)
    write_json(source / "preflight.json", {"dataset": str(tmp_path), "namespace": "vlm_gpt", "total": 5, "excluded": 1})
    write_json(source / "excluded_jobs.json", [{"case_id": "missing_reference"}])
    identity = {"model": "gpt", "namespace": "vlm_gpt"}
    write_json(source / "evaluator.json", identity)
    full_summary = tmp_path / "TextControl/method/eval_results/vlm_gpt/summary.json"
    full_summary.parent.mkdir(parents=True)
    full_summary.write_text('{"expected_cases":4}\n')
    run = source.parent / "sample2"
    prepare_sample(SimpleNamespace(source_run=source, dataset=tmp_path, namespace="vlm_gpt",
                                   sample_size=2, seed=42, run=run))
    selected = read(run / "jobs.json")
    assert len(selected) == 2 and all(j in jobs for j in selected)
    assert read(run / "evaluator.json") == identity
    assert read(source / "jobs.json") == jobs
    assert read(run / "excluded_jobs.json") == [{"case_id": "missing_reference"}]
    results = {("TextControl", "method", j["case"]["case_id"]): {"status": "gate_failed", "memory_score": 0.0}
               for j in selected}
    progress = summaries(tmp_path, run, selected, results, "vlm_gpt", identity, time.monotonic())
    assert progress["benchmark_complete"] and progress["total"] == 2
    assert progress["evaluation_scope"] == "sample_per_model"
    assert full_summary.read_text() == '{"expected_cases":4}\n'
    assert read(full_summary.parent / "sample2/summary.json")["expected_cases"] == 2
    register_evaluator(tmp_path, run, "vlm_gpt", identity)
    assert read(tmp_path / ".metadata/vlm_evaluators.json")["evaluators"]["gpt"]["model_summary"].endswith('/sample2/summary.json')


def test_superseded_run_stops_before_loading_credentials_or_calling_api(tmp_path):
    write_json(tmp_path / "superseded_by.json", {"run": "sample100"})
    with pytest.raises(ValueError, match="superseded"):
        asyncio.run(execute(SimpleNamespace(run=tmp_path)))
