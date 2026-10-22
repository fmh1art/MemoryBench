"""Isolation and coverage safeguards for long-running multi-VLM batches."""

import time

import pytest

from evaluation.dataset import completed_result, read, register_evaluator, summaries
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
