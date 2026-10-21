"""Model rater checks: selection, input boundaries, timestamps and resumability."""

import json
from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np
import pytest

from evaluation.pairwise.manifest import Clip, load_study
from evaluation.pairwise.extend_study import matched_seedance_plan
from evaluation.pairwise.model_judge import assign_task, parse_choice, prepare_input, register_model, request_body
from evaluation.pairwise.model_study import recorded_prompt, sample_plan
from evaluation.pairwise.report import summarize
from evaluation.pairwise.sheets import make_sheet
from evaluation.pairwise.store import Store


def study_file(tmp_path):
    video = tmp_path / "video.mp4"
    with av.open(str(video), "w") as output:
        stream = output.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(20):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 10, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    clip = {"path": str(video), "duration_seconds": 2}
    value = {"models": ["one", "two", "three"], "raters_per_pair": 1,
             "cases": [{"case_id": "case", "history": clip, "gt": clip,
                        "predictions": {m: clip for m in ("one", "two", "three")}}]}
    path = tmp_path / "study.json"
    path.write_text(json.dumps(value))
    return path


def test_sample_selection_and_coverage(tmp_path):
    models = [f"m{i}" for i in range(17)]
    cases = [f"mine_{i}" for i in range(58)] + [f"ood_{i}" for i in range(279)]
    plan = sample_plan(models, cases, 1000, 42)
    assert plan == sample_plan(models, cases, 1000, 42)
    assert len({p["id"] for p in plan}) == 1000
    assert sum(p["split"] == "IID" for p in plan) == 172
    from collections import Counter
    assert set(Counter((p["left"], p["right"]) for p in plan).values()) == {7, 8}
    assert len({p["case_id"] for p in plan}) == 337
    path = study_file(tmp_path)
    _, _, tasks = load_study(path)
    value = json.loads(path.read_text())
    value["provenance"] = {"selected_task_ids": [next(iter(tasks))]}
    path.write_text(json.dumps(value))
    study, _, filtered = load_study(path)
    assert len(filtered) == 1
    assert summarize([], study, replicates=0)["coverage"]["expected_tasks"] == 1
    for invalid in ([], ["missing"], [next(iter(tasks))] * 2, [1], "invalid"):
        value["provenance"]["selected_task_ids"] = invalid
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="selected_task_ids"):
            load_study(path)


def test_real_timestamps_and_no_history_in_request(tmp_path):
    path = study_file(tmp_path)
    study, signature, tasks = load_study(path)
    clip = Clip(path=study.cases[0].gt.path, start_seconds=.5, duration_seconds=2)
    sheet = make_sheet(clip, .8, tmp_path / "sheets")
    assert len(sheet["actual_seconds"]) == 8
    assert sheet["unique_frames"] == 8
    assert all(0 <= t < .8 for t in sheet["actual_seconds"])
    assert sheet["actual_seconds"][0] == pytest.approx(0)
    assert sheet["actual_seconds"][-1] == pytest.approx(.7)
    assert sheet == make_sheet(clip, .8, tmp_path / "sheets")
    store = Store(tmp_path / "votes.sqlite3", study, signature, tasks)
    register_model(store, "gpt5.6")
    row = assign_task(store, "gpt5.6", next(iter(tasks)))
    # An unusable History should never be decoded/uploaded by the model rater.
    study.cases[0].history.path = "/not-an-input-history.mp4"
    (tmp_path / "inputs").mkdir()
    value = prepare_input(store, row, tmp_path)
    assert list(value["images"]) == ["GT", "A", "B"]
    body = request_body(value, "configured-model")
    assert sum(c["type"] == "image_url" for c in body["messages"][0]["content"]) == 3
    assert "/not-an-input-history" not in json.dumps(body)
    assert "model_a" not in json.dumps(body)


def test_concurrent_assignments_and_idempotent_votes(tmp_path):
    study, signature, tasks = load_study(study_file(tmp_path))
    store = Store(tmp_path / "votes.sqlite3", study, signature, tasks)
    register_model(store, "gpt5.6")
    register_model(store, "gpt5.6")
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda task: assign_task(store, "gpt5.6", task), tasks))
    assert len({r["id"] for r in rows}) == 3
    for row in rows:
        again = assign_task(store, "gpt5.6", row["task"])
        assert again["id"] == row["id"] and again["swapped"] == row["swapped"]
        for _ in range(2):
            store.save("gpt5.6", row["id"], 0, {"choice": "A"}, submit=True)
    assert len(store.records()) == 3
    assert {r["rater"] for r in store.records()} == {"gpt5.6"}


def test_invalid_output_never_becomes_tie():
    def response(content, finish="stop"):
        return {"choices": [{"finish_reason": finish, "message": {"content": content}}]}
    for choice in ("A", "B", "tie"):
        assert parse_choice(response(json.dumps({"choice": choice}))) == choice
    for content in ('{"choice":"Tie"}', '{"choice":"A","reason":"x"}', "A",
                    '{"choice":"A","choice":"B"}', '{"choice":null}', "[]"):
        with pytest.raises(ValueError):
            parse_choice(response(content))
    with pytest.raises(ValueError):
        parse_choice(response('{"choice":"A"}', "length"))


def test_recorded_text_has_case_identity_and_accepts_source_file_newline(tmp_path):
    import hashlib
    path = tmp_path / "generation.json"
    path.write_text(json.dumps({"evaluation_case_id": "case", "future_action": "移开白板",
                               "future_action_sha256": hashlib.sha256("移开白板\n".encode()).hexdigest()}))
    assert recorded_prompt(tmp_path, "Seedance2.0", "case")[:2] == ("移开白板", True)
    assert recorded_prompt(tmp_path, "Seedance2.0", "different-case") == (None, False, None)
    path.unlink()
    (tmp_path / "metadata.json").write_text(json.dumps({"case_name": "other", "prompt": "not this case"}))
    assert recorded_prompt(tmp_path, "LVP", "case") == (None, False, None)


def test_sim_reference_blinding_and_independent_horizons(tmp_path):
    path = study_file(tmp_path)
    data = json.loads(path.read_text())
    data["cases"][0]["gt"]["duration_seconds"] = 1
    data["provenance"] = {"reference_models": ["one", "two"], "model_references": {"case": {
        model: {"clip": {"path": str(tmp_path / "video.mp4"), "duration_seconds": 2},
                "memory_target": f"target-{model}"} for model in ("one", "two")}}}
    path.write_text(json.dumps(data))
    study, signature, tasks = load_study(path)
    store = Store(tmp_path / "votes.sqlite3", study, signature, tasks)
    register_model(store, "gpt5.6")
    (tmp_path / "inputs").mkdir()
    for task in tasks.values():
        row = assign_task(store, "gpt5.6", task["id"])
        value = prepare_input(store, row, tmp_path)
        images = value["images"]
        sim_count = sum(m in ("one", "two") for m in (task["left"], task["right"]))
        assert len(images) == 3 + sim_count
        assert images["GT"]["gt_horizon_seconds"] == 1
        for side, model in (("A", value["model_a"]), ("B", value["model_b"])):
            if model in ("one", "two"):
                assert images[f"GT_{side}"]["gt_horizon_seconds"] == 2
                assert images[side]["sampled_duration_seconds"] == 2
                assert f'"{side} 的对应参考": "GT_{side}（仿真）"' in value["prompt"]
                assert f'"{side} 的目标对象或范围": "target-{model}"' in value["prompt"]
            else:
                assert f"GT_{side}" not in images
                assert images[side]["sampled_duration_seconds"] == 1
        assert sum(c["type"] == "image_url" for c in request_body(value, "model")["messages"][0]["content"]) == len(images)
    del data["provenance"]["model_references"]["case"]["one"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Missing required model reference"):
        load_study(path)


def test_partial_cohort_is_explicit_and_counts_only_possible_tasks(tmp_path):
    path = study_file(tmp_path)
    data = json.loads(path.read_text())
    del data["cases"][0]["predictions"]["three"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="common cohort"):
        load_study(path)
    data["provenance"] = {"allow_partial_models": True}
    path.write_text(json.dumps(data))
    study, _, tasks = load_study(path)
    assert len(tasks) == 1
    assert summarize([], study, replicates=0)["coverage"]["expected_tasks"] == 1


def test_matched_seedance_sample_has_fixed_equal_quotas_and_no_old_cases():
    cases = [f"mine_{i}" for i in range(58)] + [f"case_{i}" for i in range(279)]
    previous = [{"case_id": "mine_0", "left": "Seedance2.0", "right": "Seedance2.5"}]
    plan, eligible = matched_seedance_plan(cases, previous, 200, 20261022)
    assert plan == matched_seedance_plan(cases, previous, 200, 20261022)[0]
    assert len(plan) == len({t["id"] for t in plan}) == 600
    assert "mine_0" not in eligible
    groups = {}
    for task in plan:
        groups.setdefault((task["left"], task["right"]), set()).add(task["case_id"])
    assert len(groups) == 3 and all(len(ids) == 200 for ids in groups.values())
    assert len(set.intersection(*groups.values())) == 200


def test_runner_rejects_future_tail_requests_before_api(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from evaluation.pairwise.model_judge import execute
    path = study_file(tmp_path)
    data = json.loads(path.read_text())
    data["models"][2] = "three_tail"
    data["cases"][0]["predictions"]["three_tail"] = data["cases"][0]["predictions"].pop("three")
    path.write_text(json.dumps(data))
    study, signature, tasks = load_study(path)
    store = Store(tmp_path / "votes.sqlite3", study, signature, tasks)
    config = {"key": "test-only", "llm_name": "model", "max_parrel_requests": 1,
              "openai_base_url": "https://must-not-call.invalid"}
    args = SimpleNamespace(run=tmp_path, rater="gpt5.6", limit=None, preview=False)
    with pytest.raises(ValueError, match="_tail models are excluded"):
        asyncio.run(execute(args, store, config))
