"""Focused protocol, concurrency, privacy, persistence and scoring checks."""

import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from evaluation.pairwise import ANNOTATION_PROTOCOL
from evaluation.pairwise.manifest import load_study, split
from evaluation.pairwise.report import comparisons, export, fit_bt, leaderboard, summarize
from evaluation.pairwise.server import create_app
from evaluation.pairwise.store import Conflict, Store, normalize_vote


@pytest.fixture
def study_file(tmp_path):
    video = tmp_path / "private-model-name.mp4"
    video.write_bytes(bytes(range(256)) * 4)
    clip = {"path": str(video), "duration_seconds": 4.0}
    value = {
        "models": ["model-one", "model-two", "model-three"], "raters_per_pair": 3,
        "cases": [{"case_id": cid, "history": clip, "gt": clip,
                   "predictions": {m: clip for m in ("model-one", "model-two", "model-three")}}
                  for cid in ("drawer_mine_1", "drawer_ood_2", "cup_ood_3")],
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def store(study_file, tmp_path):
    study, signature, tasks = load_study(study_file)
    return Store(tmp_path / "votes.sqlite3", study, signature, tasks)


def answers(choice="A"):
    return {"choice": choice, "elapsed_seconds": 12.0}


def test_only_direct_a_b_tie_choices():
    for choice in ("A", "B", "tie"):
        assert normalize_vote(answers(choice))["choice"] == choice
    for invalid in (None, "unknown", "pass", "Tie", "", 1):
        with pytest.raises(ValueError):
            normalize_vote(answers(invalid))
    with pytest.raises(ValueError):
        normalize_vote({"gates": {}, "preferences": {"overall": "A"}})


def test_resume_draft_blinding_and_revision(store):
    user, code = store.register("reviewer")
    task, _ = store.next(user["id"])
    aid = task["assignment_id"]
    text = json.dumps(task)
    assert "model-one" not in text and "private-model-name" not in text and "case_id" not in text
    draft = answers("tie")
    assert store.save(user["id"], aid, 0, draft) == 1
    reloaded = Store(store.path, store.study, store.signature, store.tasks)
    assert reloaded.register("", code)[0] == user
    task2, _ = reloaded.next(user["id"])
    assert task2["assignment_id"] == aid and task2["draft"] == draft
    assert task2["clips"] == task["clips"]
    with pytest.raises(Conflict):
        store.save(user["id"], aid, 0, draft)
    assert store.save(user["id"], aid, 1, draft, submit=True) == 2
    draft["elapsed_seconds"] += 2
    assert store.save(user["id"], aid, 1, draft, submit=True) == 2
    with pytest.raises(Conflict):
        store.save(user["id"], aid, 2, answers("B"), submit=True)
    assert len(store.records()) == 1
    assert store.next(user["id"])[0]["assignment_id"] != aid
    assert store.path.stat().st_mode & 0o777 == 0o600


def test_concurrent_quota_and_distinct_raters(store):
    # One task and six simultaneous reviewers: only three active assignments.
    first = next(iter(store.tasks))
    store.tasks = {first: store.tasks[first]}
    users = [store.register(f"r{i}")[0] for i in range(6)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda u: store.next(u["id"])[0], users))
    assert sum(t is not None for t in results) == 3
    with store.connection() as db:
        assert db.execute("SELECT COUNT(DISTINCT rater) FROM assignments").fetchone()[0] == 3


def test_expired_lease_issue_and_conflicting_vote(store):
    first = next(iter(store.tasks))
    store.tasks = {first: store.tasks[first]}
    users = [store.register(f"r{i}")[0] for i in range(4)]
    aid = store.next(users[0]["id"])[0]["assignment_id"]
    with store.transaction() as db:
        db.execute("UPDATE assignments SET lease=0")
    for user in users[1:]:
        store.next(user["id"])
    with pytest.raises(Conflict):
        store.save(users[0]["id"], aid, 0, answers(), submit=True)
    with store.transaction() as db:
        db.execute("UPDATE assignments SET lease=0")
    assert store.next(users[0]["id"])[0]["assignment_id"] == aid
    store.issue(users[0]["id"], aid, "GT failed to load")
    assert store.records()[0]["state"] == "issue"
    assert store.next(users[0]["id"])[0] is None


def test_http_range_privacy_and_session_ownership(study_file, tmp_path):
    app = create_app(study_file, tmp_path / "http.sqlite3")
    with TestClient(app) as client, TestClient(app) as outsider:
        assert client.get("/api/next").status_code == 401
        assert client.post("/api/session", json={"name": "evil"}, headers={"Origin": "https://other.example"}).status_code == 403
        assert client.post("/api/session", json={"name": "alice"}).status_code == 200
        task = client.get("/api/next").json()["task"]
        url = task["clips"]["A"]["url"]
        assert outsider.get(url).status_code == 401
        outsider.post("/api/session", json={"name": "bob"})
        assert outsider.get(url).status_code == 404
        response = client.get(url, headers={"Range": "bytes=0-15"})
        assert response.status_code == 206 and len(response.content) == 16
        assert response.headers["content-range"] == "bytes 0-15/1024"
        assert "private" not in response.headers["content-disposition"]
        assert client.head(url).status_code == 200
        assert client.get("/").text.count('name="choice"') == 3
        assert 'id="gate-questions"' not in client.get("/").text
        aid = task["assignment_id"]
        assert client.post(f"/api/task/{aid}/vote", json={"revision": 0, "answers": {"choice": "unknown"}}).status_code == 422
        assert client.post(f"/api/task/{aid}/vote", json={"revision": 0, "answers": {"gates": {}, "preferences": {"overall": "A"}}}).status_code == 422
        assert client.post(f"/api/task/{aid}/draft", json={"revision": 0, "answers": answers()}).json()["revision"] == 1
        assert client.post(f"/api/task/{aid}/vote", json={"revision": 0, "answers": answers()}).status_code == 409
        assert client.post(f"/api/task/{aid}/vote", json={"revision": 1, "answers": answers()}).status_code == 200
        assert client.get("/api/report").status_code == 404
        assert "frame-ancestors 'none'" in client.get("/").headers["content-security-policy"]


def test_study_binding_and_common_scope(study_file, store):
    raw = json.loads(study_file.read_text())
    raw["seed"] = 99
    study_file.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="changed"):
        study, signature, tasks = load_study(study_file)
        Store(store.path, study, signature, tasks)
    assert split("from_mined_data") == "IID"
    assert split("quest_case") == "OOD"


def row(a, b, score, task="t1", rater="r1", case="c1"):
    return dict(a=a, b=b, score=score, task=task, rater=rater, case_id=case)


def test_bt_analytic_halfwins_prior_and_scale():
    assert np.allclose(fit_bt([row("a", "b", .5)], ["a", "b"]), [1500, 1500])
    result = fit_bt([row("a", "b", 1)], ["a", "b"])
    # One actual win plus a virtual draw: adjusted probability 1.5 / 2 = .75.
    assert np.isclose(result[0] - result[1], 400 * math.log10(3), atol=1e-5)
    assert np.isclose(result.mean(), 1500)
    result = fit_bt([row("a", "b", .5)] * 5 + [row("a", "b", 1)] * 3, ["a", "b"])
    assert np.isclose(result[0] - result[1], 400 * math.log10(6 / 3), atol=1e-5)


def test_bt_individual_judgments_components_and_rater_bootstrap():
    rows = [row("a", "b", 1, "t1", f"r{i}", "c1") for i in range(3)] + [row("a", "b", 0, "t2", "r1", "c2")]
    ratings = fit_bt(rows, ["a", "b"])
    assert np.isclose(ratings[0] - ratings[1], 400 * math.log10(3.5 / 1.5), atol=1e-5)
    assert leaderboard(rows, ["a", "b"], 0)["leaderboard"][0]["score_rate"] == .75
    extra = rows + [row("c", "d", .5, "t3", "r1")]
    report = leaderboard(extra, ["a", "b", "c", "d", "unrated"], replicates=20)
    assert not report["global_ranking_available"]
    assert len(report["components"]) == 2 and report["unrated_models"] == ["unrated"]
    assert all(e["rank"] is None for e in report["leaderboard"])
    assert all(e["ci95"] is not None for e in report["leaderboard"] if e["model"] in ("a", "b"))
    assert leaderboard(rows[:1], ["a", "b"], 20)["leaderboard"][0]["ci95"] is None
    assert report == leaderboard(extra, ["a", "b", "c", "d", "unrated"], replicates=20)
    tied = leaderboard([row("a", "b", .5, rater="r1"), row("a", "b", .5, rater="r2")], ["a", "b"], 20)
    assert all(e["top1_probability_in_component"] == .5 for e in tied["leaderboard"])


def test_tie_scoring_and_media_issue_exclusion(store, tmp_path):
    user, _ = store.register("r1")
    task = store.next(user["id"])[0]
    store.save(user["id"], task["assignment_id"], 0, answers("tie"), submit=True)
    task = store.next(user["id"])[0]
    store.issue(user["id"], task["assignment_id"], "bad history")
    records = store.records()
    assert len(comparisons(records)) == 1 and comparisons(records)[0]["score"] == .5
    summary = summarize(records, store.study, replicates=0)
    group = summary["groups"]["Overall"]
    assert group["completed_judgments"] == 1 and group["media_issues"] == 1
    assert all(e["elo"] == 1500 and e["score_rate"] == .5 and e["games"] == 1 for e in group["leaderboard"])
    assert summary["protocol"]["id"] == ANNOTATION_PROTOCOL
    assert "dimensions" not in group and "gates" not in group
    assert not summary["coverage"]["complete"]
    export(tmp_path / "export", records, summary)
    assert (tmp_path / "export/report.html").is_file()
    assert "resume_code" not in (tmp_path / "export/votes.json").read_text()
    assert not (tmp_path / "export/gates.csv").exists()
    assert "choice" in (tmp_path / "export/votes.csv").read_text()


def test_old_drafts_upgrade_without_changing_participant_or_mapping(store):
    user, code = store.register("existing")
    task = store.next(user["id"])[0]
    with store.transaction() as db:
        db.execute("DELETE FROM metadata WHERE key='annotation_protocol'")
        db.execute("UPDATE assignments SET draft=?", (json.dumps({"gates": {}, "preferences": {"overall": "A"}}),))
    updated = Store(store.path, store.study, store.signature, store.tasks)
    resumed, _ = updated.next(updated.register("", code)[0]["id"])
    assert resumed["assignment_id"] == task["assignment_id"]
    assert resumed["clips"] == task["clips"]
    assert resumed["draft"] == {} and resumed["revision"] == task["revision"] + 1


def test_legacy_votes_are_preserved_and_cannot_enter_new_report(store):
    user, _ = store.register("old-rater")
    task = store.next(user["id"])[0]
    legacy = {"gates": {}, "preferences": {"overall": "A"}}
    with store.transaction() as db:
        db.execute("DELETE FROM metadata WHERE key='annotation_protocol'")
        db.execute("UPDATE assignments SET state='voted',vote=?", (json.dumps(legacy),))
    with pytest.raises(ValueError, match="Legacy"):
        Store(store.path, store.study, store.signature, store.tasks)
    with store.connection() as db:
        assert json.loads(db.execute("SELECT vote FROM assignments").fetchone()[0]) == legacy
    with pytest.raises(ValueError):
        comparisons(store.records())


def test_nonempty_leaderboard_json_export(store, tmp_path):
    for name in ("reviewer-1", "reviewer-2"):
        user, _ = store.register(name)
        task = store.next(user["id"])[0]
        store.save(user["id"], task["assignment_id"], 0, answers("A"), submit=True)
    records = store.records()
    summary = summarize(records, store.study, replicates=10)
    json.dumps(summary, allow_nan=False)
    export(tmp_path / "export-nonempty", records, summary)
    assert "model-" in (tmp_path / "export-nonempty/report.html").read_text()


def test_prepare_uses_actual_prompts_and_detects_gt_mismatch(study_file, tmp_path, monkeypatch):
    from argparse import Namespace
    from evaluation.pairwise import prepare

    monkeypatch.setattr(prepare, "probe", lambda p, start=0: {"path": str(p), "duration_seconds": 4, "start_seconds": start})
    video = str(tmp_path / "private-model-name.mp4")
    lists = []
    for model in ("m1", "m2"):
        source = tmp_path / (model + ".json")
        source.write_text(json.dumps([dict(case_id="c", gt_video=video, gen_video=video, history_video=video, text_prompt="actual input")]))
        lists.append((model, source))
    args = Namespace(model=lists, case_id=None, seed=42, limit=0, prompts=None,
                     title="test", baseline=None, raters_per_pair=3, output=tmp_path / "out.json")
    assert prepare.build(args).cases[0].predictions["m1"].prompt == "actual input"
    data = json.loads(lists[1][1].read_text()); data[0]["gt_start_seconds"] = 1
    lists[1][1].write_text(json.dumps(data)); args.output = tmp_path / "other.json"
    with pytest.raises(ValueError, match="Different GT"):
        prepare.build(args)
