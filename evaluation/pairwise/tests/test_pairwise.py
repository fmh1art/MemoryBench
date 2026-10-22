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


def test_human_snapshot_is_readonly_anonymous_and_retains_repeat_votes(store, tmp_path):
    from evaluation.pairwise.human_report import export_snapshot
    first = next(iter(store.tasks))
    store.tasks = {first: store.tasks[first]}
    store = Store(store.path, store.study, store.signature, store.tasks, random_assignment=True)
    user, code = store.register('PRIVATE-REVIEWER-NAME')
    for choice in ('A', 'B'):
        task, _ = store.next(user['id'])
        store.save(user['id'], task['assignment_id'], 0, answers(choice), submit=True)
    draft, _ = store.next(user['id'])
    store.save(user['id'], draft['assignment_id'], 0, answers('tie'))
    original = store.records()
    destination = tmp_path / 'public-human-snapshot'
    receipt = export_snapshot(store.path, destination)
    assert receipt['votes'] == 2 and receipt['repeated_rater_tasks_retained'] == 1
    assert receipt['drafts_excluded'] == 1 and receipt['active_raters'] == 1
    public = json.loads((destination / 'votes.json').read_text())
    assert len({r['id'] for r in public}) == 2 and len({r['rater'] for r in public}) == 1
    assert sorted((r['model_a'], r['model_b'], r['vote']['choice']) for r in public) == sorted(
        (r['model_a'], r['model_b'], r['vote']['choice']) for r in original)
    text = (destination / 'votes.json').read_text() + (destination / 'snapshot.json').read_text()
    for private in ('PRIVATE-REVIEWER-NAME', code, store.secret, user['id'], 'secret_hash'):
        assert private not in text
    assert store.records() == original
    assert store.next(user['id'])[0]['assignment_id'] == draft['assignment_id']


def test_pooled_elo_equal_vote_weight_and_historical_deduplication(tmp_path):
    import hashlib
    from evaluation.pairwise.human_report import compare, score
    source, human, output = tmp_path / 'vlm', tmp_path / 'human', tmp_path / 'out'
    (source / 'report').mkdir(parents=True)
    human.mkdir()
    votes = [{'id': str(i), 'rater': f'vlm-{i}', 'task': 'same-task', 'state': 'voted',
              'case_id': 'case', 'split': 'OOD', 'model_a': 'one', 'model_b': 'two',
              'vote': answers('A')} for i in range(2)]
    baseline = score(votes, ['one', 'two'], 'baseline', 1, False)
    (source / 'protocol.json').write_text(json.dumps({'id': ANNOTATION_PROTOCOL}))
    (source / 'report/votes.json').write_text(json.dumps(votes))
    (source / 'report/summary.json').write_text(json.dumps(baseline))
    extra = [dict(votes[0], id='human-vote-1', rater='human-1', vote=answers('B'))]
    (human / 'votes.json').write_text(json.dumps(extra))
    (human / 'snapshot.json').write_text(json.dumps({'annotation_protocol': ANNOTATION_PROTOCOL,
        'study_tasks': 1, 'active_raters': 1, 'captured_at': '2026-10-22T20:00:00+08:00',
        'votes_sha256': hashlib.sha256((human / 'votes.json').read_bytes()).hexdigest()}))
    result = compare(source, human, output)
    assert result['vlm_judgments'] == 2 and result['pooled_judgments'] == 3
    row = next(row for row in result['comparison'] if row['model'] == 'one')
    assert row['vlm_elo'] > row['pooled_elo'] > 1500
    assert abs(sum(row['pooled_elo'] for row in result['comparison']) / 2 - 1500) < 1e-8
    assert sum(row['pooled_games'] for row in result['comparison']) == 6
    assert result['protocol']['bootstrap_replicates'] == 0
    (source / 'report/votes.json').write_text(json.dumps(votes + votes[:1]))
    with pytest.raises(ValueError, match='Duplicate imported record IDs'):
        compare(source, human, tmp_path / 'duplicate-out')


def test_random_repeated_votes_preserve_old_rows_and_drafts(store, monkeypatch):
    user, code = store.register('repeat-reviewer')
    first, _ = store.next(user['id'])
    store.save(user['id'], first['assignment_id'], 0, answers('A'), submit=True)
    pending, _ = store.next(user['id'])
    store.save(user['id'], pending['assignment_id'], 0, answers('tie'))
    reloaded = Store(store.path, store.study, store.signature, store.tasks, random_assignment=True)
    assert reloaded.session(code) == user
    resumed, _ = reloaded.next(user['id'])
    assert resumed['assignment_id'] == pending['assignment_id']
    assert resumed['draft'] == answers('tie')
    reloaded.save(user['id'], resumed['assignment_id'], 1, answers('tie'), submit=True)
    task_id = next(iter(store.tasks))
    monkeypatch.setattr('evaluation.pairwise.store.secrets.choice', lambda values: task_id)
    ids = {first['assignment_id'], resumed['assignment_id']}
    for _ in range(12):
        task, progress = reloaded.next(user['id'])
        assert task['assignment_id'] not in ids
        ids.add(task['assignment_id'])
        assert not progress['personal_complete'] and progress['remaining_tasks'] is None
        with ThreadPoolExecutor(max_workers=3) as pool:
            concurrent = list(pool.map(lambda _: reloaded.next(user['id'])[0], range(3)))
        assert {t['assignment_id'] for t in concurrent} == {task['assignment_id']}
        reloaded.save(user['id'], task['assignment_id'], 0, answers('B'), submit=True)
        reloaded.save(user['id'], task['assignment_id'], 0, answers('B'), submit=True)
    records = reloaded.records()
    assert len(records) == 14
    assert records[0]['vote'] == answers('A')
    assert len({r['id'] for r in records}) == 14
    assert sum(r['task'] == task_id for r in records) >= 12


def test_simulation_references_are_blinded_and_session_protected(study_file, tmp_path):
    value = json.loads(study_file.read_text())
    references = {}
    for case in value['cases']:
        references[case['case_id']] = {}
        for model in value['models']:
            path = tmp_path / f"{case['case_id']}-{model}.mp4"
            path.write_bytes(f'simulation reference {model}'.encode())
            references[case['case_id']][model] = {'clip': {'path': str(path), 'duration_seconds': 3},
                                                'memory_target': 'simulation target'}
    value['provenance'] = {'model_references': references, 'reference_models': value['models']}
    study_file.write_text(json.dumps(value))
    app = create_app(study_file, tmp_path / 'references.sqlite3', random_assignment=True)
    with TestClient(app) as client:
        client.post('/api/session', json={'name': 'sim-reviewer'})
        task = client.get('/api/next').json()['task']
        assert set(task['references']) == {'A', 'B'}
        assert 'model-one' not in json.dumps(task)
        for reference in task['references'].values():
            assert reference['memory_target'] == 'simulation target'
            assert client.get(reference['url']).content.startswith(b'simulation reference ')
        with TestClient(app) as stranger:
            assert stranger.get(task['references']['A']['url']).status_code == 401


def test_simulation_timeline_preserves_long_prediction_without_changing_real_tasks(store):
    key, task = next((key, task) for key, task in store.tasks.items() if 'model-one' in (task['left'], task['right']))
    store.tasks = {key: task}
    case = task['case']
    case.predictions['model-one'].duration_seconds = 9.5
    case.predictions['model-one'].start_seconds = .5
    store.study.provenance['model_references'] = {
        case.case_id: {'model-one': {'clip': case.gt.model_dump()}}}
    user, _ = store.register('long-simulation')
    assigned, _ = store.next(user['id'])
    assert assigned['timeline_seconds'] == 9
    master = assigned['playback_master']
    assert master in ('A', 'B')
    assert assigned['clips'][master]['duration_seconds'] - assigned['clips'][master]['start_seconds'] == 9
    with store.connection() as db:
        row = store.assigned(db, user['id'], assigned['assignment_id'])
    store.study.provenance['model_references'] = {}
    real = store.public_task(row)
    assert real['timeline_seconds'] == 4 and real['playback_master'] == 'gt'


def test_human_study_all_non_tail_partial_cases_and_missing_sim_reference(study_file, tmp_path, monkeypatch):
    from argparse import Namespace
    from evaluation.pairwise.human_study import prepare
    from evaluation.pairwise.restart_demo import validate_extension

    root, data = tmp_path / 'predictions', tmp_path / 'data'
    for group in ('VisualSignal', 'TextControl', 'DirectAction'):
        (root / group).mkdir(parents=True)
    models = json.loads(study_file.read_text())['models']
    for model in [*models, 'model-four', 'model-four_tail']:
        (root / 'VisualSignal' / model).mkdir()
    entries = []
    for i, case in enumerate(json.loads(study_file.read_text())['cases']):
        cid = case['case_id']
        for name in ('input/rgb_prefix.mp4', 'ground_truth/future.mp4'):
            file = data / cid / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'video')
        for model in [*models, 'model-four_tail', *(['model-four'] if i == 0 else [])]:
            video = root / 'VisualSignal' / model / f'{cid}.mp4'
            video.write_bytes(b'video')
            entries.append({'method': model, 'case_name': cid, 'benchmark_case_directory': cid,
                            'prediction': str(video.relative_to(root)), 'other_files': 'missing'})
    (root / '.metadata').mkdir()
    manifest = root / '.metadata/manifest.json'
    manifest.write_text(json.dumps({'predictions': entries}))
    jobs = tmp_path / 'jobs.json'
    jobs.write_text('[]')
    monkeypatch.setattr('evaluation.pairwise.human_study.probe',
                        lambda path: {'path': str(path), 'duration_seconds': 4, 'start_seconds': 0})
    args = Namespace(predictions=root, data=data, jobs=jobs, source=study_file,
                     output=tmp_path / 'full/study.json', sim_data=None, workers=2)
    prepare(args)
    study, _, tasks = load_study(args.output)
    assert len(study.cases) == 3 and len(tasks) == 12
    assert set(study.models) == {*models, 'model-four'}
    assert study.provenance['deployment_ready']
    _, _, old_tasks = load_study(study_file)
    validate_extension(old_tasks, tasks)
    changed = next(iter(tasks.values()))['case']
    changed.memory_target = 'changed meaning'
    with pytest.raises(ValueError, match='scope changed'):
        validate_extension(old_tasks, tasks)

    # A missing simulation GT produces an audit and blocks deployment, rather than silently dropping a model.
    (root / 'DirectAction/GE-SIM').mkdir()
    entry = dict(entries[0], method='GE-SIM')
    manifest.write_text(json.dumps({'predictions': [*entries, entry]}))
    jobs.write_text(json.dumps([{'model': 'GE-SIM', 'case': {'case_id': entry['case_name'],
        'gt_video': '/old/MemoryDataV1-Sim/case/prediction_gt.mp4'}}]))
    args.output = tmp_path / 'incomplete/study.json'
    with pytest.raises(ValueError, match='missing model references'):
        prepare(args)
    incomplete, _, _ = load_study(args.output)
    assert incomplete.provenance['deployment_ready'] is False
    audit = json.loads(args.output.with_suffix('.audit.json').read_text())
    assert any(m['model'] == 'GE-SIM' and m['reason'] == 'missing simulation GT' for m in audit['missing'])

    # A reference published beside the prediction is bound by its current provenance,
    # rather than a stale jobs record that pointed at a different GT file.
    (root / 'missing').mkdir()
    (root / 'missing/gt.mp4').write_bytes(b'simulation GT')
    (root / 'evalsim').mkdir()
    entry.update(evaluation_results='evalsim', prediction_sha256='current-prediction-hash')
    manifest.write_text(json.dumps({'predictions': [*entries, entry]}))
    (root / 'evalsim/provenance.json').write_text(json.dumps({
        'model': 'GE-SIM', 'case_id': entry['case_name'], 'ground_truth_type': 'simulation',
        'matches_current_prediction': True, 'current_prediction_sha256': 'current-prediction-hash',
        'ground_truth': '/old/MemoryBenchPredictionsV1/missing/gt.mp4'}))
    args.output = tmp_path / 'with-sim/study.json'
    prepare(args)
    complete, _, complete_tasks = load_study(args.output)
    assert complete.provenance['deployment_ready']
    assert len(complete_tasks) == 16
    assert complete.provenance['reference_models'] == ['GE-SIM']


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


def test_open_assignment_allows_concurrent_and_later_reviewers(store):
    first = next(iter(store.tasks))
    store.tasks = {first: store.tasks[first]}
    store.open_assignment = True
    users = [store.register(f"open-{i}")[0] for i in range(6)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        claimed = list(pool.map(lambda user: store.next(user["id"])[0], users))
    assert all(claimed)
    for user, task in zip(users, claimed):
        store.save(user["id"], task["assignment_id"], 0, answers(), submit=True)
        next_task, progress = store.next(user["id"])
        assert next_task is None
        assert progress["personal_complete"] and progress["remaining_tasks"] == 0
    assert len(store.records()) == 6
    later, _ = store.register("later-reviewer")
    task, progress = store.next(later["id"])
    assert task is not None
    assert progress["study_complete"] and not progress["personal_complete"]
    assert progress["open_assignment"] and progress["remaining_tasks"] == 1


def test_extra_votes_on_one_pair_do_not_complete_other_pairs(store):
    original_tasks = store.tasks
    first = next(iter(original_tasks))
    store.tasks = {first: original_tasks[first]}
    store.open_assignment = True
    for i in range(6):
        user, _ = store.register(f"extra-{i}")
        task = store.next(user["id"])[0]
        store.save(user["id"], task["assignment_id"], 0, answers(), submit=True)
    two = list(original_tasks)[:2]
    store.tasks = {key: original_tasks[key] for key in two}
    fresh, _ = store.register("fresh-reviewer")
    task, progress = store.next(fresh["id"])
    assert task is not None and not progress["study_complete"]


def test_switch_to_open_assignment_preserves_votes_drafts_and_resume(store):
    voted, voted_code = store.register("existing-voter")
    task = store.next(voted["id"])[0]
    store.save(voted["id"], task["assignment_id"], 0, answers("B"), submit=True)
    drafting, draft_code = store.register("existing-draft")
    draft_task = store.next(drafting["id"])[0]
    store.save(drafting["id"], draft_task["assignment_id"], 0, answers("tie"))
    before = store.records()
    reopened = Store(store.path, store.study, store.signature, store.tasks,
                     open_assignment=True)
    assert reopened.records() == before
    assert reopened.register("", voted_code)[0] == voted
    assert reopened.register("", draft_code)[0] == drafting
    resumed = reopened.next(drafting["id"])[0]
    assert resumed["assignment_id"] == draft_task["assignment_id"]
    assert resumed["clips"] == draft_task["clips"]
    assert resumed["draft"] == answers("tie") and resumed["revision"] == 1


def test_open_assignment_http_configuration_and_personal_completion(study_file, tmp_path):
    app = create_app(study_file, tmp_path / "open-http.sqlite3", open_assignment=True)
    with TestClient(app) as client:
        assert client.get("/api/config").json()["open_assignment"] is True
        client.post("/api/session", json={"name": "reviewer"})
        for _ in range(9):
            task = client.get("/api/next").json()["task"]
            assert task
            response = client.post(f"/api/task/{task['assignment_id']}/vote",
                                   json={"revision": 0, "answers": answers()})
            assert response.status_code == 200
        response = client.get("/api/next").json()
        assert response["task"] is None
        assert response["progress"]["personal_complete"]
        assert response["progress"]["completed"] == 9
        assert not response["progress"]["study_complete"]


@pytest.mark.parametrize('expand', [False, True])
def test_backup_restart_preserves_live_votes_session_and_draft(study_file, tmp_path, expand):
    import os
    import signal
    import socket
    import subprocess
    import sys
    import time
    from argparse import Namespace
    from pathlib import Path

    import httpx

    from evaluation.pairwise.restart_demo import restart, stopped, wait_ready

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    database = tmp_path / "live.sqlite3"
    args = Namespace(study=study_file, db=database, host="127.0.0.1", port=port,
                     log=tmp_path / "server.log", receipt=tmp_path / "status.json")
    process = subprocess.Popen(
        [sys.executable, "-m", "evaluation.pairwise.server", "--study", str(study_file),
         "--db", str(database), "--host", args.host, "--port", str(port)],
        cwd=Path(__file__).resolve().parents[3], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True)
    new_pid = None
    try:
        _, signature, _ = load_study(study_file)
        wait_ready(process, args.host, port, signature, require_open=False)
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False) as client:
            client.post("/api/session", json={"name": "existing-human"}).raise_for_status()
            first = client.get("/api/next").json()["task"]
            client.post(f"/api/task/{first['assignment_id']}/vote",
                        json={"revision": 0, "answers": answers("B")}).raise_for_status()
            draft = client.get("/api/next").json()["task"]
            client.post(f"/api/task/{draft['assignment_id']}/draft",
                        json={"revision": 0, "answers": answers("tie")}).raise_for_status()
            args.pid = process.pid
            if expand:
                expanded = json.loads(study_file.read_text())
                expanded['models'].append('model-four')
                for case in expanded['cases']:
                    case['predictions']['model-four'] = dict(case['predictions']['model-one'])
                args.new_study = tmp_path / 'expanded.json'
                args.new_study.write_text(json.dumps(expanded))
                args.random_assignment = True
            receipt = restart(args)
            new_pid = receipt["server_pid"]
            assert receipt["before"]["votes"] == receipt["after"]["votes"] == 1
            assert receipt["existing_records_preserved"]
            assert client.get("/api/session").json()["name"] == "existing-human"
            assert client.get("/api/config").json()["open_assignment"]
            if expand:
                assert client.get('/api/config').json()['random_assignment']
                assert client.get('/api/config').json()['total_tasks'] == 18
            resumed = client.get("/api/next").json()["task"]
            assert resumed["assignment_id"] == draft["assignment_id"]
            assert resumed["draft"] == answers("tie")
            assert resumed["clips"] == draft["clips"]
            backup = Path(receipt["backup"])
            assert backup.is_file() and (backup.parent / "before-stop.sqlite3").is_file()
            with sqlite3.connect(backup) as db:
                assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                assert db.execute("SELECT COUNT(*) FROM assignments WHERE state='voted'").fetchone()[0] == 1
            assert "resume_code" not in args.receipt.read_text()
            assert "existing-human" not in args.receipt.read_text()
    finally:
        if new_pid and not stopped(new_pid):
            os.kill(new_pid, signal.SIGTERM)
            for _ in range(50):
                if stopped(new_pid):
                    break
                time.sleep(.1)
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)


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
