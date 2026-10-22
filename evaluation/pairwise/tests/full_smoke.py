"""Validate a full human study using a NEW isolated database and real media.

Scripted votes are plumbing checks, never human annotations. Optionally exercise
the same isolated server with browser_full_smoke.py in a Playwright environment.
"""

import argparse
from collections import deque
import itertools
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

from fastapi.testclient import TestClient
import uvicorn

from evaluation.pairwise.server import create_app
from evaluation.pairwise import store as store_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--browser-python', type=Path)
    parser.add_argument('--browser-only', action='store_true', help='Rerun browser checks without repeating the API traversal')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    database = output / 'SCRIPTED-VALIDATION-ONLY.sqlite3'
    app = create_app(args.study, database, random_assignment=True)
    store = app.state.store
    assert store.study.provenance['deployment_ready']
    # Check the full source union rather than the intersection or a sampled plan.
    records = json.loads((args.predictions / '.metadata/manifest.json').read_text())['predictions']
    expected = {}
    for item in records:
        if not item['method'].lower().endswith('_tail'):
            expected.setdefault(item['case_name'], set()).add(item['method'])
    actual = {case.case_id: set(case.predictions) for case in store.study.cases}
    assert actual == expected, 'Some valid source case/model predictions are absent'
    assert store.study.baseline is None and 'selected_task_ids' not in store.study.provenance
    assert len(store.tasks) == sum(len(models) * (len(models) - 1) // 2 for models in expected.values())
    by_pair = {}
    for task in store.tasks.values():
        by_pair.setdefault((task['left'], task['right']), []).append(task['id'])
    assert set(by_pair) == set(itertools.combinations(sorted(store.study.models), 2))
    # Vary cases across every model pair to avoid validating only the old demo cases.
    plan = [values[i % len(values)] for i, (_, values) in enumerate(sorted(by_pair.items()))]
    if args.browser_only:
        if not args.browser_python:
            parser.error('--browser-only requires --browser-python')
        plan = []
    sequence = deque(plan)
    original_choice = store_module.secrets.choice
    store_module.secrets.choice = lambda values: sequence.popleft()
    receipt = {'validation_only': True, 'models': len(store.study.models),
               'cases': len(store.study.cases), 'total_tasks': len(store.tasks),
               'tested_model_pairs': len(plan), 'media_range_checks': 0}
    server = None
    try:
        with store.transaction() as db:
            db.execute("INSERT INTO metadata VALUES('validation_only','SCRIPTED TEST VOTES; NOT HUMAN PREFERENCES')")
        with TestClient(app) as client:
            client.post('/api/session', json={'name': 'SCRIPTED-API-VALIDATION'}).raise_for_status()
            previous_aid = None
            for i, task_id in enumerate(plan):
                result = client.get('/api/next').json()
                task = result['task']
                assert task and result['progress']['random_assignment']
                aid = task['assignment_id']
                assert aid != previous_aid
                with store.connection() as db:
                    row = db.execute('SELECT task FROM assignments WHERE id=?', (aid,)).fetchone()
                assert row['task'] == task_id
                assert client.get('/api/next').json()['task']['assignment_id'] == aid
                for clip in [*task['clips'].values(), *task['references'].values()]:
                    response = client.get(clip['url'], headers={'Range': 'bytes=0-1023'})
                    assert response.status_code == 206 and len(response.content) == 1024
                    assert response.headers['content-type'].startswith('video/mp4')
                    receipt['media_range_checks'] += 1
                answer = {'choice': ('A', 'B', 'tie')[i % 3], 'elapsed_seconds': 1}
                client.post(f'/api/task/{aid}/draft', json={'revision': 0, 'answers': answer}).raise_for_status()
                revision = client.get('/api/next').json()['task']['revision']
                client.post(f'/api/task/{aid}/vote', json={'revision': revision, 'answers': answer}).raise_for_status()
                client.post(f'/api/task/{aid}/vote', json={'revision': revision, 'answers': answer}).raise_for_status()
                previous_aid = aid
            assert len(store.records()) == len(plan)
        # Browser samples include real/real, real/SIM and SIM/SIM references.
        browser_pairs = [
            ('DWM', 'Hand2World'), ('DWM', 'GE-SIM'), ('DreamDojo', 'GE-SIM'),
            ('Seedance2.0', 'Wan2.1-VACE-14B')]
        covered = {model for pair in browser_pairs for model in pair}
        remaining = sorted(set(store.study.models) - covered)
        while remaining:
            a = remaining.pop(0)
            b = remaining.pop(0) if remaining else 'DWM'
            browser_pairs.append(tuple(sorted((a, b))))
        assert {m for pair in browser_pairs for m in pair} == set(store.study.models)
        browser_plan = [by_pair[pair][0] for pair in browser_pairs]
        sequence.extend(browser_plan + [browser_plan[0]])
        if args.browser_python:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning', access_log=False))
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            for _ in range(100):
                if server.started:
                    break
                time.sleep(.1)
            assert server.started
            subprocess.run([str(args.browser_python), str(Path(__file__).with_name('browser_full_smoke.py')),
                            '--url', f'http://127.0.0.1:{port}', '--output', str(output),
                            '--count', str(len(browser_plan))], check=True, env=os.environ.copy())
            receipt['browser'] = json.loads((output / 'browser-receipt.json').read_text())
        receipt['status'] = 'passed'
    finally:
        store_module.secrets.choice = original_choice
        if server:
            server.should_exit = True
            thread.join(timeout=10)
        (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
