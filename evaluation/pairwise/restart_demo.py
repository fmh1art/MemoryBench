"""Back up and restart the root-owned human demo without replacing its database."""

import argparse
import json
import itertools
import os
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .manifest import digest, load_study, model_reference


def rater_stats(path, name):
    """Read aggregate annotation progress without exposing identities, sessions or choices."""
    with connect_readonly(path) as db:
        db.execute('BEGIN')
        raters = db.execute('SELECT id FROM raters WHERE lower(trim(name))=lower(trim(?))', (name,)).fetchall()
        if not raters:
            return {'found': False}
        if len(raters) != 1:
            raise ValueError('Ambiguous rater name')
        study = json.loads(db.execute("SELECT value FROM metadata WHERE key='study'").fetchone()[0])
        rows = db.execute('SELECT task,state,updated FROM assignments WHERE rater=?', (raters[0][0],)).fetchall()
    task_cases = {digest([case['case_id'], a, b])[:24]: case['case_id']
                  for case in study['cases'] for a, b in itertools.combinations(sorted(case['predictions']), 2)}
    voted = [row for row in rows if row[1] == 'voted']
    distinct = {row[0] for row in voted}
    return {'found': True, 'votes': len(voted), 'distinct_comparisons': len(distinct),
            'distinct_cases': len({task_cases.get(key, key) for key in distinct}),
            'repeated_comparisons': len(voted) - len(distinct),
            'drafts': sum(row[1] == 'assigned' for row in rows),
            'issues': sum(row[1] == 'issue' for row in rows),
            'study_cases': len(study['cases']), 'study_models': len(study['models']),
            'last_vote_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(max(row[2] for row in voted))) if voted else None}


def latest_rater_record(path, name):
    """Resolve the latest submitted case and displayed prompts, without session secrets."""
    with connect_readonly(path) as db:
        db.execute('BEGIN')
        row = db.execute("""SELECT a.task,a.swapped,a.updated,a.rater FROM assignments a
            JOIN raters r ON r.id=a.rater WHERE lower(trim(r.name))=lower(trim(?))
            AND a.state='voted' ORDER BY a.updated DESC,a.created DESC LIMIT 1""", (name,)).fetchone()
        if row is None:
            return {'found': False}
        total = db.execute("SELECT COUNT(*) FROM assignments WHERE rater=? AND state='voted'", (row[3],)).fetchone()[0]
        study = json.loads(db.execute("SELECT value FROM metadata WHERE key='study'").fetchone()[0])
    for case in study['cases']:
        for left, right in itertools.combinations(sorted(case['predictions']), 2):
            if digest([case['case_id'], left, right])[:24] == row[0]:
                a, b = (right, left) if row[1] else (left, right)
                return {'found': True, 'submitted_count': total, 'case_id': case['case_id'],
                        'submitted_at_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(row[2])),
                        'action_description': case.get('action_description', ''),
                        'prompts': {slot: {'recorded': case['predictions'][model].get('prompt_recorded', False),
                                           'text': case['predictions'][model].get('prompt')}
                                    for slot, model in (('A', a), ('B', b))}}
    raise ValueError('Latest assignment is absent from the stored study')


def validate_extension(old_tasks, new_tasks, old_study=None, new_study=None):
    """Only append comparisons; existing votes must retain their exact interpretation."""
    if old_tasks.keys() - new_tasks.keys():
        raise ValueError('New study removes existing tasks')
    for key, old in old_tasks.items():
        new = new_tasks[key]
        if any(old[k] != new[k] for k in ('left', 'right')):
            raise ValueError('Existing model mapping changed')
        old_case, new_case = old['case'].model_dump(), new['case'].model_dump()
        for model in (old['left'], old['right']):
            if old_case['predictions'][model] != new_case['predictions'][model]:
                raise ValueError('Existing prediction or prompt changed')
            if old_study is not None and model_reference(old_study, old_case['case_id'], model) != model_reference(new_study, new_case['case_id'], model):
                raise ValueError('Existing model reference changed')
        old_case.pop('predictions')
        new_case.pop('predictions')
        if old_case != new_case:
            raise ValueError('Existing reference or evaluation scope changed')


def connect_readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def snapshot(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with connect_readonly(source) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)
        if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Backup integrity check failed")
    destination.chmod(0o600)


def counts(path):
    with connect_readonly(path) as db:
        db.execute('BEGIN')
        states = dict(db.execute("SELECT state,COUNT(*) FROM assignments GROUP BY state"))
        return {"raters": db.execute("SELECT COUNT(*) FROM raters").fetchone()[0],
                "votes": states.get("voted", 0), "draft_assignments": states.get("assigned", 0),
                "issues": states.get("issue", 0)}


def rater_counts(path):
    """List reviewer names and submitted counts only, excluding session secrets."""
    with connect_readonly(path) as db:
        db.execute('BEGIN')
        rows = db.execute("""
            SELECT r.name, COUNT(a.id) AS votes
            FROM raters r LEFT JOIN assignments a ON a.rater=r.id AND a.state='voted'
            GROUP BY r.id,r.name ORDER BY votes DESC,r.name
        """).fetchall()
        return {'raters': [{'name': name, 'votes': votes} for name, votes in rows],
                'total_votes': sum(votes for _, votes in rows)}


def preserved_state(path):
    with connect_readonly(path) as db:
        return {
            "metadata": dict(db.execute("SELECT key,value FROM metadata WHERE key IN ('signature','secret','annotation_protocol')")),
            "raters": dict((r[0], r[1:]) for r in db.execute("SELECT id,name,secret_hash FROM raters")),
            "assignments": dict((r[0], r[1:]) for r in db.execute("SELECT id,rater,task,swapped FROM assignments")),
            "votes": dict((r[0], r[1:]) for r in db.execute("SELECT id,vote FROM assignments WHERE state='voted'")),
            "drafts": dict((r[0], r[1:]) for r in db.execute("SELECT id,draft,revision,state FROM assignments WHERE state='assigned'")),
            "issues": dict((r[0], r[1:]) for r in db.execute("SELECT id,issue FROM assignments WHERE state='issue'")),
        }


def stopped(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state == "Z"
    except FileNotFoundError:
        return True


def wait_ready(process, host, port, signature, *, require_open=True):
    host = "127.0.0.1" if host == "0.0.0.0" else host
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(3000):
        if process.poll() is not None:
            break
        try:
            with opener.open(f"http://{host}:{port}/api/config", timeout=1) as response:
                config = json.load(response)
            if config["study_id"] == signature[:16] and (not require_open or config.get("open_assignment")):
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(.2)
    raise RuntimeError("Demo health check failed; inspect the server log")


def restart(args):
    study_path, db_path = args.study.resolve(), args.db.resolve()
    if not db_path.is_file():
        raise ValueError("Existing database is required; refusing to create an empty voting database")
    command = Path(f"/proc/{args.pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
    cwd = Path(f"/proc/{args.pid}/cwd").resolve()
    def argument(name):
        return command[command.index(name) + 1]
    def original_path(name):
        return (cwd / argument(name)).resolve()
    if ("evaluation.pairwise.server" not in command or original_path("--db") != db_path or
            original_path("--study") != study_path or int(argument("--port")) != args.port or
            argument("--host") != args.host):
        raise ValueError("PID does not match the requested demo; nothing was stopped")
    old_study, signature, old_tasks = load_study(study_path)
    new_study_path = getattr(args, 'new_study', None)
    new_study_path = new_study_path.resolve() if new_study_path else study_path
    new_study, new_signature, new_tasks = load_study(new_study_path)
    if new_study.provenance.get('deployment_ready') is False:
        raise ValueError('Expanded study is incomplete; supply missing simulation references before deployment')
    validate_extension(old_tasks, new_tasks, old_study, new_study)
    random_assignment = getattr(args, 'random_assignment', False)
    with connect_readonly(db_path) as db:
        saved = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        if not saved or saved[0] != signature:
            raise ValueError("Study signature differs; preserving the running service and existing database")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    backup_dir = db_path.parent / "backups" / ("open-assignment-" + stamp)
    snapshot(db_path, backup_dir / "before-stop.sqlite3")
    os.kill(args.pid, signal.SIGTERM)
    for _ in range(150):
        if stopped(args.pid):
            break
        time.sleep(.1)
    else:
        raise RuntimeError("Old server did not stop; backup saved, no replacement server started")
    final_backup = backup_dir / "after-stop.sqlite3"
    snapshot(db_path, final_backup)
    before = preserved_state(final_backup)
    before_counts = counts(final_backup)
    with connect_readonly(final_backup) as db:
        old_metadata = dict(db.execute('SELECT key,value FROM metadata'))
    def set_metadata(values):
        with sqlite3.connect(db_path) as db:
            for key, value in values.items():
                if value is None:
                    db.execute('DELETE FROM metadata WHERE key=?', (key,))
                else:
                    db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', (key, value))
    migration = {'signature': new_signature, 'study': new_study.model_dump_json(),
                 'cookie_namespace': old_metadata.get('cookie_namespace', signature[:12]),
                 'session_namespace': old_metadata.get('session_namespace', signature[:16])}
    new_command = [sys.executable, "-m", "evaluation.pairwise.server", "--study", str(new_study_path),
                   "--db", str(db_path), "--host", args.host, "--port", str(args.port), "--open-assignment"]
    if random_assignment:
        new_command.append('--random-assignment')
    if '--secure-cookie' in command:
        new_command.append('--secure-cookie')
    process = None
    try:
        set_metadata(migration)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("ab") as log:
            process = subprocess.Popen(new_command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        wait_ready(process, args.host, args.port, new_signature)
    except Exception:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        set_metadata({key: old_metadata.get(key) for key in migration})
        with args.log.open("ab") as log:
            restored = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        wait_ready(restored, args.host, args.port, signature, require_open=False)
        raise RuntimeError("New demo failed; original command restored using the same database") from None
    after = preserved_state(db_path)
    before['metadata']['signature'] = new_signature
    if after["metadata"] != before["metadata"] or any(
            after[table].get(key) != value
            for table in ("raters", "assignments", "votes", "issues") for key, value in before[table].items()):
        raise RuntimeError("Preservation check failed; both consistent backups are retained")
    with connect_readonly(db_path) as db:
        for aid, (draft, revision, state) in before['drafts'].items():
            row = db.execute('SELECT draft,revision,state FROM assignments WHERE id=?', (aid,)).fetchone()
            if row is None or row[1] < revision or (row[1] == revision and row[2] == state and row[0] != draft):
                raise RuntimeError('Draft preservation check failed; consistent backups are retained')
    receipt = {"server_pid": process.pid, "port": args.port, "open_assignment": True,
               "random_assignment": random_assignment, 'models': len(new_study.models),
               'cases': len(new_study.cases), 'total_tasks': len(new_tasks),
               'study': str(new_study_path),
               "backup": str(final_backup), "before": before_counts, "after": counts(db_path),
               "existing_records_preserved": True, "timestamp": stamp}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    args.receipt.chmod(0o644)  # Aggregate counts only; no votes, names, secrets or resume codes.
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, help="PID of the existing root demo")
    parser.add_argument('--stats-rater', help='Read-only aggregate progress for this reviewer; do not restart')
    parser.add_argument('--stats', action='store_true', help='Read-only annotation totals; do not restart')
    parser.add_argument('--stats-raters', action='store_true', help='Read-only submitted counts per reviewer')
    parser.add_argument('--latest-rater', help='Read-only latest submitted case and anonymous A/B prompts')
    parser.add_argument('--export-human', type=Path, help='Read-only pseudonymized vote snapshot to a new directory')
    parser.add_argument("--study", type=Path, default=Path("evaluation/pairwise/runs/demo/study-with-prompts.json"))
    parser.add_argument('--new-study', type=Path, help='Append-only expanded study; retain existing records and browser sessions')
    parser.add_argument('--random-assignment', action='store_true', help='Random sampling with replacement')
    parser.add_argument("--db", type=Path, default=Path("/var/lib/memorybench-human/votes.sqlite3"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8920)
    parser.add_argument("--log", type=Path, default=Path("/var/lib/memorybench-human/server.log"))
    parser.add_argument("--receipt", type=Path, default=Path("/var/lib/fmh/runtime/memorybench-human-status.json"))
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("Run in a root terminal; the live database and server are root-owned")
    if args.export_human:
        from .human_report import export_snapshot
        print(json.dumps(export_snapshot(args.db, args.export_human), ensure_ascii=False, indent=2))
    elif args.latest_rater:
        print(json.dumps(latest_rater_record(args.db, args.latest_rater), ensure_ascii=False, indent=2))
    elif args.stats_raters:
        print(json.dumps(rater_counts(args.db), ensure_ascii=False, indent=2))
    elif args.stats:
        print(json.dumps(counts(args.db), ensure_ascii=False, indent=2))
    elif args.stats_rater:
        print(json.dumps(rater_stats(args.db, args.stats_rater), ensure_ascii=False, indent=2))
    else:
        if args.pid is None:
            parser.error('--pid is required for restart')
        restart(args)


if __name__ == "__main__":
    main()
