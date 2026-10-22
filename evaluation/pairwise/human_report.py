"""Export private human votes safely and compare VLM-only with pooled Elo."""

import argparse
from datetime import datetime
import hashlib
import hmac
import itertools
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from . import ANNOTATION_PROTOCOL, VERSION
from .manifest import digest, split
from .report import comparisons, export, leaderboard, pair_statistics, write_csv
from .store import normalize_vote


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def export_snapshot(database, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new snapshot directory; never overwrite an existing human snapshot')
    with sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        metadata = dict(db.execute('SELECT key,value FROM metadata'))
        if metadata['annotation_protocol'] != ANNOTATION_PROTOCOL:
            raise ValueError('Human annotation protocol does not match A/B/Tie')
        rows = db.execute("SELECT id,rater,task,swapped,vote FROM assignments WHERE state='voted' ORDER BY id").fetchall()
        registered = db.execute('SELECT COUNT(*) FROM raters').fetchone()[0]
        drafts = db.execute("SELECT COUNT(*) FROM assignments WHERE state='assigned'").fetchone()[0]
        issues = db.execute("SELECT COUNT(*) FROM assignments WHERE state='issue'").fetchone()[0]
        captured = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    study = json.loads(metadata['study'])
    tasks = {digest([case['case_id'], a, b])[:24]: (case['case_id'], a, b)
             for case in study['cases'] for a, b in itertools.combinations(sorted(case['predictions']), 2)}
    def anonymous(value):
        return 'human-' + hmac.new(metadata['secret'].encode(), value.encode(), hashlib.sha256).hexdigest()[:16]
    records = []
    for row in rows:
        cid, left, right = tasks[row['task']]
        a, b = (right, left) if row['swapped'] else (left, right)
        rater = anonymous(row['rater'])
        records.append({'id': 'human-vote-' + hashlib.sha256(row['id'].encode()).hexdigest()[:24],
                        'rater': rater, 'rater_name': rater, 'task': row['task'], 'state': 'voted',
                        'case_id': cid, 'split': split(cid), 'model_a': a, 'model_b': b,
                        'vote': normalize_vote(json.loads(row['vote'])), 'source_kind': 'human'})
    if len({record['id'] for record in records}) != len(records):
        raise ValueError('Duplicate exported human record IDs')
    snapshot = {'captured_at': captured, 'annotation_protocol': ANNOTATION_PROTOCOL,
                'study_signature': metadata['signature'], 'models': study['models'],
                'study_cases': len(study['cases']), 'study_tasks': len(tasks),
                'votes': len(records), 'active_raters': len({r['rater'] for r in records}),
                'registered_raters': registered, 'drafts_excluded': drafts, 'issues_excluded': issues,
                'unique_tasks': len({r['task'] for r in records}),
                'unique_cases': len({r['case_id'] for r in records}),
                'repeated_rater_tasks_retained': len(records) - len({(r['rater'], r['task']) for r in records}),
                'input_protocol': 'Human web: full History and GT/A/B videos, recorded text prompts, applicable per-model simulation GTs; one direct A/B/Tie choice',
                'privacy': 'Stable HMAC rater pseudonyms; no names, resume codes, session secrets, raw rater IDs, exact vote timestamps or SQLite databases exported',
                'snapshot_semantics': 'Contains all completed human votes at capture time; do not add another cumulative human snapshot'}
    output.mkdir(parents=True)
    write_json(output / 'votes.json', records)
    snapshot['votes_sha256'] = hashlib.sha256((output / 'votes.json').read_bytes()).hexdigest()
    write_json(output / 'snapshot.json', snapshot)
    return snapshot


def score(records, models, title, expected_tasks, mixed):
    groups = {}
    for group in ('Overall', 'IID', 'OOD'):
        selected = [r for r in records if group == 'Overall' or r['split'] == group]
        groups[group] = leaderboard(comparisons(selected), models, replicates=0, seed=42) | {
            'completed_judgments': len(selected), 'media_issues': 0,
            'pairs': pair_statistics(selected, models)}
    return {'version': VERSION, 'title': title,
            'protocol': {'id': ANNOTATION_PROTOCOL,
                         'evaluator': {'kind': 'mixed_human_and_model' if mixed else 'pooled_model'},
                         'weighting': 'Each actual judgment has weight 1; ties half a win; human repeats retained; no majority voting',
                         'fit': 'Bradley-Terry MLE; one virtual draw per model spread evenly over opponents; Elo mean 1500 per connected component',
                         'bootstrap_replicates': 0, 'confidence': 'Point estimates only; no confidence intervals computed',
                         'input_difference': 'VLM: eight-frame contact sheets per video, no History; human: full videos with History. Pooled results mix these input protocols.'},
            'coverage': {'complete': False, 'expected_tasks': expected_tasks,
                         'observed_unique_tasks': len({r['task'] for r in records}),
                         'completed_judgments': len(records), 'scope': 'Current snapshot of an ongoing random-with-replacement study'},
            'groups': groups}


def compare(vlm_run, human_snapshot, output):
    vlm_run, human_snapshot, output = map(Path, (vlm_run, human_snapshot, output))
    if (output / 'comparison.csv').exists():
        raise ValueError('Comparison already exists; use a new output directory')
    protocol = json.loads((vlm_run / 'protocol.json').read_text())
    snapshot = json.loads((human_snapshot / 'snapshot.json').read_text())
    if protocol['id'] != ANNOTATION_PROTOCOL or snapshot['annotation_protocol'] != ANNOTATION_PROTOCOL:
        raise ValueError('Different annotation protocols cannot be pooled')
    vlm = json.loads((vlm_run / 'report/votes.json').read_text())
    human = json.loads((human_snapshot / 'votes.json').read_text())
    if hashlib.sha256((human_snapshot / 'votes.json').read_bytes()).hexdigest() != snapshot['votes_sha256']:
        raise ValueError('Human snapshot checksum differs')
    for record in vlm + human:
        normalize_vote(record['vote'])
        if record['state'] != 'voted' or record['model_a'] == record['model_b']:
            raise ValueError('Expected actual A/B/Tie comparisons only')
        if any(record[k].lower().endswith('_tail') for k in ('model_a', 'model_b')):
            raise ValueError('Tail model found in comparison')
    if len({r['id'] for r in vlm + human}) != len(vlm) + len(human):
        raise ValueError('Duplicate imported record IDs; do not add cumulative historical runs')
    if len({(r['rater'], r['task']) for r in vlm}) != len(vlm):
        raise ValueError('VLM historical votes were imported twice')
    models = sorted({r[k] for r in vlm + human for k in ('model_a', 'model_b')})
    baseline = score(vlm, models, 'MemoryBench · VLM only', snapshot['study_tasks'], False)
    pooled = score(vlm + human, models, 'MemoryBench · VLM + human', snapshot['study_tasks'], True)
    published = json.loads((vlm_run / 'report/summary.json').read_text())['groups']['Overall']['leaderboard']
    before = {r['model']: r for r in baseline['groups']['Overall']['leaderboard']}
    for row in published:
        if abs(before[row['model']]['elo'] - row['elo']) > 1e-5:
            raise ValueError('VLM recomputation differs from the published leaderboard')
    for name, records, summary in (('vlm_only', vlm, baseline), ('vlm_plus_human', vlm + human, pooled)):
        export(output / name, records, summary)
        page = output / name / 'report.html'
        page.write_text(page.read_text().replace('按评估者 bootstrap', '本次仅计算 Elo 点估计，不计算置信区间'))
    comparison = []
    for row in pooled['groups']['Overall']['leaderboard']:
        old = before[row['model']]
        comparison.append({'model': row['model'], 'vlm_rank': old['rank'], 'vlm_elo': old['elo'],
                           'pooled_rank': row['rank'], 'pooled_elo': row['elo'],
                           'elo_delta': row['elo'] - old['elo'],
                           'vlm_games': old['games'], 'pooled_games': row['games'],
                           'human_games': row['games'] - old['games']})
    write_csv(output / 'comparison.csv', comparison)
    receipt = {'captured_at': snapshot['captured_at'], 'vlm_judgments': len(vlm),
               'human_judgments': len(human), 'pooled_judgments': len(vlm) + len(human),
               'vlm_raters': len({r['rater'] for r in vlm}), 'human_raters': snapshot['active_raters'],
               'vlm_source': vlm_run.name, 'vlm_votes_sha256': hashlib.sha256((vlm_run / 'report/votes.json').read_bytes()).hexdigest(),
               'human_votes_sha256': snapshot['votes_sha256'], 'published_vlm_elo_reproduced': True,
               'comparison': comparison, 'protocol': pooled['protocol']}
    write_json(output / 'comparison.json', receipt)
    lines = ['# Human snapshot and pooled Elo', '',
             f"Captured {snapshot['captured_at']}: {len(vlm)} VLM judgments + {len(human)} human judgments = {len(vlm) + len(human)}.", '',
             'Human votes are pseudonymized and exclude unsubmitted drafts and playback reports. No session credentials or real reviewer names are included.',
             'This human snapshot is cumulative; never add it to another cumulative human snapshot. Use only the latest merged VLM run, not its overlapping historical source runs.', '',
             'Each judgment counts equally, including human repeats. Ties count as half a win. Bradley–Terry, one virtual draw per model, connected-component mean Elo 1500. Point estimates only.', '',
             'VLMs see contact sheets without History; humans see full videos with History and applicable SIM references. The pooled score therefore mixes input protocols and is an interim result.', '',
             'See `comparison.csv`, `vlm_only/report.html`, `vlm_plus_human/report.html`, and `snapshot/snapshot.json`.']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vlm-run', type=Path, required=True)
    parser.add_argument('--human-snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.vlm_run, args.human_snapshot, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
