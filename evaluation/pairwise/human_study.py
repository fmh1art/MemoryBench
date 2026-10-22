"""Prepare every available non-tail prediction for human comparisons, without sampling."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import subprocess

from .manifest import load_study
from .prepare import probe

SIM_MODELS = {'GE-SIM', 'DreamDojo'}
SCOPE = ('action_description', 'memory_target', 'success_criterion',
         'read_start_seconds', 'read_end_seconds')


def saved_prompt(directory, model, cid):
    for name in ('adapter.json', 'prediction.json', 'metadata.json', 'result.json', 'generation.json'):
        path = directory / name
        if not path.is_file():
            continue
        value = json.loads(path.read_text())
        identity = value.get('case_id', value.get('case_name', value.get('evaluation_case_id')))
        if identity != cid or value.get('model', model) != model:
            continue
        if 'prompt' in value and (value['prompt'] is None or isinstance(value['prompt'], str)):
            return {'prompt': value['prompt'], 'prompt_recorded': True}
        text = value.get('future_action')
        if model.startswith('Seedance') and isinstance(text, str) and value.get('future_action_sha256') in {
                hashlib.sha256((text + suffix).encode()).hexdigest() for suffix in ('', '\n')}:
            return {'prompt': text, 'prompt_recorded': True}
    return {}


def prepare(args):
    root, data = args.predictions.resolve(), args.data.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Use a new output path to preserve existing studies')
    index = json.loads((root / '.metadata/manifest.json').read_text())['predictions']
    jobs = {(j['model'], j['case']['case_id']): j for j in json.loads(args.jobs.read_text())}
    models = sorted({p.name for group in ('VisualSignal', 'TextControl', 'DirectAction')
                     for p in (root / group).iterdir() if p.is_dir() and not p.name.lower().endswith('_tail')})
    entries = [p for p in index if p['method'] in models]
    directories = {p['case_name']: p['benchmark_case_directory'] for p in index if p.get('benchmark_case_directory')}
    entries = [dict(p, benchmark_case_directory=p.get('benchmark_case_directory') or directories[p['case_name']])
               for p in entries]
    if len({(p['method'], p['case_name']) for p in entries}) != len(entries):
        raise ValueError('Duplicate model/case in prediction index')
    old, _, _ = load_study(args.source)
    old_cases = {case.case_id: case.model_dump() for case in old.cases}
    candidates, paths = {}, set()
    missing = []
    @lru_cache(maxsize=None)
    def is_file(path):
        return path.is_file()
    def locate(entry):
        cid, model = entry['case_name'], entry['method']
        base = data / entry['benchmark_case_directory']
        prediction = root / entry['prediction']
        history, gt = base / 'input/rgb_prefix.mp4', base / 'ground_truth/future.mp4'
        if not all(is_file(p) for p in (prediction, history, gt)):
            return {'model': model, 'case_id': cid, 'reason': 'missing prediction/history/GT'}
        reference = None
        job = jobs.get((model, cid))
        if model in SIM_MODELS:
            if job is None:
                raise ValueError(f'Missing simulation source record: {model}/{cid}')
            fallback = root / entry['other_files'] / 'gt.mp4'
            if args.sim_data:
                marker = '/MemoryDataV1-Sim/'
                reference = args.sim_data / job['case']['gt_video'].split(marker, 1)[1]
            if reference is None or not is_file(reference):
                reference = fallback
            if not is_file(reference):
                return {'model': model, 'case_id': cid, 'reason': 'missing simulation GT'}
        return cid, model, entry, job, prediction, reference, history, gt
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, result in enumerate(pool.map(locate, entries), 1):
            if isinstance(result, dict):
                missing.append(result)
            else:
                cid, model, entry, job, prediction, reference, history, gt = result
                candidates.setdefault(cid, {})[model] = (entry, job, prediction, reference)
                paths.update((history, gt, prediction))
                if reference:
                    paths.add(reference)
            if n % 500 == 0:
                print(json.dumps({'located': n, 'total': len(entries)}), flush=True)
    # Full ffprobe validation is task-specific; never read unrelated dataset assets.
    ordered = sorted(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = output.with_suffix('.probe-cache.json')
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    def checked(path):
        try:
            stat = path.stat()
            fingerprint = [stat.st_size, stat.st_mtime_ns]
            cached = cache.get(str(path))
            if cached and cached['fingerprint'] == fingerprint:
                return cached['clip']
            clip = probe(path)
            cache[str(path)] = {'fingerprint': fingerprint, 'clip': clip}
            return clip
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            return {'error': type(error).__name__}
    checked_paths = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, (path, result) in enumerate(zip(ordered, pool.map(checked, ordered)), 1):
            checked_paths[path] = result
            if n % 500 == 0:
                cache_path.write_text(json.dumps(cache))
                print(json.dumps({'validated': n, 'total': len(ordered)}), flush=True)
    cache_path.write_text(json.dumps(cache))
    cases, references, reference_sources = [], {}, {}
    def build_case(item):
        cid, predictions = item
        entry = next(iter(predictions.values()))[0]
        base = data / entry['benchmark_case_directory']
        history, gt = base / 'input/rgb_prefix.mp4', base / 'ground_truth/future.mp4'
        if any('error' in checked_paths[p] for p in (history, gt)):
            missing.append({'case_id': cid, 'reason': 'invalid history/GT'})
            return None
        # Keep the old nine comparisons byte-for-byte equivalent, including their text and scope.
        case = old_cases.get(cid, {'case_id': cid, 'history': checked_paths[history],
                                   'gt': checked_paths[gt], 'predictions': {}})
        if cid not in old_cases:
            real_job = next((j for m, (_, j, _, _) in predictions.items() if m not in SIM_MODELS and j), None)
            if real_job:
                case.update({k: real_job['case'][k] for k in SCOPE if k in real_job['case']})
        for model, (entry, job, prediction, reference) in predictions.items():
            if any('error' in checked_paths[p] for p in (prediction, reference) if p):
                missing.append({'model': model, 'case_id': cid, 'reason': 'invalid video'})
                continue
            if model not in case['predictions']:
                clip = dict(checked_paths[prediction])
                clip['start_seconds'] = job['case'].get('gen_start_seconds', 0) if job else 0
                clip.update(saved_prompt(root / entry['other_files'], model, cid))
                case['predictions'][model] = clip
            if reference:
                clip = dict(checked_paths[reference])
                clip['start_seconds'] = job['case'].get('gt_start_seconds', 0)
                # These are distinct simulation worlds. Never substitute the real GT.
                with reference.open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if reference == root / entry['other_files'] / 'gt.mp4':
                    # Canonical published references supersede the older evaluation-jobs paths.
                    provenance = json.loads((root / entry['evaluation_results'] / 'provenance.json').read_text())
                    expected = str(reference.relative_to(root))
                    if (provenance.get('model') != model or provenance.get('case_id') != cid
                            or provenance.get('ground_truth_type') != 'simulation'
                            or not provenance.get('matches_current_prediction')
                            or provenance.get('current_prediction_sha256') != entry['prediction_sha256']
                            or not provenance.get('ground_truth', '').endswith('/MemoryBenchPredictionsV1/' + expected)):
                        raise ValueError(f'Unverified canonical simulation reference: {model}/{cid}')
                    source = 'current per-prediction provenance'
                else:
                    if actual != job['gt_sha256']:
                        raise ValueError(f'Simulation GT checksum mismatch: {model}/{cid}')
                    source = 'evaluation jobs SHA256'
                reference_sources[f'{cid}/{model}'] = {'path': str(reference), 'sha256': actual, 'source': source}
                references.setdefault(cid, {})[model] = {
                    'clip': clip, **{k: job['case'].get(k, '') for k in SCOPE[:3]}}
        if len(case['predictions']) >= 2:
            return case
        return None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, case in enumerate(pool.map(build_case, sorted(candidates.items())), 1):
            if case:
                cases.append(case)
            if n % 100 == 0:
                print(json.dumps({'prepared_cases': n, 'total': len(candidates)}), flush=True)
    active_models = {m for case in cases for m in case['predictions']}
    summary = {'requested_models': models, 'available_models': sorted(active_models),
               'cases': len(cases), 'tasks': sum(len(c['predictions']) * (len(c['predictions']) - 1) // 2 for c in cases),
               'predictions_per_model': dict(Counter(m for c in cases for m in c['predictions'])),
               'missing_count': len(missing), 'missing': missing}
    ready = active_models == set(models) and not any(p['reason'] == 'missing simulation GT' for p in missing)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    study = {'title': old.title, 'seed': old.seed, 'raters_per_pair': old.raters_per_pair,
             'models': sorted(active_models), 'cases': cases,
             'provenance': {'allow_partial_models': True, 'model_references': references,
                            'reference_sources': reference_sources,
                            'deployment_ready': ready, 'requested_models': models,
                            'reference_models': sorted(SIM_MODELS & active_models),
                            'sampling': 'all non-tail available predictions; every within-case unordered pair',
                            'assignment': 'uniform random with replacement',
                            'source_study': str(args.source.resolve())}}
    output.write_text(json.dumps(study, ensure_ascii=False, indent=2) + '\n')
    load_study(output)
    print(json.dumps({k: v for k, v in summary.items() if k != 'missing'}, ensure_ascii=False), flush=True)
    if not ready:
        raise ValueError('Candidate prepared, but missing model references must be supplied before deployment; see audit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sim-data', type=Path)
    parser.add_argument('--workers', type=int, default=12)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
