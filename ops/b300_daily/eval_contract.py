"""Pure protocol/admission helpers for the pinned 38k evaluation contract."""
from pathlib import Path
import hashlib
import json
import math
from gpu_policy import ELIGIBLE_GPUS, POLICY_ID, validate_active

RECIPE_ID = 'memorybench38k-v34-20261012'
GPUS = ELIGIBLE_GPUS
SCORING_VERSION = 'memory.v3.4.missing_evidence_zero'
COMMIT = '45ab68fea8af30b874e4984054116657d29a8b7b'
SCHEMA_REMINDER = ('Schema reminder: observability evidence.frame_indices may cite ONLY the integers in '
    'sampling.read_frame_indices. action_following evidence.frame_indices may cite ONLY integers 1 through 12. '
    'These are contact-sheet tile indices, not original video frame numbers. Preserve the existing Gate criteria.\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_recipe(recipe):
    if (recipe['recipe_id'] != RECIPE_ID or recipe['gpus'] != GPUS or recipe['gpu_policy'] != POLICY_ID
            or recipe['evaluator_commit'] != COMMIT or recipe['scoring_version'] != SCORING_VERSION
            or recipe['expected_cases'] != 649 or recipe['judge']['tp'] != 1 or recipe['judge']['dp'] != 'auto_idle'
            or recipe['judge']['host'] != '127.0.0.1' or recipe['judge']['port'] != 18047
            or recipe['api_concurrency'] != 512 or recipe['case_workers'] != 64 or recipe['automated_workers'] != 32
            or recipe['judge']['model'] != 'qwen3.8-27b' or recipe['judge']['temperature'] != 0
            or recipe['judge']['thinking'] is not False or recipe['judge']['max_tokens'] != 8192
            or recipe['judge']['timeout'] != 600 or recipe['judge']['retries'] != 5):
        raise ValueError('Unsupported evaluation recipe or GPU policy')


def mapped_cases(reference, target, dataset):
    """Retain every 38k case field except its generated-video checkpoint prefix."""
    prefix = Path(target).parent/reference['checkpoint']/'predictions'/reference['prediction_run']
    result = []
    for original in reference['cases']:
        rel = Path(original['gen_video']).relative_to(prefix)
        if rel.name != 'future.mp4' or Path(original['gt_video']) != Path(dataset)/rel.parent/'ground_truth/future.mp4':
            raise ValueError('Reference GT/generated future pairing differs')
        row = dict(original)
        from queueing import RUN_NAME
        row['gen_video'] = str(Path(target)/'predictions'/RUN_NAME/rel)
        result.append(row)
    ids = [case['case_id'] for case in result]
    if len(ids) != 649 or len(set(ids)) != 649:
        raise ValueError('Expected exactly 649 unique reference cases')
    return result


def require_inference_complete(state, batch, prediction_record, request):
    for item in (state, batch):
        if (item.get('status') != 'COMPLETE' or item.get('expected_jobs') != 649
                or item.get('completed_jobs') != 649 or item.get('failures')
                or item.get('checkpoint') != request['checkpoint_name']):
            raise ValueError('Full successful inference is required before evaluation')
    if (prediction_record['run_status'] != 'COMPLETE'
            or prediction_record['model_publication_sha256'] != request['publication_sha256']
            or len(prediction_record['cases']) != 649 or len(set(prediction_record['cases'])) != 649):
        raise ValueError('Inference publication closure differs')


def require_scores_complete(summary, automated):
    if (summary.get('scoring_version') != SCORING_VERSION or summary.get('benchmark_complete') is not True
            or summary.get('total_cases') != 649 or summary.get('memory_score_n') != 649
            or summary.get('error_cases') != 0 or automated.get('total') != 649
            or automated.get('complete') != 649 or automated.get('errors')):
        raise ValueError('Incomplete evaluation; never publish a completed score')
    if not math.isfinite(summary['memory_score_100']) or not 0 <= summary['memory_score_100'] <= 100:
        raise ValueError('Invalid evaluation score')


def judge_command(recipe, gpus):
    check_recipe(recipe)
    validate_active(gpus)
    j = recipe['judge']
    return [j['executable'], 'serve', j['model_dir'], '--served-model-name', 'qwen3.8-27b',
        '--host', '127.0.0.1', '--port', str(j['port']), '--tensor-parallel-size', '1',
        '--data-parallel-size', str(len(gpus)), '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.80',
        '--max-model-len', '16384', '--max-num-seqs', '128', '--max-num-batched-tokens', '32768',
        '--enable-chunked-prefill', '--enable-prefix-caching', '--reasoning-parser', 'qwen3',
        '--default-chat-template-kwargs', '{"enable_thinking":false}', '--generation-config', 'auto',
        '--override-generation-config', '{"max_new_tokens":8192}',
        '--limit-mm-per-prompt', '{"image":8,"video":1}', '--media-io-kwargs', '{"video":{"num_frames":32}}',
        '--mm-encoder-tp-mode', 'data', '--mm-processor-cache-gb', '0',
        '--allowed-local-media-path', recipe['dataset']]
