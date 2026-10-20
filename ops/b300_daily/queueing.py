"""KDA/VSA publication handoff; old Direct64 states cannot enter this queue."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo
from kda_contract import ALGORITHM, identity, read, validate_sidecar

FORMAT = 'husicheng_daily_memorybench_kdavsa_lw20_v1'
RUN_NAME = 'memorybench-camera-v2-kdavsa-lw20-daily-v1'
OUTPUT_POLICY = 'prediction_videos_only_v1'
EVAL_RECIPE = 'memorybench38k-v34-20261012'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def checkpoint_identity(target):
    target = Path(target)
    counters = identity(target.name)
    markers = list((target/'.uploads').glob('*.json'))
    if len(markers) != 1:
        raise ValueError('Exactly one closed model publication record required')
    raw = markers[0].read_bytes(); record = json.loads(raw)
    if (record['publication_status'] != 'BOS_PUBLISHED' or record['checkpoint_name'] != target.name
            or record['artifact_type'] != 'checkpoint'):
        raise ValueError('Unpublished or mismatched KDA/VSA model')
    for name in ('weights/dit.safetensors','weights/dit.kdavsa.json','configs/.config','others/SOURCE.json'):
        if not (target/name).is_file() or (target/name).stat().st_size != record['files'][name]['size']:
            raise ValueError('Published artifact is missing or has changed size: '+name)
    config, source = read(target/'configs/.config'), read(target/'others/SOURCE.json')
    saved = validate_sidecar(read(target/'weights/dit.kdavsa.json'),
                             read(Path(__file__).with_name('architecture.json')))
    if config['model']['algorithm'] != ALGORITHM or any(saved[k] != v for k,v in counters.items()):
        raise ValueError('KDA/VSA model identity differs')
    if any(source[k] != v for k,v in counters.items()):
        raise ValueError('Exported source lineage differs')
    return dict(checkpoint_name=target.name, publication_marker=markers[0].name,
        publication_sha256=hashlib.sha256(raw).hexdigest(),source_commit=source['source_commit'],
        runtime_archive_sha256=record['files']['others/runtime.tar.gz']['sha256'],
        weight_sha256=record['files']['weights/dit.safetensors']['sha256'],
        weight_bytes=record['files']['weights/dit.safetensors']['size'])


def enqueue(target, root):
    fields = checkpoint_identity(target)
    path = Path(root)/(fields['checkpoint_name']+'.json')
    body = dict(format=FORMAT,run_name=RUN_NAME,output_policy=OUTPUT_POLICY,post_eval=EVAL_RECIPE,**fields)
    if path.exists():
        previous = read(path)
        if any(previous.get(k) != v for k,v in body.items()):
            raise ValueError('Conflicting queued checkpoint')
        return path
    write(path,dict(body,enqueued_at=now()))
    return path


def validate_request(request, config):
    if (request['format'] != FORMAT or request['run_name'] != RUN_NAME
            or request.get('post_eval') != EVAL_RECIPE or request['output_policy'] != OUTPUT_POLICY):
        raise ValueError('Unsupported KDA/VSA daily request')
    identity(request['checkpoint_name'])
    target = Path(config['checkpoint_root'])/request['checkpoint_name']
    actual = checkpoint_identity(target)
    if any(request.get(k) != v for k,v in actual.items()):
        raise ValueError('Model publication identity changed')
    if (request['source_commit'] != config['source_commit']
            or request['runtime_archive_sha256'] != config['runtime_archive_sha256']):
        raise ValueError('Unreviewed KDA/VSA source runtime')
    return target
