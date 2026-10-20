"""Only the approved Dense80k -> KDA23/VSA7 72-rank staged run is eligible."""
import hashlib
import json
from pathlib import Path
import re

ALGORITHM = 'kda23_vsa7_p4d'
RUN = 'kda_vsa_dense80k_lw20_110k_20261016'
FORMAT = 'husicheng_kda_vsa72_inference_v1'
STAGES = {'alignment': (0, 5000), 'direction': (5000, 5000), 'flowmatch': (10000, 100000)}
NAME = re.compile(r'kda_vsa_d80k_lw20_(alignment|direction|flowmatch)_step([1-9][0-9]*)_total([1-9][0-9]*)_cursor([1-9][0-9]*)_(\d{8})')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counters(stage, step):
    if stage not in STAGES or type(step) is not int or not 0 < step <= STAGES[stage][1]:
        raise ValueError('Unsupported checkpoint stage/update')
    total = STAGES[stage][0] + step
    return total, 80000 + total


def identity(name):
    match = NAME.fullmatch(name)
    if match is None:
        raise ValueError('Unsupported KDA/VSA checkpoint identity')
    total, cursor = counters(match[1], int(match[2]))
    if (total, cursor) != (int(match[3]), int(match[4])):
        raise ValueError('Stage update, pipeline update and source cursor disagree')
    return dict(stage=match[1], stage_step=int(match[2]), pipeline_step=total, next_data_cursor=cursor)


def validate_complete(complete, pipeline, architecture, stage, step):
    contract = complete['contract']
    total, cursor = counters(stage, step)
    if (complete['format'] != 'evophys_sm100_efla_flowmatch_full_state_v1'
            or complete['step'] != step or contract['format'] != 'dense80k_kda_vsa72_stage_v1'
            or contract['kind'] != 'kda_vsa' or contract['stage'] != stage
            or contract['world_size'] != 72 or contract['offset'] != STAGES[stage][0]
            or contract['max_steps'] != STAGES[stage][1] or contract['pipeline'] != pipeline
            or contract['architecture'] != architecture):
        raise ValueError('Closed checkpoint differs from the approved KDA/VSA pipeline')
    return dict(algorithm=ALGORITHM, source_dense_step=80000, stage=stage,
                stage_step=step, pipeline_step=total, next_data_cursor=cursor)


def select_latest(root, pipeline, architecture):
    """Read only closure/archive receipts; compare total updates across stages."""
    candidates = []
    for stage in STAGES:
        paths = sorted((p for p in (Path(root)/stage).glob('state_step*') if p.name[10:].isdigit()),
                       key=lambda p: int(p.name[10:]), reverse=True)
        for path in paths:
            if path.is_symlink():
                raise ValueError('A checkpoint cannot be a symlink')
            if not (path/'COMPLETE.json').is_file() or not (path/'ARCHIVE.json').is_file():
                continue
            complete, archive = read(path/'COMPLETE.json'), read(path/'ARCHIVE.json')
            model = validate_complete(complete, pipeline, architecture, stage, int(path.name[10:]))
            if (archive['format'] != 'kda_vsa_bos_stream_archive_v1' or archive['stage'] != stage
                    or archive['step'] != model['stage_step'] or archive['destination'] != str(path)
                    or archive['closure_sha256'] != digest(path/'COMPLETE.json')):
                raise ValueError('KDA/VSA durable archive closure differs')
            if model['pipeline_step'] > 2:
                candidates.append((model['pipeline_step'], path, model))
            break
    if not candidates:
        raise RuntimeError('No complete archived trained KDA/VSA state; no Direct64 fallback')
    return max(candidates, key=lambda item: item[0])


def validate_sidecar(saved, architecture):
    if (saved['format'] != FORMAT or saved['run'] != RUN or saved['architecture'] != architecture
            or saved['source_dense_step'] != 80000 or saved['algorithm'] != ALGORITHM):
        raise ValueError('Unsupported KDA/VSA exported architecture or lineage')
    total, cursor = counters(saved['stage'], saved['stage_step'])
    if saved['pipeline_step'] != total or saved['next_data_cursor'] != cursor:
        raise ValueError('Exported counters disagree')
    return saved
