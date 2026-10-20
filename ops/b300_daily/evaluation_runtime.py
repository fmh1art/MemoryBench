"""Persistent, identity-checked judge caches and closed video staging.

This module changes startup/storage orchestration only, never scoring semantics.
"""
import json
import os
from pathlib import Path
import shutil

from queueing import read, write


def runtime_policy(recipe):
    policy = recipe['runtime']
    if (policy['version'] != 'persistent_judge_v1'
            or policy['engine_ready_timeout_seconds'] != 2400
            or policy['service_ready_timeout_seconds'] != 2700
            or policy['progress_interval_seconds'] != 30
            or not Path(policy['cache_root']).is_absolute()):
        raise ValueError('Unsupported judge startup policy')
    return policy


def cache_identity(recipe):
    # Native vLLM adds its hardware/config/parallelism fingerprint underneath
    # this namespace. An operator must review any outer model/environment change.
    return dict(format='memorybench_judge_cache_v1',recipe=recipe['recipe_id'],
        basis_receipt_sha256=recipe['basis_receipt_sha256'],
        judge_packages=recipe['judge_packages'],judge_model=recipe['judge']['model'],
        model_dir=recipe['judge']['model_dir'],dtype='bfloat16',tensor_parallel_size=1)


def persistent_cache(recipe):
    root = Path(runtime_policy(recipe)['cache_root'])
    identity = cache_identity(recipe)
    if root.exists():
        if not (root/'CACHE_IDENTITY.json').is_file() or read(root/'CACHE_IDENTITY.json') != identity:
            raise ValueError('Unreviewed or mismatched persistent judge cache')
    else:
        root.mkdir(parents=True,exist_ok=False)
        write(root/'CACHE_IDENTITY.json',identity)
    return root


def judge_environment(recipe, runtime, selected, inherited):
    policy = runtime_policy(recipe)
    env = dict(inherited)
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,selected)),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_ENGINE_READY_TIMEOUT_S=str(policy['engine_ready_timeout_seconds']),
        FLASHINFER_WORKSPACE_BASE=str(runtime),
        FLASHINFER_CUBIN_DIR='/mnt/bos/tm9b1x/fmh/model/.dependencies/flashinfer-cubins',
        VLLM_CACHE_ROOT=str(runtime/'cache'),TRITON_CACHE_DIR=str(runtime/'triton-cache'),
        TORCH_EXTENSIONS_DIR=str(runtime/'extensions'),HF_HOME=str(runtime/'hf-cache'),
        TORCHINDUCTOR_CACHE_DIR=str(runtime/'torch-cache'))
    env['PATH'] = str(Path(recipe['judge']['executable']).parent)+':/usr/local/cuda-13.0/bin:'+env['PATH']
    return env


def media_record(canonical, sizes):
    return dict(format='memorybench_closed_media_v1',case=canonical,sizes=sizes)


def staged_media(canonical, work):
    """Reuse only explicitly closed files; never silently accept a partial copy."""
    root = Path(work)/'media'/canonical['case_id']
    paths = {key:root/key/'future.mp4' for key in ('gt_video','gen_video')}
    receipt = root/'MEDIA_COMPLETE.json'
    if receipt.exists():
        record = read(receipt)
        sizes = {key:path.stat().st_size for key,path in paths.items()}
        if record != media_record(canonical,sizes) or any(size<=0 for size in sizes.values()):
            raise ValueError('Cached media identity/size changed: '+canonical['case_id'])
        return paths
    if any(path.exists() for path in paths.values()):
        raise ValueError('Unclosed cached media requires explicit recovery: '+canonical['case_id'])
    sizes = {}
    for key,destination in paths.items():
        source = Path(canonical[key]); destination.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as inp, destination.open('xb') as out:
            shutil.copyfileobj(inp,out,8*1024**2)
        sizes[key] = source.stat().st_size
        if sizes[key] <= 0 or sizes[key] != destination.stat().st_size:
            raise ValueError('Video cache copy incomplete: '+str(source))
    write(receipt,media_record(canonical,sizes))
    return paths
