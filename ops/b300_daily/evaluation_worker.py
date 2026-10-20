"""Pinned 38k evaluator, with an owned idle-GPU judge and separate score closure.

No inference or model semantics live here. The archived native evaluator resolves
PTS alignment, frame sampling, Gate, metrics and aggregation without modifications.
"""
import argparse
import asyncio
import concurrent.futures
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

from queueing import read, write, now, validate_request
from gpu_policy import idle_snapshot, validate_policy
from evaluation_runtime import runtime_policy, persistent_cache, judge_environment, staged_media
from evaluation_publish import publish_closed_tree
from checklist_backup import save as save_checklist_backup
from benchmark_revision import validate_checkout
from eval_contract import (COMMIT, GPUS, RECIPE_ID, SCHEMA_REMINDER, SCORING_VERSION,
    check_recipe, digest, judge_command, mapped_cases, require_inference_complete,
    require_scores_complete)


def preflight(recipe, config, target):
    """One metadata/source audit per evaluation; never rehash large model weights."""
    check_recipe(recipe)
    runtime_policy(recipe)
    validate_policy(config)
    if (config['gpus'] != GPUS or config['dataset'] != recipe['dataset']
            or config['benchmark_commit'] != recipe['benchmark_commit']
            or config['benchmark_repo'] != recipe['benchmark_repo']):
        raise ValueError('Inference/evaluation dataset or GPU contract differs')
    basis = Path(recipe['basis'])
    if (digest(basis/'RECEIPT.json') != recipe['basis_receipt_sha256']
            or digest(str(basis)+'.tar.gz') != recipe['basis_archive_sha256']):
        raise ValueError('Frozen reference receipt/archive changed')
    receipt = read(basis/'RECEIPT.json')
    for name, expected in receipt['source_files'].items():
        if digest(basis/name) != expected:
            raise ValueError('Frozen evaluator/input changed: '+name)
    reference = read(basis/'reference_run.json')
    if (reference['evaluator_commit'] != COMMIT
            or reference['dataset_commit'] != recipe['benchmark_commit']
            or reference['scoring_version'] != SCORING_VERSION):
        raise ValueError('Reference evaluation identity differs')
    data = Path(recipe['dataset'])
    checkout = validate_checkout(data, reference, recipe)
    versions = {name:importlib.metadata.version(name) for name in recipe['packages']}
    if versions != recipe['packages']:
        raise ValueError('Native evaluation environment changed: '+repr(versions))
    judge_python = str(Path(recipe['judge']['executable']).with_name('python'))
    probe = 'import importlib.metadata,json; print(json.dumps({n:importlib.metadata.version(n) for n in '+repr(list(recipe['judge_packages']))+'}))'
    judge_versions = json.loads(subprocess.check_output([judge_python,'-c',probe],text=True))
    if judge_versions != recipe['judge_packages']:
        raise ValueError('Judge environment changed: '+repr(judge_versions))
    model = Path(recipe['judge']['model_dir'])
    for name, item in receipt['model_metadata'].items():
        if str((model/name).resolve()) != item['resolved'] or digest(model/name) != item['sha256']:
            raise ValueError('Judge model metadata changed: '+name)
    for name, item in receipt['model_weights'].items():
        if str((model/name).resolve()) != item['resolved'] or (model/name).stat().st_size != item['size']:
            raise ValueError('Judge model shard changed: '+name)
    sys.path.insert(0,str(basis))
    from evaluation.scoring import SCORING_VERSION as native_version
    from evaluation.schema import Case
    if native_version != SCORING_VERSION:
        raise ValueError('Native scoring version differs')
    cases = mapped_cases(reference,target,data)
    # Already-resolved reference memory_objects are retained, not re-inferred.
    for item in cases:
        if Case.model_validate(item).model_dump() != item:
            raise ValueError('Canonical case requires an unreviewed schema conversion')
    return cases, receipt, dict(status='PREFLIGHT_PASS',recipe=RECIPE_ID,cases=649,
        evaluator_commit=COMMIT,packages=versions,judge_packages=judge_versions,benchmark_checkout=checkout,
        reference_summary_replay=receipt['golden_summary_replay'],gpus=GPUS,
        new_gpu_inference_or_scoring=False,checked_at=now())


def state_update(work, **values):
    path = work/'state.json'
    state = read(path)
    state.update(**values,updated_at=now())
    write(path,state)


def prepare_case(canonical, work, target):
    from evaluation.schema import Case
    marker = read(Path(canonical['gen_video']).with_name('completion.json'))
    if (marker['status'] != 'PASS' or marker.get('camera_contract_id') != 'dense_opencv_camera_v2_20261011'
            or Path(marker['checkpoint']).parent.parent != target):
        raise ValueError('Case inference closure differs: '+canonical['case_id'])
    cached = dict(canonical)
    for key,destination in staged_media(canonical,work).items():
        cached[key] = str(destination)
    return canonical,Case.model_validate(cached)


def stop_owned(process):
    """Only a subprocess group created by this worker, never a shared service."""
    if process is None:
        return
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        process.wait(); return
    deadline = time.monotonic()+45
    while time.monotonic() < deadline:
        process.poll()
        try: os.killpg(process.pid,0)
        except ProcessLookupError: return
        time.sleep(1)
    try: os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError: pass
    process.wait(timeout=15)


async def evaluate(pairs, recipe, output, provenance):
    from evaluation.run import evaluate_case
    from inventory_repair import make_judge as Judge
    from evaluation.scoring import summarize
    j = recipe['judge']
    endpoint = f"http://{j['host']}:{j['port']}/v1/chat/completions"
    judge = Judge(endpoint,j['model'],concurrency=512,timeout=600,retries=5,
        temperature=0,max_tokens=8192,extra_body={'chat_template_kwargs':{'enable_thinking':False}})
    limit = asyncio.Semaphore(64)
    records = {}
    start = time.monotonic()

    def progress():
        good = sum(r.get('memory_score') is not None and r['status'] != 'error' for r in records.values())
        write(output/'progress.json',dict(vlm_completed=good,total=649,processed=len(records),
            errors=len(records)-good,elapsed_seconds=time.monotonic()-start,updated_at=now()))
        print('VLM',good,'/649', 'errors',len(records)-good,flush=True)

    async def one(pair, attempt, selected_judge, note=None):
        canonical, case = pair
        async with limit:
            destination = output/case.case_id
            if destination.exists():
                history = output/'failed-attempts'/f'{case.case_id}-{time.time_ns()}'
                history.parent.mkdir(exist_ok=True)
                destination.rename(history)
            result = await evaluate_case(case,selected_judge,destination)
            result.update(case=canonical,provenance=provenance,attempt=attempt)
            if note: result['retry_note'] = note
            write(destination/'result.json',result)
            records[case.case_id] = result
            progress()

    try:
        for attempt in range(1,4):
            pending = [p for p in pairs if p[1].case_id not in records
                or records[p[1].case_id].get('memory_score') is None
                or records[p[1].case_id]['status'] == 'error']
            if not pending: break
            if attempt > 1: judge.limit = asyncio.Semaphore(256)
            await asyncio.gather(*(one(p,attempt,judge) for p in pending))
    finally:
        await judge.close()
    # Reproduce the documented 38k evidence-index repair, only for that error.
    # A valid Gate failure / missing-evidence zero is final, never score-selected.
    pending = [p for p in pairs if records[p[1].case_id]['status'] == 'error'
        and 'Evidence must cite permitted Read frame indices' in records[p[1].case_id].get('error','')]
    if pending:
        retry = Judge(endpoint,j['model'],concurrency=4,timeout=240,retries=3,
            temperature=0,max_tokens=8192,extra_body={'chat_template_kwargs':{'enable_thinking':False}})
        retry.rubrics['gate'] = SCHEMA_REMINDER+retry.rubrics['gate']
        try:
            for pair in pending:
                for attempt in range(1,4):
                    await one(pair,'evidence-index-'+str(attempt),retry,
                        'Repeated existing frame-index constraints; scoring unchanged')
                    result = records[pair[1].case_id]
                    if result['status'] != 'error' and result.get('memory_score') is not None: break
        finally: await retry.close()
    if set(records) != {p[1].case_id for p in pairs}:
        raise ValueError('Evaluation case coverage differs')
    summary = summarize([records[p[1].case_id] for p in pairs])
    write(output/'summary.json',summary)
    return summary


def start_judge(recipe, work):
    # Admission can change while videos are staged; wait again, never preempt.
    while True:
        selected,used,owners = idle_snapshot()
        if selected: break
        state_update(work,phase='WAITING_GPU',memory_used_MiB=used,occupied_pids=owners)
        time.sleep(30)
    with socket.socket() as probe:
        probe.bind((recipe['judge']['host'],recipe['judge']['port']))
    runtime = persistent_cache(recipe)
    env = judge_environment(recipe,runtime,selected,os.environ)
    with (work/'judge.log').open('ab',buffering=0) as log:
        process = subprocess.Popen(judge_command(recipe,selected),env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        state_update(work,phase='STARTING_JUDGE',judge_pid=process.pid,judge_pgid=process.pid,
                     gpus=selected,dp=len(selected),gpu_policy=recipe['gpu_policy'],
                     judge_command=judge_command(recipe,selected),judge_cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],
                     judge_started_at=now(),judge_cache=str(runtime),startup_policy=runtime_policy(recipe))
    except BaseException:
        stop_owned(process)
        raise
    return process,selected


async def wait_judge(process, recipe, work):
    import httpx
    j = recipe['judge']; policy = runtime_policy(recipe)
    start = time.monotonic(); deadline = start+policy['service_ready_timeout_seconds']; next_update = start
    async with httpx.AsyncClient(trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None: raise RuntimeError('Owned judge exited; see judge.log')
            if time.monotonic() >= next_update:
                state_update(work,startup_elapsed_seconds=round(time.monotonic()-start,1),
                    service_ready_timeout_seconds=policy['service_ready_timeout_seconds'])
                next_update = time.monotonic()+policy['progress_interval_seconds']
            try:
                response = await client.get(f"http://{j['host']}:{j['port']}/v1/models",timeout=10)
                if response.status_code == 200:
                    if [m['id'] for m in response.json()['data']] != [j['model']]:
                        raise ValueError('Unexpected model on owned judge endpoint')
                    state_update(work,judge_ready_at=now(),startup_elapsed_seconds=round(time.monotonic()-start,1))
                    return
            except httpx.HTTPError: pass  # Expected during owned server startup.
            await asyncio.sleep(5)
    raise TimeoutError('Owned judge exceeded the recorded service startup deadline')


async def score(pairs, recipe, work, output, provenance, judge):
    inputs = work/'automated-inputs'
    for canonical,case in pairs:
        write(inputs/case.case_id/'result.json',dict(case_id=case.case_id,
            case=case.model_dump(),status='prediction_complete'))
    env = dict(os.environ,PYTHONPATH=recipe['basis'],CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command = [recipe['python'],'-m','evaluation.automated.run','--results',str(inputs),
        '--output',str(output/'automated'),'--workers','32','--device','cpu','--resume']
    with (work/'automated.log').open('ab',buffering=0) as log:
        auto = subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,
                                stderr=subprocess.STDOUT,start_new_session=True)
        try:
            state_update(work,phase='AUTOMATED_SCORING_AND_JUDGE_STARTUP',automated_pid=auto.pid,
                automated_started_at=now())
            await wait_judge(judge,recipe,work)
            state_update(work,phase='SCORING',vlm_started_at=now())
            summary = await evaluate(pairs,recipe,output,provenance)
            while auto.poll() is None: await asyncio.sleep(5)
            if auto.returncode != 0: raise RuntimeError('Automated metrics failed; see automated.log')
        finally: stop_owned(auto)
    return summary,read(output/'automated/summary.json')


def publish(work, target, recipe, request, cases, summary, automated, active_gpus):
    from csv_checklist import complete_evaluation
    require_scores_complete(summary,automated)
    output = work/'results'
    baseline = read(Path(recipe['basis'])/'reference_summary.json')
    write(output/'comparison_with_38k.json',dict(reference_checkpoint=read(Path(recipe['basis'])/'reference_run.json')['checkpoint'],
        reference_memory_score_100=baseline['memory_score_100'],memory_score_100=summary['memory_score_100'],
        memory_delta=summary['memory_score_100']-baseline['memory_score_100'],
        reference_gate_pass_rate=baseline['gate_pass_rate'],gate_pass_rate=summary['gate_pass_rate'],
        reference_dp=8,actual_dp=len(active_gpus),actual_gpus=active_gpus,
        note='Same native protocol and inputs. Actual idle-GPU parallelism is recorded; VLM outputs are not guaranteed bit-identical.'))
    provenance = output/'provenance'; provenance.mkdir()
    shutil.copytree(work/'code',provenance/'adapter')
    shutil.copyfile(str(recipe['basis'])+'.tar.gz',provenance/'memorybench38k_basis.tar.gz')
    shutil.copyfile(work/'preflight.json',provenance/'preflight.json')
    destination = target/'results'/recipe['evaluation_run']
    # New named directory only. A partial publication is retained for diagnosis;
    # an explicit recovery is required instead of overwriting someone else's run.
    total_files = publish_closed_tree(output,destination,
        lambda **values:state_update(work,**values))
    hashes = {name:digest(output/name) for name in ('summary.json','automated/summary.json','run.json','comparison_with_38k.json')}
    if any(digest(destination/name) != expected for name,expected in hashes.items()):
        raise ValueError('Published score/protocol readback differs')
    record = dict(upload_id=recipe['evaluation_run'],uploaded_at=now(),checkpoint_name=target.name,
        artifact_type='results',run_status='COMPLETE',benchmark=dict(platform='modelscope',repo_type='dataset',
            repo_id=recipe['benchmark_repo'],commit=recipe['benchmark_commit'],case_root='.'),
        cases=[c['case_id'] for c in cases],artifact_paths=[f'v0.5/ckps/{target.name}/results/{recipe["evaluation_run"]}/'],
        prediction_run=request['run_name'],model_publication_sha256=request['publication_sha256'],
        recipe=RECIPE_ID,core_sha256=hashes,published_files=total_files,
        benchmark_checkout=read(work/'preflight.json')['benchmark_checkout'])
    marker = target/'results/.uploads'/f'{recipe["evaluation_run"]}.json'
    marker.parent.mkdir(exist_ok=True)
    with marker.open('x') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    if read(marker) != record: raise ValueError('Result closure readback differs')
    archive = target.parents[2]
    checklist = archive/'EVAL_CHECKLIST.csv'
    lock = Path('/mnt/cfs/d22676/husicheng/experiments/bos_daily_20261010/publisher.lock')
    with lock.open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX)
        before = checklist.read_bytes()
        updated = complete_evaluation(before.decode(),target.name,request['run_name'],
            recipe['evaluation_run'],summary,automated,dict(recipe,active_gpus=active_gpus),now()).encode()
        save_checklist_backup(work, 'evaluation', before)
        if checklist.read_bytes() != before: raise ValueError('Concurrent checklist update')
        with checklist.open('wb') as stream:
            stream.write(updated); stream.flush(); os.fsync(stream.fileno())
        if checklist.read_bytes() != updated: raise ValueError('Checklist readback differs')
    state_update(work,status='COMPLETE',phase='COMPLETE',completed_cases=649,errors=0,
        memory_score_100=summary['memory_score_100'],gate_pass_rate=summary['gate_pass_rate'],
        output=str(destination),published_files=total_files,finished_at=now())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work',type=Path)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--target',type=Path)
    args = parser.parse_args()
    recipe = read(Path(__file__).with_name('eval_recipe.json'))
    if args.preflight:
        _,_,result = preflight(recipe,read(args.config),args.target)
        print(json.dumps(result,ensure_ascii=False),flush=True); return
    work = args.work
    judge = None
    def interrupted(signum, frame): raise InterruptedError('Evaluation worker interrupted')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    try:
        request,config = read(work/'request.json'),read(work/'infer_config.json')
        target = validate_request(request,config)
        prediction = target/'predictions'/request['run_name']
        prediction_record = read(target/'predictions/.uploads'/f'{request["run_name"]}.json')
        require_inference_complete(read(work.parent/'state.json'),read(prediction/'batch-completion.json'),
            prediction_record,request)
        cases,receipt,verification = preflight(recipe,config,target)
        expected_paths = {str(Path(c['gt_video']).parent.parent.relative_to(recipe['dataset'])) for c in cases}
        if set(prediction_record['cases']) != expected_paths:
            raise ValueError('Closed inference case set differs from the 38k reference')
        write(work/'preflight.json',verification)
        output = work/'results'; output.mkdir()
        # Start the owned judge immediately, overlapping its cold/warm startup
        # with CPU media staging and then native automated metrics.
        judge,active_gpus = start_judge(recipe,work)
        state_update(work,phase='PREPARING_MEDIA_AND_JUDGE_STARTUP')
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            pairs = list(pool.map(lambda c:prepare_case(c,work,target),cases))
        provenance = dict(recipe=RECIPE_ID,checkpoint=target.name,prediction_run=request['run_name'],
            evaluation_run=recipe['evaluation_run'],evaluator_commit=COMMIT,scoring_version=SCORING_VERSION,
            dataset_commit=recipe['benchmark_commit'],manifest_sha256=receipt['manifest_sha256'],expected_cases=649,
            benchmark_checkout=verification['benchmark_checkout'],
            eligible_gpus=GPUS,gpu_policy=recipe['gpu_policy'],tp=1,
            api_concurrency=512,case_workers=64,automated_workers=32,
            temperature=0,thinking=False,max_tokens=8192,
            reference_parallelism=receipt['observed_reference_parallelism'],
            reference_checkpoint=receipt['reference_checkpoint'],model_publication_sha256=request['publication_sha256'],
            retry_policy=recipe['retry_policy'],started_at=now())
        from inventory_repair import POLICY
        provenance['inventory_format_repair'] = dict(policy=POLICY,
            trigger='native Inventory exhausted retries with only the both-zero ObjectGroup error',
            change='response_format only; original prompts/images/native validators and scoring preserved',
            valid_responses_unchanged=True,first_schema_valid_response_final=True)
        provenance.update(gpus=active_gpus,dp=len(active_gpus),startup_policy=runtime_policy(recipe))
        write(output/'run.json',dict(**provenance,cases=cases))
        summary,automated = asyncio.run(score(pairs,recipe,work,output,provenance,judge))
        stop_owned(judge); judge = None
        state_update(work,phase='PUBLISHING')
        publish(work,target,recipe,request,cases,summary,automated,active_gpus)
    except BaseException:
        state_update(work,status='FAILED',error=traceback.format_exc())
        raise
    finally:
        stop_owned(judge)


if __name__ == '__main__': main()
