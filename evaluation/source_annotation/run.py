"""Concurrent, resumable Ark source descriptions over completed WMData inventories."""
from __future__ import annotations
import argparse
import base64
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import threading
import time
import zlib
import shutil
import subprocess
import sys
from datetime import datetime,timezone

import httpx
from .schema import VideoAnnotation
from .sheets import VERSION,atomic_json,make_sheet

MODEL='ep-20250612104210-ss27q'
PRIMARY='https://ark.cn-beijing.volces.com/api/v3'
FALLBACK='https://ark-cn-beijing.bytedance.net/api/v3'
FAMILIES=('lightwheel-egopro','lightwheel-egostandard')
PROMPT_PATH=Path(__file__).with_name('PROMPT.md')
POLICY_VERSION='explicit-occlusion-reappearance-v1'

def build_config(model=MODEL):
    return dict(model=model,prompt=PROMPT_PATH.read_text(),sheet_version=VERSION,primary=PRIMARY,
                fallback_only_on_connection_failure=FALLBACK,schema=VideoAnnotation.model_json_schema(),policy_version=POLICY_VERSION)


def now():return datetime.now(timezone.utc).isoformat()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def parse_response(response,duration,actual_seconds=None,prior_annotation=None):
    if len(response.get('choices',[]))!=1:raise ValueError('Expected one response choice')
    choice=response['choices'][0]
    if choice.get('finish_reason')!='stop':raise ValueError('Incomplete response: '+str(choice.get('finish_reason')))
    content=choice['message']['content'].strip()
    if content.startswith('```') and content.endswith('```'):
        content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
    def unique(pairs):
        if len(dict(pairs))!=len(pairs):raise ValueError('Duplicate JSON keys')
        return dict(pairs)
    value=json.loads(content,object_pairs_hook=unique)
    if prior_annotation is not None:
        value=dict(prior_annotation,memorybench=value['memorybench'])
    return VideoAnnotation.model_validate(value).validate_timeline(duration,actual_seconds=actual_seconds)


class FatalAPIError(RuntimeError):pass


class APIUnavailable(RuntimeError):pass


class ArkClient:
    def __init__(self,key_file,workers=24,rps=4):
        self.key=Path(key_file).read_text().strip()
        if not self.key:raise ValueError('Empty API credential')
        self.client=httpx.Client(timeout=httpx.Timeout(210,connect=20,read=180),trust_env=False,
                                 limits=httpx.Limits(max_connections=workers,max_keepalive_connections=workers))
        self.lock=threading.Lock();self.next_request=0.0;self.cooldown_until=0.0;self.rps=rps

    def clean(self,message):return str(message).replace(self.key,'[REDACTED]')

    def throttle(self):
        with self.lock:
            current=time.monotonic()
            target=max(current,self.next_request,self.cooldown_until)
            self.next_request=target+1/self.rps if self.rps>0 else target
        if target>current:time.sleep(target-current)

    def post(self,payload):
        endpoint=PRIMARY
        for attempt in range(7):
            self.throttle()
            try:
                response=self.client.post(endpoint+'/chat/completions',json=payload,
                    headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
            except (httpx.ConnectError,httpx.ConnectTimeout):
                if endpoint==PRIMARY:
                    # Only network connection failure triggers the user-authorized fallback.
                    endpoint=FALLBACK
                    continue
                if attempt==6:raise APIUnavailable('Both primary and fallback could not connect')
                time.sleep(min(30,2**attempt)+random.random());continue
            except httpx.HTTPError as exc:
                if attempt==6:raise APIUnavailable(type(exc).__name__+': '+self.clean(exc)) from None
                time.sleep(min(30,2**attempt)+random.random());continue
            if response.status_code in (401,403,404) or (400<=response.status_code<500 and response.status_code not in (408,409,429)):
                raise FatalAPIError(f'HTTP {response.status_code}: '+self.clean(response.text[:2000]))
            if response.status_code==429 or response.status_code>=500 or response.status_code in (408,409):
                try:delay=float(response.headers.get('retry-after',''))
                except ValueError:delay=min(60,2**(attempt+2))
                delay=min(180,max(1,delay))+random.random()
                with self.lock:self.cooldown_until=max(self.cooldown_until,time.monotonic()+delay)
                if attempt==6:raise APIUnavailable(f'Provider repeatedly returned HTTP {response.status_code}')
                continue
            response.raise_for_status()
            return response.json(),endpoint,response.headers.get('x-request-id') or response.headers.get('x-tt-logid')
        raise APIUnavailable('No completed request')

    def close(self):self.client.close()


class Store:
    def __init__(self,path,config):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tasks(
          id TEXT PRIMARY KEY,family TEXT NOT NULL,package TEXT NOT NULL,video TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending',attempts INTEGER NOT NULL DEFAULT 0,
          result_path TEXT,error TEXT,updated_at TEXT);
        CREATE INDEX IF NOT EXISTS task_status ON tasks(status);
        ''')
        existing={r[1] for r in self.db.execute('PRAGMA table_info(tasks)')}
        for name,kind in [('prior_result_path','TEXT'),('decision','TEXT'),('result_zlib','BLOB'),('prompt_tokens','INTEGER'),('completion_tokens','INTEGER'),('total_tokens','INTEGER')]:
            if name not in existing:self.db.execute(f'ALTER TABLE tasks ADD COLUMN {name} {kind}')
        self.db.execute('CREATE INDEX IF NOT EXISTS task_decision ON tasks(decision,status)')
        self.db.execute('CREATE INDEX IF NOT EXISTS task_review_order ON tasks(status,(prior_result_path IS NULL))')
        previous=self.db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
        if previous and json.loads(previous[0])!=config:raise ValueError('Run configuration changed; use a new database/output directory')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES('config',?)",(json.dumps(config,sort_keys=True),))
        self.db.execute("UPDATE tasks SET status='pending' WHERE status='running'")
        self.db.commit()
        for ident,path in self.db.execute("SELECT id,result_path FROM tasks WHERE status='succeeded' AND result_zlib IS NULL").fetchall():
            self.finish(dict(id=ident),path)

    def import_ready(self,inventory):
        summary_file=Path(inventory)/'summary.json'
        try:summary=json.loads(summary_file.read_text())
        except (FileNotFoundError,json.JSONDecodeError):return
        for family in FAMILIES:
            if family not in summary:continue
            key='imported:'+family
            if self.db.execute('SELECT 1 FROM metadata WHERE key=?',(key,)).fetchone():continue
            count=0;batch=[]
            with (Path(inventory)/(family+'.csv')).open(newline='') as f:
                for row in csv.DictReader(f):
                    ident=family+'/'+row['package']
                    batch.append((ident,family,row['package'],row['video'],
                        'failed' if row['error'] else 'pending',row['error'],now()))
                    count+=1
                    if len(batch)>=1000:
                        self.db.executemany('INSERT OR IGNORE INTO tasks(id,family,package,video,status,error,updated_at) VALUES(?,?,?,?,?,?,?)',batch);batch=[]
                if batch:self.db.executemany('INSERT OR IGNORE INTO tasks(id,family,package,video,status,error,updated_at) VALUES(?,?,?,?,?,?,?)',batch)
            if count!=summary[family]['packages']:raise ValueError(f'Incomplete inventory CSV for {family}: {count}')
            self.db.execute('INSERT INTO metadata VALUES(?,?)',(key,str(count)));self.db.commit()
            print(json.dumps(dict(event='inventory_imported',family=family,recordings=count,time=now())),flush=True)

    def imported(self):return len(self.db.execute("SELECT key FROM metadata WHERE key LIKE 'imported:%'").fetchall())
    def claim(self,n):
        rows=self.db.execute("SELECT id,family,package,video,prior_result_path FROM tasks WHERE status='pending' ORDER BY (prior_result_path IS NULL),rowid LIMIT ?",(n,)).fetchall()
        self.db.executemany("UPDATE tasks SET status='running',attempts=attempts+1,updated_at=? WHERE id=?",[(now(),r[0]) for r in rows]);self.db.commit()
        return [dict(zip(('id','family','package','video','prior_result_path'),row)) for row in rows]
    def finish(self,row,path,error=None,requeue=False):
        status='pending' if requeue else 'failed' if error else 'succeeded'
        result=None
        if path:
            if isinstance(path,tuple):path,result=path
            else:result=json.loads(Path(path).read_text())
        decision=result['annotation']['memorybench']['decision'] if result else None
        tokens={key:sum((t.get('usage') or {}).get(key,0) or 0 for t in result['request_traces']) if result else 0
                for key in ('prompt_tokens','completion_tokens','total_tokens')}
        self.db.execute('UPDATE tasks SET status=?,result_path=?,error=?,updated_at=?,decision=?,result_zlib=?,prompt_tokens=?,completion_tokens=?,total_tokens=? WHERE id=?',
            (status,str(path) if path else None,error,now(),decision,zlib.compress(json.dumps(result,ensure_ascii=False,allow_nan=False).encode(),6) if result else None,
             tokens['prompt_tokens'],tokens['completion_tokens'],tokens['total_tokens'],row['id']));self.db.commit()
    def counts(self):return dict(self.db.execute('SELECT status,count(*) FROM tasks GROUP BY status').fetchall())


def annotate(row,args,client,config,prep):
    folder=Path(args.output_root)/row['family']/row['package'][:2]/row['package']
    result_path=folder/'result.json'
    if result_path.is_file():
        previous=json.loads(result_path.read_text())
        if previous.get('config_sha256')==digest(config):
            VideoAnnotation.model_validate(previous['annotation']).validate_timeline(previous['sheet']['duration_seconds'])
            return result_path,previous
    started=time.monotonic()
    prior=json.loads(Path(row['prior_result_path']).read_text()) if row.get('prior_result_path') else None
    sheet_folder=Path(prior['sheet']['image_path']).parent if prior else folder
    with prep:sheet=make_sheet(row['video'],sheet_folder)
    package_path=Path(row['video']).parents[2]
    dataset_info=json.loads((package_path/'dataset.json').read_text())
    prompt=config['prompt']+'\n视频总时长：'+f"{sheet['duration_seconds']:.6f} 秒。\n实际抽样秒数："+json.dumps(sheet['actual_seconds'])
    if prior:
        prompt+='\n本次仅复核 memorybench：整体描述、分段活动和主要物体已保存，不要重写。只返回 {\"memorybench\": {...}}，其中 memorybench 严格使用上面的结构及新遮挡硬条件。'
    raw_image=Path(sheet['image_path']).read_bytes()
    image_url='data:image/jpeg;base64,'+base64.b64encode(raw_image).decode()
    payload=dict(model=args.model,messages=[dict(role='user',content=[dict(type='text',text=prompt),
        dict(type='image_url',image_url=dict(url=image_url,detail='high'))])],
        max_tokens=4096,response_format=dict(type='json_object'),stream=False)
    traces=[];last_error=None
    for validation_attempt in range(3):
        response,endpoint,request_id=client.post(payload)
        traces.append(dict(endpoint=endpoint,request_id=request_id,provider_id=response.get('id'),
            usage=response.get('usage'),finish_reason=response.get('choices',[{}])[0].get('finish_reason'),time=now()))
        try:annotation=parse_response(response,sheet['duration_seconds'],sheet['actual_seconds'],prior['annotation'] if prior else None);break
        except (ValueError,KeyError,IndexError,TypeError) as exc:
            last_error=client.clean(exc)
            if validation_attempt==2:
                atomic_json(folder/'invalid_response.json',dict(traces=traces,provider_response=response,error=last_error))
                raise ValueError('Response failed schema/timeline validation: '+last_error) from None
            payload['messages'][0]['content'][0]['text']=prompt+'\n上次输出未通过校验，请纠正：'+last_error[:2000]
            if response.get('choices',[{}])[0].get('finish_reason')=='length':payload['max_tokens']=8192
    if prior:traces=prior['request_traces']+traces
    result=dict(policy_version=POLICY_VERSION,schema_version='memorybench.source_video_annotation.v1',status='vlm_screened_requires_human_review',
        id=row['id'],family=row['family'],package=row['package'],recording_id=row['package'].split('__')[0],
        source_video=row['video'],wmdata_dataset=dataset_info,model=args.model,config_sha256=digest(config),
        prompt_sha256=hashlib.sha256(config['prompt'].encode()).hexdigest(),sheet=sheet,
        annotation=annotation.model_dump(),provider_response=response,request_traces=traces,
        created_at=now(),elapsed_seconds=time.monotonic()-started,
        boundary_status='coarse_uniform_samples_not_frame_accurate',human_verified=False)
    if prior:
        result['description_provenance']=dict(result_path=row['prior_result_path'],prompt_sha256=prior['prompt_sha256'],reused_unchanged=True)
    atomic_json(result_path,result)
    return result_path,result


def write_status(store,args,started):
    counts=store.counts()
    summary=dict(updated_at=now(),counts=counts,imported_sources=store.imported(),expected_sources=len(FAMILIES),
        elapsed_seconds=time.monotonic()-started,pause_reason=getattr(args,'pause_reason',None),model=args.model,workers=args.workers,
        prepare_workers=args.prepare_workers,requests_per_second=args.requests_per_second,
        completed=store.imported()==len(FAMILIES) and not counts.get('pending',0) and not counts.get('running',0))
    summary['decisions']=dict(store.db.execute("SELECT decision,count(*) FROM tasks WHERE status='succeeded' GROUP BY decision").fetchall())
    usage=store.db.execute("SELECT COALESCE(sum(prompt_tokens),0),COALESCE(sum(completion_tokens),0),COALESCE(sum(total_tokens),0) FROM tasks WHERE status='succeeded'").fetchone()
    summary['usage']=dict(zip(('prompt_tokens','completion_tokens','total_tokens'),usage))
    atomic_json(Path(args.output_root)/'progress.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def export_results(store,output):
    output=Path(output);tmp=output/'annotations.jsonl.tmp'
    decisions={};tokens={};count=0
    with tmp.open('w') as handle:
        for (raw,) in store.db.execute("SELECT result_zlib FROM tasks WHERE status='succeeded' ORDER BY id"):
            row=json.loads(zlib.decompress(raw))
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');count+=1
            decision=row['annotation']['memorybench']['decision'];decisions[decision]=decisions.get(decision,0)+1
            for trace in row['request_traces']:
                for k,v in (trace.get('usage') or {}).items():
                    if isinstance(v,int):tokens[k]=tokens.get(k,0)+v
    tmp.replace(output/'annotations.jsonl')
    atomic_json(output/'screening_summary.json',dict(records=count,decisions=decisions,usage=tokens,updated_at=now()))


def supervise(args):
    child_args=[a for a in sys.argv[1:] if a!='--supervise']
    recoveries=0;failed_passes=0;rate=args.requests_per_second
    while True:
        code=subprocess.run([sys.executable,'-m','evaluation.source_annotation.run',*child_args]).returncode
        if code==0:return
        try:progress=json.loads((args.output_root/'progress.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):raise SystemExit(code)
        reason=progress.get('pause_reason') or ''
        temporary=any(word in reason.lower() for word in ('provider repeatedly','could not connect','timeout','connecterror'))
        if temporary:
            recoveries+=1
            if '429' in reason:rate=max(0.25,rate/2)
            delay=min(900,60*2**min(4,recoveries-1))
            print(json.dumps(dict(event='supervisor_retry',reason=reason,delay_seconds=delay,requests_per_second=rate,time=now())),flush=True)
            time.sleep(delay)
            child_args.extend(['--requests-per-second',str(rate)])
            continue
        if not reason and progress.get('completed') and progress.get('counts',{}).get('failed') and failed_passes<2:
            failed_passes+=1
            if '--retry-failed' not in child_args:child_args.append('--retry-failed')
            time.sleep(30);continue
        raise SystemExit(code)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,default=Path('/var/lib/fmh/data/lightwheel-duration-stats-20261023'))
    p.add_argument('--output-root',type=Path,default=Path('/mnt/bos/tm9b1x/fmh/data/MemoryBenchAnnotationsV2/lightwheel/source_vlm/ark-20261023-v3'))
    p.add_argument('--database',type=Path,default=Path('/var/lib/fmh/runtime/lightwheel-source-vlm/tasks-v3.sqlite3'))
    p.add_argument('--key-file',type=Path,default=Path('/var/lib/fmh/credentials/ark-memorybench-api-key'))
    p.add_argument('--model',default=MODEL)
    p.add_argument('--workers',type=int,default=24)
    p.add_argument('--prepare-workers',type=int,default=8)
    p.add_argument('--requests-per-second',type=float,default=4)
    p.add_argument('--max-new',type=int,default=0,help='Pilot: stop after this many new tasks; 0 = all')
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--export-only',action='store_true')
    p.add_argument('--supervise',action='store_true',help='Restart transient API outages and retry failed recordings twice')
    args=p.parse_args()
    if min(args.workers,args.prepare_workers)<1:raise ValueError('Worker counts must be positive')
    args.output_root.mkdir(parents=True,exist_ok=True)
    if args.supervise:supervise(args);return
    config=build_config(args.model)
    # Local-only run lock prevents two tmux sessions claiming the same queue.
    import fcntl
    args.database.parent.mkdir(parents=True,exist_ok=True)
    with args.database.with_suffix('.lock').open('w') as run_lock:
        fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store=Store(args.database,config)
        if args.retry_failed:store.db.execute("UPDATE tasks SET status='pending' WHERE status='failed'");store.db.commit()
        atomic_json(args.output_root/'run_config.json',dict(config=config,config_sha256=digest(config),created_at=now()))
        if args.export_only:export_results(store,args.output_root);return
        client=ArkClient(args.key_file,args.workers,args.requests_per_second)
        prep=threading.Semaphore(args.prepare_workers)
        started=time.monotonic();last_report=0;submitted=0;fatal=None;inflight={}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                while True:
                    store.import_ready(args.inventory)
                    if shutil.disk_usage(args.database.parent).free < 512*1024*1024:
                        fatal='Local queue storage below 512 MiB free; pending tasks preserved'
                    available=0 if fatal else args.workers-len(inflight)
                    if args.max_new:available=min(available,max(0,args.max_new-submitted))
                    for row in store.claim(available):
                        inflight[pool.submit(annotate,row,args,client,config,prep)]=row;submitted+=1
                    if inflight:
                        done,_=concurrent.futures.wait(inflight,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
                        for future in done:
                            row=inflight.pop(future)
                            try:store.finish(row,future.result())
                            except (FatalAPIError,APIUnavailable) as exc:
                                fatal=client.clean(exc);store.finish(row,None,fatal,requeue=True)
                            except Exception as exc:
                                error=client.clean(exc)
                                store.finish(row,None,error)
                                print(json.dumps(dict(event='task_failed',id=row['id'],error=error,time=now())),flush=True)
                    elif (args.max_new and submitted>=args.max_new) or store.imported()==len(FAMILIES):break
                    else:time.sleep(5)
                    if time.monotonic()-last_report>=30:write_status(store,args,started);last_report=time.monotonic()
                    if fatal:
                        # Drain already-running work without claiming more API requests.
                        for future,row in list(inflight.items()):
                            try:store.finish(row,future.result())
                            except Exception as exc:store.finish(row,None,client.clean(exc),requeue=True)
                        inflight.clear()
                        break
                    if args.max_new and submitted>=args.max_new and not inflight:break
        finally:
            args.pause_reason=fatal
            client.close();write_status(store,args,started)
            export_results(store,args.output_root)
        if fatal:raise SystemExit('Paused with pending tasks: '+fatal)
        if store.counts().get('failed'):raise SystemExit('Some recordings failed; rerun with --retry-failed')


if __name__=='__main__':main()
