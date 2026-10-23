"""Sampling, screening validation, resume state and primary-only routing."""
import copy
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import httpx
import numpy as np
import pytest

from evaluation.source_annotation.schema import VideoAnnotation
from evaluation.source_annotation.sheets import make_sheet
from evaluation.source_annotation.run import ArkClient,PRIMARY,FALLBACK,FatalAPIError,Store,parse_response,export_results


def valid_annotation():
    return dict(overall_description='整理桌面上的纸张',segments=[dict(start_seconds=0,end_seconds=3,description='整理纸张',evidence_frames=[1,12,24])],
        main_objects=['纸张'],memorybench=dict(decision='unsuitable',confidence='medium',rationale='没有明确的记忆读取事件',
        memory_events=[],candidate_cases=[],limitations=['均匀抽帧可能漏掉短暂事件']))


def test_schema_bounds_and_evidence():
    good=valid_annotation()
    VideoAnnotation.model_validate(good).validate_timeline(3)
    bad=copy.deepcopy(good);bad['segments'][0]['evidence_frames']=[25]
    with pytest.raises(ValueError):VideoAnnotation.model_validate(bad).validate_timeline(3)
    bad=copy.deepcopy(good);bad['memorybench']['decision']='suitable'
    with pytest.raises(ValueError):VideoAnnotation.model_validate(bad).validate_timeline(3)
    bad=copy.deepcopy(good);bad['segments'][0]['end_seconds']=4
    with pytest.raises(ValueError):VideoAnnotation.model_validate(bad).validate_timeline(3)
    response=dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(good)))])
    assert parse_response(response,3).memorybench.decision=='unsuitable'
    response['choices'][0]['finish_reason']='length'
    with pytest.raises(ValueError):parse_response(response,3)


def test_contact_sheet_pts_and_readonly_cache(tmp_path):
    video=tmp_path/'source.mp4'
    with av.open(str(video),'w') as c:
        s=c.add_stream('libx264',rate=8);s.width,s.height,s.pix_fmt=64,48,'yuv420p'
        for i in range(24):
            f=av.VideoFrame.from_ndarray(np.full((48,64,3),i*10,dtype=np.uint8),format='rgb24')
            f.pts=i;f.time_base=Fraction(1,8)
            for p in s.encode(f):c.mux(p)
        for p in s.encode():c.mux(p)
    before=video.read_bytes()
    meta=make_sheet(video,tmp_path/'sheets')
    assert (meta['layout_rows'],meta['layout_columns'])==(4,6)
    assert len(meta['actual_pts'])==meta['unique_frames']==24
    assert meta['actual_seconds'][0]==0
    assert meta['actual_seconds'][-1]==pytest.approx(23/8)
    assert all(a<=b for a,b in zip(meta['actual_seconds'],meta['actual_seconds'][1:]))
    assert make_sheet(video,tmp_path/'sheets')==meta
    assert video.read_bytes()==before


@pytest.mark.parametrize('mode',['primary_ok','connection_failure','auth_failure','rate_limit'])
def test_routing_only_falls_back_for_connection_failure(tmp_path,mode):
    key=tmp_path/'key';key.write_text('test-private-key')
    client=ArkClient(key,workers=1,rps=0);client.client.close()
    hosts=[]
    def handler(request):
        hosts.append(request.url.host)
        if len(hosts)==1:
            if mode=='connection_failure':raise httpx.ConnectError('unreachable',request=request)
            if mode=='auth_failure':return httpx.Response(401,json={'error':'denied'})
            if mode=='rate_limit':return httpx.Response(429,headers={'retry-after':'0'},json={'error':'slow down'})
        return httpx.Response(200,json={'choices':[]})
    client.client=httpx.Client(transport=httpx.MockTransport(handler));client.throttle=lambda:None
    try:
        if mode=='auth_failure':
            with pytest.raises(FatalAPIError):client.post({})
        else:client.post({})
        if mode=='connection_failure':assert hosts==[httpx.URL(PRIMARY).host,httpx.URL(FALLBACK).host]
        else:assert set(hosts)=={httpx.URL(PRIMARY).host}
    finally:client.close()


def test_queue_resume_and_compressed_export(tmp_path):
    import csv,zlib
    inventory=tmp_path/'inventory';inventory.mkdir()
    family='lightwheel-egopro'
    (inventory/'summary.json').write_text(json.dumps({family:dict(packages=2)}))
    with (inventory/(family+'.csv')).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['package','video','error']);writer.writeheader()
        writer.writerow(dict(package='one',video='/read-only/one.mp4',error=''))
        writer.writerow(dict(package='two',video='/read-only/two.mp4',error=''))
    db=tmp_path/'queue.sqlite3';store=Store(db,{'test':'v1'})
    store.import_ready(inventory);store.import_ready(inventory)
    assert store.counts()==dict(pending=2)
    row=store.claim(1)[0]
    assert store.counts()==dict(pending=1,running=1)
    store.db.close()
    store=Store(db,{'test':'v1'})
    assert store.counts()==dict(pending=2)
    row=store.claim(1)[0]
    result=dict(annotation=valid_annotation(),request_traces=[dict(usage=dict(prompt_tokens=10,completion_tokens=20,total_tokens=30))])
    path=tmp_path/'result.json'
    store.finish(row,(path,result))
    assert store.counts()==dict(pending=1,succeeded=1)
    raw,decision,tokens=store.db.execute("SELECT result_zlib,decision,total_tokens FROM tasks WHERE status='succeeded'").fetchone()
    assert json.loads(zlib.decompress(raw))==result
    assert decision=='unsuitable' and tokens==30
    output=tmp_path/'export';output.mkdir();export_results(store,output)
    assert json.loads((output/'annotations.jsonl').read_text())==result
    assert json.loads((output/'screening_summary.json').read_text())['records']==1
    store.db.close()
    with pytest.raises(ValueError):Store(db,{'test':'different'})


def explicit_suitable():
    value=valid_annotation()
    value['memorybench'].update(decision='suitable',memory_events=[
        dict(kind='write',description='看到目标',evidence_frames=[2]),
        dict(kind='occlusion',description='柜门关闭，内容不可见',evidence_frames=[8]),
        dict(kind='read',description='开门后同一物体重新出现',evidence_frames=[18])],candidate_cases=[dict(
            prefix_start_seconds=0,prefix_end_seconds=4,future_start_seconds=4,future_end_seconds=10,
            memory_target='柜内红色杯子',why_requires_memory='开门前杯子被柜门挡住',
            write_evidence_frames=[2],occlusion_evidence_frames=[8],read_evidence_frames=[18],
            occlusion_description='不透明柜门关闭，杯子不可见',reappearance_description='再次开门，杯子重新可见',
            same_target_evidence='同一柜格内相同颜色和形状的杯子')])
    return value


def test_suitable_requires_explicit_ordered_occlusion_and_reappearance():
    stamps=np.linspace(0,9.9,24).tolist()
    good=explicit_suitable()
    VideoAnnotation.model_validate(good).validate_timeline(10,actual_seconds=stamps)
    for field,value in [('occlusion_evidence_frames',[]),('occlusion_description',''),('same_target_evidence',''),
                        ('read_evidence_frames',[6]),('future_start_seconds',2)]:
        bad=copy.deepcopy(good);bad['memorybench']['candidate_cases'][0][field]=value
        with pytest.raises(ValueError):VideoAnnotation.model_validate(bad).validate_timeline(10,actual_seconds=stamps)
    bad=copy.deepcopy(good);bad['memorybench']['memory_events']=[e for e in bad['memorybench']['memory_events'] if e['kind']!='occlusion']
    with pytest.raises(ValueError):VideoAnnotation.model_validate(bad).validate_timeline(10)


def test_memory_review_preserves_original_descriptions():
    prior=valid_annotation()
    response=dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(dict(memorybench=explicit_suitable()['memorybench']))))])
    value=parse_response(response,10,np.linspace(0,9.9,24).tolist(),prior)
    assert value.overall_description==prior['overall_description']
    assert value.segments[0].description==prior['segments'][0]['description']
    assert value.main_objects==prior['main_objects']
    assert value.memorybench.decision=='suitable'


def test_occlusion_fork_reuses_negatives_and_rechecks_positives(tmp_path):
    from evaluation.source_annotation.migrate import fork_for_occlusion
    from evaluation.source_annotation.run import build_config,digest
    oldconfig={'model':'ep-20250612104210-ss27q','old_rule':True}
    olddb=tmp_path/'old.sqlite3';store=Store(olddb,oldconfig)
    family='lightwheel-egopro'
    for name in ['negative','positive','pending','cached']:
        store.db.execute('INSERT INTO tasks(id,family,package,video,status) VALUES(?,?,?,?,?)',(family+'/'+name,family,name,'/readonly/'+name+'.mp4','pending'))
    store.db.commit()
    for name,annotation in [('negative',valid_annotation()),('positive',explicit_suitable())]:
        path=tmp_path/(name+'.json')
        result=dict(annotation=annotation,request_traces=[],config_sha256=digest(oldconfig))
        path.write_text(json.dumps(result))
        store.finish(dict(id=family+'/'+name),(path,result))
    oldoutput=tmp_path/'old-output'
    path=oldoutput/family/'ca'/'cached'/'result.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(annotation=valid_annotation(),request_traces=[],config_sha256=digest(oldconfig))))
    store.db.execute("UPDATE tasks SET status='running' WHERE package='cached'");store.db.commit();store.db.close()
    newdb=tmp_path/'new.sqlite3'
    summary=fork_for_occlusion(olddb,oldoutput,newdb,tmp_path/'new-output')
    assert summary['cached_results_recovered']==1
    assert summary['descriptions_retained_without_call']==2
    assert summary['suitable_to_recheck']==1
    new=Store(newdb,build_config())
    assert new.counts()==dict(pending=2,succeeded=2)
    assert new.claim(1)[0]['package']=='positive'
    assert new.db.execute("SELECT prior_result_path FROM tasks WHERE package='positive'").fetchone()[0]==str(tmp_path/'positive.json')
    with __import__('sqlite3').connect(olddb) as old:
        assert old.execute("SELECT decision FROM tasks WHERE package='positive'").fetchone()[0]=='suitable'
        assert old.execute("SELECT status FROM tasks WHERE package='cached'").fetchone()[0]=='running'
