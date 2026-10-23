"""Source isolation and real RGB/MCAP -> annotation -> staged case integration."""
import json
import shutil
import subprocess
from fractions import Fraction

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mem_case_anno_demo.lightwheel import build_app
from mem_case_anno_demo.export_lightwheel import export_episode

UUID='00003af1-e42f-4b55-9c48-6b4c7b6fbcbb'


@pytest.fixture
def source(tmp_path):
    root=tmp_path/'training'
    recording=root/'lightwheel-egopro'/UUID
    recording.mkdir(parents=True)
    media=tmp_path/'media';media.mkdir()
    video=media/f'{UUID}.mp4'
    with av.open(str(video),'w') as c:
        s=c.add_stream('libx264',rate=30)
        s.width,s.height,s.pix_fmt=64,48,'yuv420p'
        s.options={'bf':'0','tune':'zerolatency'}
        for i in range(12):
            frame=av.VideoFrame.from_ndarray(np.full((48,64,3),i*20,dtype=np.uint8),format='rgb24')
            frame.pts=i;frame.time_base=Fraction(1,30)
            for packet in s.encode(frame):c.mux(packet)
        for packet in s.encode():c.mux(packet)
    timestamps=[i*33_333_333 for i in range(12)]
    video.with_suffix('.timestamps.json').write_text(json.dumps(dict(sensor_timestamp_ns=timestamps,source_time_origin_ns=123456789)))
    complete=dict(family='EgoPro',source=dict(source_path='/missing/raw.mcap',source_time_origin_ns=123456789,
        source_metadata=dict(task_name='test task',devices=[dict(device_type='head_left_camera',fps='30')])))
    (recording/'COMPLETE.json').write_text(json.dumps(complete))
    return root,media,recording,video,complete


def annotation(client):
    case=client.get('/api/cases').json()[0]
    return dict(id='new-lightwheel-episode',case_id=case['id'],view_id='left',revision=0,
                status='complete',annotation_scope='prefix_generation',
                marks=dict(prefix_start=0,prefix_end=4,t=6,eval_end=10))


def test_mp4_annotation_export_and_isolation(source,tmp_path):
    root,media,recording,video,complete=source
    before={p:p.read_bytes() for p in [video,recording/'COMPLETE.json']}
    out=tmp_path/'annotations'
    with TestClient(build_app(root,out,media)) as c:
        assert c.get('/api/health').json()['source_kind']=='lightwheel'
        a=annotation(c)
        base=f"/api/cases/{a['case_id']}/views/left"
        meta=c.get(base).json()
        assert meta['frame_count']==12
        assert meta['sensor_timestamp_ns'][6]==6*33_333_333
        assert c.get(base+'/video',headers={'Range':'bytes=0-99'}).status_code==206
        assert c.get(base+'/frames/3.jpg').status_code==200
        saved=c.post('/api/annotations',json=a)
        assert saved.status_code==200,saved.text
        row=saved.json()
        assert row['anchor']['frame']==3
        assert row['source']['lightwheel']['original_source']==str(video)
        assert c.post('/api/annotations',json=a).status_code==409
        exported=c.get('/api/export/evaluation')
        assert len(exported.text.splitlines())==1
        assert c.get('/api/export/inference').status_code==200
        assert c.post(f'/api/lightwheel/recordings/lightwheel-egopro/{UUID}').status_code==200
    if not shutil.which('ffmpeg'):pytest.skip('ffmpeg required for staged RGB export')
    case=export_episode(row,tmp_path/'staging')
    for name in ['input/rgb_prefix.mp4','ground_truth/future.mp4']:
        with av.open(str(case/name)) as v:
            assert sum(1 for _ in v.decode(video=0))==4
    with av.open(str(case/'input/anchor.png')) as v:
        image=next(v.decode(video=0)).to_ndarray(format='rgb24')
        assert abs(float(image.mean())-60)<5
    assert json.loads((case/'case.json').read_text())['status'].startswith('rgb_draft')
    with pytest.raises(FileExistsError):export_episode(row,tmp_path/'staging')
    assert all(p.read_bytes()==content for p,content in before.items())


def test_missing_source_is_explicit(source,tmp_path):
    root,media,recording,video,complete=source
    with TestClient(build_app(root,tmp_path/'out')) as c:
        a=annotation(c)
        r=c.get(f"/api/cases/{a['case_id']}/views/left")
        assert r.status_code==422
        assert 'latent' in r.json()['detail']
        assert c.post('/api/lightwheel/recordings/lightwheel-egopro/not-a-uuid').status_code==422


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='ffmpeg required')
def test_native_mcap_lazy_extraction(source,tmp_path):
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    from mcap.writer import Writer
    root,media,recording,video,complete=source
    fd=descriptor_pb2.FileDescriptorProto(name='camera.proto',package='test',syntax='proto3')
    msg=fd.message_type.add(name='Camera')
    msg.field.add(name='format',number=1,type=9,label=1)
    msg.field.add(name='data',number=2,type=12,label=1)
    descriptors=descriptor_pb2.FileDescriptorSet();descriptors.file.add().CopyFrom(fd)
    pool=descriptor_pool.DescriptorPool();pool.Add(fd)
    cls=message_factory.GetMessageClass(pool.FindMessageTypeByName('test.Camera'))
    raw=tmp_path/'left.h264'
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-c:v','copy','-bsf:v','h264_mp4toannexb',str(raw)],check=True)
    with av.open(str(raw)) as v:
        packets=[bytes(p) for p in v.demux(video=0) if p.size]
    mcap=tmp_path/'raw.mcap'
    with mcap.open('wb') as handle:
        writer=Writer(handle);writer.start()
        schema=writer.register_schema(name='test.Camera',encoding='protobuf',data=descriptors.SerializeToString())
        channel=writer.register_channel(topic='/sensor/camera/head_left/video',message_encoding='protobuf',schema_id=schema)
        for i,packet in enumerate(packets):
            stamp=123456789+i*33_333_333
            writer.add_message(channel_id=channel,log_time=stamp,publish_time=stamp,data=cls(format='h264',data=packet).SerializeToString())
        writer.finish()
    complete['source']['source_path']=str(mcap)
    (recording/'COMPLETE.json').write_text(json.dumps(complete))
    out=tmp_path/'out'
    app=build_app(root,out)
    assert not (out/'media_cache').exists()
    with TestClient(app) as c:
        a=annotation(c)
        r=c.get(f"/api/cases/{a['case_id']}/views/left")
        assert r.status_code==200,r.text
        assert r.json()['frame_count']==12
        assert r.json()['sensor_timestamp_ns'][0]==0
        assert c.post('/api/annotations',json=a).status_code==200
    assert list((out/'media_cache').rglob('left.mp4'))


def test_native_wmdata_packages(source,tmp_path):
    import pandas as pd
    root,media,recording,video,complete=source
    wmroot=tmp_path/'wmdata'
    for family in ['lightwheel-egopro','lightwheel-egostandard']:
        package=wmroot/family/(UUID+'__2e818ab2fc')
        (package/'meta').mkdir(parents=True)
        (package/'data/rgb').mkdir(parents=True)
        shutil.copyfile(video,package/'data/rgb/t00000-left.mp4')
        with av.open(str(video)) as v:
            stream=v.streams.video[0];tb=stream.time_base
            pts=[f.pts for f in v.decode(stream)]
        pd.DataFrame([dict(file_id=0,relative_path='data/rgb/t00000-left.mp4',storage_type='video',
            content_type='rgb_video',video_width=64,video_height=48,video_time_base_num=tb.numerator,
            video_time_base_den=tb.denominator,checksum='native-test')]).to_parquet(package/'meta/files.parquet')
        pd.DataFrame([dict(stream_id='rgb-left',file_id=0,logical_sample_start=0,
            video_frame_pts=pts,video_timestamp_ns=[i*40_000_000 for i in range(12)])]).to_parquet(package/'meta/extents.parquet')
        pd.DataFrame([dict(stream_id='rgb-left',camera_frame_id='head-left',pixel_domain_id='native-rgb')]).to_parquet(package/'meta/streams.parquet')
    with TestClient(build_app(wmroot,tmp_path/'native-out',limit=1)) as c:
        cases=c.get('/api/cases').json()
        assert len(cases)==2
        for case in cases:
            base=f"/api/cases/{case['id']}/views/left"
            r=c.get(base)
            assert r.status_code==200,r.text
            meta=r.json()
            assert meta['sensor_timestamp_ns'][6]==240_000_000
            assert meta['lightwheel_source']['source_format']=='wmdata'
            assert c.get(base+'/video',headers={'Range':'bytes=0-99'}).status_code==206
            assert c.get(base+'/frames/4.jpg').status_code==200
        row=annotation(c)
        r=c.post('/api/annotations',json=row)
        assert r.status_code==200,r.text
        assert r.json()['source']['lightwheel']['wmdata_package'].endswith(UUID+'__2e818ab2fc')
        infer=c.get('/api/export/inference').text
        assert 'WMData hand landmarks21' in infer
        family='lightwheel-egopro'
        assert c.post(f'/api/lightwheel/recordings/{family}/{UUID}').status_code==200
        assert c.post(f'/api/lightwheel/recordings/{family}/{UUID}__2e818ab2fc').status_code==200
        assert c.post(f'/api/lightwheel/recordings/{family}/bad').status_code==422
    assert not (tmp_path/'native-out/media_cache').exists()
    # A stricter queue must not prevent saving an already-open human annotation.
    import sqlite3
    screening=tmp_path/'empty-screening.sqlite3'
    with sqlite3.connect(screening) as db:
        db.execute('CREATE TABLE tasks(id TEXT,family TEXT,package TEXT,status TEXT,decision TEXT)')
    with TestClient(build_app(wmroot,tmp_path/'native-out',source_format='wmdata',suitable_only=True,screening_databases=[screening])) as c:
        assert c.get('/api/cases').json()==[]
        edited=dict(row,revision=r.json()['revision'])
        saved=c.post('/api/annotations',json=edited)
        assert saved.status_code==200,saved.text
        assert saved.json()['revision']==2



def test_suitable_only_live_selection_and_latest_precedence(tmp_path):
    import sqlite3
    from mem_case_anno_demo.screening import SuitableSelection
    root=tmp_path/'wmdata'
    package=UUID+'__abc'
    other='0000805e-039e-4bcd-b295-18375b1e4242__def'
    family='lightwheel-egopro'
    for name in [package,other]:(root/family/name).mkdir(parents=True)
    def database(name,rows):
        p=tmp_path/name
        with sqlite3.connect(p) as db:
            db.execute('CREATE TABLE tasks(id TEXT PRIMARY KEY,family TEXT,package TEXT,status TEXT,decision TEXT)')
            db.executemany('INSERT INTO tasks VALUES(?,?,?,?,?)',[(family+'/'+pkg,family,pkg,status,decision) for pkg,status,decision in rows])
        return p
    old=database('old.sqlite3',[(package,'succeeded','suitable'),(other,'succeeded','potential')])
    new=database('new.sqlite3',[])
    selection=SuitableSelection([old,new],refresh_seconds=0)
    assert list(selection.refresh())==[family+'/'+package]
    with TestClient(build_app(root,tmp_path/'out',source_format='wmdata',suitable_only=True,screening_databases=[old,new])) as c:
        c.app.state.catalog.selection.refresh_seconds=0
        cases=c.get('/api/cases').json()
        assert [x['id'] for x in cases]==[family+'__'+package]
        assert c.get('/api/health').json()['selection']=='suitable_only'
        assert c.post(f'/api/lightwheel/recordings/{family}/{UUID}').status_code==200
        assert c.post(f'/api/lightwheel/recordings/{family}/{other}').status_code==422
        with sqlite3.connect(new) as db:
            db.executemany('INSERT INTO tasks VALUES(?,?,?,?,?)',[
                (family+'/'+package,family,package,'succeeded','unsuitable'),
                (family+'/'+other,family,other,'succeeded','suitable')])
        cases=c.get('/api/cases').json()
        assert [x['id'] for x in cases]==[family+'__'+other]
        assert c.post(f'/api/lightwheel/recordings/{family}/{UUID}').status_code==422
    assert list(selection.refresh())==[family+'/'+other]
