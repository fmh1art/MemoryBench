"""Exercise annotation boundaries, data isolation, persistence and VFR seeking."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mem_case_anno_demo.server import create_app


@pytest.fixture(params=["rgb-file", 0])
def client(tmp_path, request):
    case = tmp_path / "data" / "case-vfr"
    (case / "meta").mkdir(parents=True)
    (case / "data/rgb").mkdir(parents=True)
    video = case / "data/rgb/left.mp4"
    # Irregular PTS, deliberately different from sensor time.
    pts_ms = [100, 140, 160, 210, 240, 280, 300, 360, 400, 430, 490, 520]
    with av.open(str(video), "w") as c:
        s = c.add_stream("libx264", rate=30)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        s.time_base = Fraction(1, 1000)
        s.codec_context.time_base = Fraction(1, 1000)
        for i, pts in enumerate(pts_ms):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), i * 20, dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in s.encode(frame):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)
    with av.open(str(video)) as c:
        stream = c.streams.video[0]
        tb = stream.time_base
        pts = [f.pts for f in c.decode(stream)]
    pd.DataFrame(
        [
            dict(
                file_id=request.param,
                relative_path="data/rgb/left.mp4",
                storage_type="video",
                content_type="rgb_video",
                video_width=64,
                video_height=48,
                video_time_base_num=tb.numerator,
                video_time_base_den=tb.denominator,
                checksum="test-source",
            )
        ]
    ).to_parquet(case / "meta/files.parquet")
    pd.DataFrame(
        [
            dict(
                stream_id="rgb-stream",
                file_id=request.param,
                logical_sample_start=0,
                logical_sample_count=len(pts),
                video_frame_pts=pts,
                video_timestamp_ns=[
                    1_000_000_000 + i * 50_000_000 for i in range(len(pts))
                ],
            )
        ]
    ).to_parquet(case / "meta/extents.parquet")
    pd.DataFrame(
        [
            dict(
                stream_id="rgb-stream",
                camera_frame_id="camera-left",
                pixel_domain_id="rgb-left-native",
            )
        ]
    ).to_parquet(case / "meta/streams.parquet")
    app = create_app(tmp_path / "data", tmp_path / "annotations")
    with TestClient(app) as c:
        yield c


def valid(client):
    case = client.get("/api/cases").json()[0]
    return dict(
        id="test-event",
        case_id=case["id"],
        view_id=case["views"][0]["id"],
        revision=0,
        status="complete",
        title="Drawer",
        marks=dict(prefix_start=0, t=5, action_end=7, eval_start=8, eval_end=12),
        evidence=[1, 3],
        target_description="red object secret label",
        expected_state="inside drawer",
        gate_criterion="drawer interior visible",
        references=[dict(frame=9, box=[0.1, 0.2, 0.7, 0.8])],
    )


def test_round_trip_exports_gate_and_conflict(client):
    a = valid(client)
    response = client.post("/api/annotations", json=a)
    assert response.status_code == 200, response.text
    row = response.json()
    assert row["segments"]["prefix"]["end_exclusive"]["frame"] == 5
    assert row["anchor"]["frame"] == 4  # Never use first unseen RGB as anchor.
    assert row["segments"]["generation"]["frame_count"] == 7
    assert row["segments"]["evaluation"]["frame_count"] == 4
    assert row["segments"]["evaluation"]["end_exclusive"]["sensor_timestamp_ns"] is None
    assert row["evaluation_policy"]["gate_pass"] is None
    assert row["evaluation_policy"]["score"] == "gate_pass ? object_match_score : 0"
    assert row["memory_delay_sensor_seconds"] == pytest.approx(0.1)
    assert client.get("/api/annotations").json() == [row]
    inference = client.get("/api/export/inference")
    assert inference.headers["x-episode-count"] == "1"
    import json

    request = json.loads(inference.text)
    for forbidden in [
        "target_description",
        "expected_state",
        "gate_criterion",
        "references",
        "evidence",
        "evaluation",
        "title",
    ]:
        assert forbidden not in request
    assert "secret label" not in inference.text
    assert request["future_actions"] == row["segments"]["generation"]
    assert request["future_head_pose"] == row["segments"]["generation"]
    assert client.post("/api/annotations", json=a).status_code == 409
    a["revision"] = row["revision"]
    a["status"] = "draft"
    assert client.post("/api/annotations", json=a).status_code == 200
    assert client.get("/api/export/inference").text == ""
    assert list((client.app.state.output_root / "history/test-event").glob("*.json"))


@pytest.mark.parametrize(
    "change",
    [
        {"marks": dict(prefix_start=0, t=5, action_end=9, eval_start=8, eval_end=12)},
        {
            "marks": dict(
                prefix_start=0, t=None, action_end=7, eval_start=8, eval_end=12
            )
        },
        {"marks": dict(prefix_start=5, t=5, action_end=7, eval_start=8, eval_end=12)},
        {"marks": dict(prefix_start=0, t=5, action_end=7, eval_start=8, eval_end=13)},
        {"marks": dict(prefix_start=0, t=5, action_end=7, eval_start=8, eval_end=8)},
    ],
)
def test_invalid_complete_annotation_is_rejected(client, change):
    a = valid(client)
    a.update(change)
    assert client.post("/api/annotations", json=a).status_code == 422
    assert client.get("/api/annotations").json() == []


def keyframes_only(client):
    return {
        key: value
        for key, value in valid(client).items()
        if key in ("id", "case_id", "view_id", "revision", "status", "marks")
    }


def test_keyframes_only_multiple_samples_in_one_case(client):
    first = keyframes_only(client)
    r = client.post("/api/annotations", json=first)
    assert r.status_code == 200, r.text
    row = r.json()
    assert row["annotation_scope"] == "keyframes"
    assert row["validation_errors"] == []
    for key in (
        "title",
        "evidence",
        "target_description",
        "expected_state",
        "gate_criterion",
        "references",
        "notes",
    ):
        assert key not in row
    second = first | {
        "id": "test-event-2",
        "marks": dict(prefix_start=1, t=6, action_end=8, eval_start=9, eval_end=12),
    }
    assert client.post("/api/annotations", json=second).status_code == 200
    assert client.get("/api/cases").json()[0]["completed"] == 2
    rows = client.get("/api/annotations", params={"case_id": first["case_id"]}).json()
    assert len(rows) == 2
    import json

    for kind in ("inference", "evaluation"):
        exported = [
            json.loads(line)
            for line in client.get(f"/api/export/{kind}").text.splitlines()
        ]
        assert len(exported) == 2
        key = "annotation_id" if kind == "inference" else "id"
        assert {r[key] for r in exported} == {"test-event", "test-event-2"}
        assert {r["case_id"] for r in exported} == {first["case_id"]}
    # Editing one sample must leave the other untouched.
    first["revision"] = row["revision"]
    first["marks"]["eval_start"] = 9
    assert client.post("/api/annotations", json=first).status_code == 200
    rows = client.get("/api/annotations").json()
    assert next(r for r in rows if r["id"] == second["id"])["marks"] == second["marks"]


def test_editing_keyframes_preserves_legacy_labels_without_blocking(client):
    original = valid(client)
    saved = client.post("/api/annotations", json=original).json()
    update = keyframes_only(client)
    update["revision"] = saved["revision"]
    # The new prefix excludes old evidence, and the new evaluation excludes old ROI.
    update["marks"] = dict(
        prefix_start=3, t=6, action_end=8, eval_start=10, eval_end=12
    )
    r = client.post("/api/annotations", json=update)
    assert r.status_code == 200, r.text
    result = r.json()
    for key in (
        "title",
        "evidence",
        "target_description",
        "expected_state",
        "gate_criterion",
        "references",
    ):
        assert result[key] == saved[key]
    assert result["auxiliary_annotations_need_review"] is True
    assert "memory_delay_sensor_seconds" not in result
    assert "evidence" not in result["segments"]


def test_draft_and_original_frame_pts(client):
    a = valid(client)
    a["status"] = "draft"
    a["marks"]["t"] = None
    assert client.post("/api/annotations", json=a).status_code == 200
    assert client.get("/api/export/evaluation").text == ""
    base = f"/api/cases/{a['case_id']}/views/{a['view_id']}"
    meta = client.get(base).json()
    assert meta["frame_count"] == 12
    assert meta["media_seconds"][0] > 0
    assert len(set(np.round(np.diff(meta["media_seconds"]), 3))) > 1
    from PIL import Image
    from io import BytesIO

    for i in [0, 5, 11]:
        r = client.get(f"{base}/frames/{i}.jpg")
        assert r.status_code == 200
        assert int(r.headers["x-media-pts"]) == meta["media_pts"][i]
        # A wrong VFR seek returns a visibly different grey level.
        assert np.asarray(Image.open(BytesIO(r.content))).mean() == pytest.approx(
            i * 20, abs=3
        )
    assert client.get(f"{base}/frames/12.jpg").status_code == 422
    r = client.get(f"{base}/video", headers={"Range": "bytes=0-127"})
    assert r.status_code == 206 and len(r.content) == 128
    assert client.get("/api/cases/invalid/views/invalid/video").status_code == 404


def test_prefix_generation_roundtrip(client, tmp_path):
    app = create_app(tmp_path / 'data', tmp_path / 'ranges', 'prefix_generation')
    with TestClient(app) as c:
        case = c.get('/api/cases').json()[0]
        payload = dict(id='range-one', case_id=case['id'], view_id=case['views'][0]['id'],
                       annotation_scope='prefix_generation', status='complete', revision=0,
                       marks=dict(prefix_start=1, t=5, eval_end=12))
        result = c.post('/api/annotations', json=payload)
        assert result.status_code == 200, result.text
        row = result.json()
        assert set(row['segments']) == {'prefix', 'generation'}
        assert row['segments']['prefix']['end_exclusive'] == row['segments']['generation']['start']
        assert row['anchor']['frame'] == 4
        assert row['segments']['generation']['end_exclusive']['is_eof']
        assert row['hand_source'] == 'source'
        assert 'evaluation_policy' not in row
        payload['id'] = 'range-two'
        assert c.post('/api/annotations', json=payload).status_code == 200
        assert len(c.get('/api/annotations').json()) == 2
        exported = c.get('/api/export/inference').text.strip().splitlines()
        assert len(exported) == 2
        import json
        assert json.loads(exported[0])['generate']['frame_count'] == 7
        payload['marks']['t'] = 12
        assert c.post('/api/annotations', json=payload).status_code == 422
        payload['marks']['t'] = 5
        payload['id'] = 'range-one'
        assert c.post('/api/annotations', json=payload).status_code == 409
        payload['revision'] = 1
        payload['marks']['prefix_start'] = 0
        assert c.post('/api/annotations', json=payload).status_code == 200
        assert (tmp_path / 'ranges/history/range-one/000001.json').exists()
        assert 'ACWM_RANGE_MODE=true' in c.get('/').text


@pytest.mark.parametrize('p,pe,g,ge', [(0,3,7,12),(0,8,4,10),(0,5,0,5),(7,12,0,3)])
def test_independent_segments_assembled_timeline(client, tmp_path, p, pe, g, ge):
    import json
    with TestClient(create_app(tmp_path/'data', tmp_path/'independent', 'prefix_generation')) as c:
        case=c.get('/api/cases').json()[0]
        payload=dict(id='splice',case_id=case['id'],view_id=case['views'][0]['id'],revision=0,
                     annotation_scope='prefix_generation',status='complete',boundaries_linked=False,
                     marks=dict(prefix_start=p,prefix_end=pe,t=g,eval_end=ge))
        response=c.post('/api/annotations',json=payload)
        assert response.status_code==200,response.text
        row=response.json(); mapping=row['assembly']
        assert row['boundaries_linked'] is False
        assert c.get('/api/annotations').json()[0]['boundaries_linked'] is False
        expected=list(range(p,pe))+list(range(g,ge))
        assert mapping['source_frame_indices']==expected
        assert mapping['source_sensor_timestamp_ns']==[1_000_000_000+i*50_000_000 for i in expected]
        assert mapping['case_timestamp_ns']==[i*1_000_000_000//30 for i in range(len(expected))]
        assert mapping['generation_frame_range']==[pe-p,len(expected)]
        assert row['anchor']['frame']==pe-1
        assert mapping['source_intervals_overlap']==(max(p,g)<min(pe,ge))
        exported=json.loads(c.get('/api/export/inference').text)
        assert exported['assembly']==mapping
        assert exported['rgb_prefix']['end_exclusive']['frame']==pe
        assert exported['generate']['start']['frame']==g
        assert exported['schema_version']=='acwm.memory_inference_request.v2'


def test_range_case_name_persists_and_exports(client, tmp_path):
    import json
    with TestClient(create_app(tmp_path/'data', tmp_path/'names', 'prefix_generation')) as c:
        case=c.get('/api/cases').json()[0]
        payload=dict(id='named-case',case_id=case['id'],view_id=case['views'][0]['id'],revision=0,
                     annotation_scope='prefix_generation',status='complete',title=' 抽屉取物_第一次 ',
                     marks=dict(prefix_start=0,prefix_end=3,t=7,eval_end=12))
        row=c.post('/api/annotations',json=payload).json()
        assert row['title']=='抽屉取物_第一次'
        assert c.get('/api/annotations').json()[0]['title']==row['title']
        assert json.loads(c.get('/api/export/evaluation').text)['title']==row['title']
        inference=json.loads(c.get('/api/export/inference').text)
        assert inference['case_name']==row['title']
        assert inference['text_condition']==''
        payload.update(title='抽屉取物_改名',revision=1)
        changed=c.post('/api/annotations',json=payload).json()
        assert changed['title']=='抽屉取物_改名'
        assert changed['marks']==row['marks']
        old=json.loads((tmp_path/'names/history/named-case/000001.json').read_text())
        assert old['title']==row['title']


def test_delete_keeps_backup_checks_revision_and_blocks_resurrection(client, tmp_path):
    import json
    with TestClient(create_app(tmp_path/'data',tmp_path/'delete-test','prefix_generation')) as c:
        case=c.get('/api/cases').json()[0]
        payload=dict(id='delete-me',case_id=case['id'],view_id=case['views'][0]['id'],revision=0,
                     annotation_scope='prefix_generation',status='complete',title='待删除',
                     marks=dict(prefix_start=0,prefix_end=3,t=7,eval_end=12))
        original=c.post('/api/annotations',json=payload).json()
        payload['id']='keep-me';other=c.post('/api/annotations',json=payload).json()
        assert c.delete('/api/annotations/delete-me?revision=0').status_code==409
        assert len(c.get('/api/annotations').json())==2
        assert c.delete('/api/annotations/delete-me?revision=1').json()['recoverable']
        assert c.get('/api/annotations').json()==[other]
        assert json.loads((tmp_path/'delete-test/deleted/delete-me.json').read_text())==original
        assert c.get('/api/deleted-annotations').json()==['delete-me']
        assert c.get('/api/cases').json()[0]['completed']==1
        for kind in ['evaluation','inference']:
            rows=(tmp_path/f'delete-test/exports/{kind}.jsonl').read_text().splitlines()
            assert len(rows)==1 and 'keep-me' in rows[0] and 'delete-me' not in rows[0]
        payload.update(id='delete-me',revision=0)
        assert c.post('/api/annotations',json=payload).status_code==409
        assert c.delete('/api/annotations/delete-me?revision=1').status_code==404
        assert c.delete('/api/annotations/bad.id?revision=1').status_code==422
    with TestClient(create_app(tmp_path/'data',tmp_path/'delete-test','prefix_generation')) as c:
        assert c.get('/api/annotations').json()==[other]
        assert c.get('/api/deleted-annotations').json()==['delete-me']
