"""Read-only Lightwheel source catalog for the existing range annotation UI.

Training tensors are never decoded or modified. MCAP RGB is prepared lazily on CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import av
from fastapi import HTTPException

from .server import Catalog, atomic_json, create_app

DEFAULT_ROOT = Path('/mnt/bos/tm9b1x/v05-training-ready/egoquest-lightwheel-fps2-20261001-v1/data')
DEFAULT_OUTPUT = Path('/mnt/bos/tm9b1x/fmh/data/MemoryBenchAnnotationsV2/lightwheel')
FAMILIES = ('lightwheel-egopro', 'lightwheel-egostandard')
TOPIC = '/sensor/camera/head_left/video'


def message_class(schema):
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    if schema.encoding != 'protobuf':
        raise ValueError(f'Unsupported MCAP encoding: {schema.encoding}')
    descriptors = descriptor_pb2.FileDescriptorSet()
    descriptors.ParseFromString(schema.data)
    pool = descriptor_pool.DescriptorPool()
    pending = list(descriptors.file)
    while pending:
        remaining = []
        for descriptor in pending:
            try:
                pool.Add(descriptor)
            except Exception:
                remaining.append(descriptor)
        if len(remaining) == len(pending):
            raise ValueError('Unresolved MCAP protobuf schema dependencies')
        pending = remaining
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(schema.name))


class LightwheelCatalog(Catalog):
    source_kind = 'lightwheel'
    def __init__(self, data_root, output_root, media_root=None, limit=100, recording=None):
        self.root = Path(data_root).resolve()
        self.cache = Path(output_root).resolve() / 'media_cache'
        self.media_root = Path(media_root).resolve() if media_root else None
        self.cases, self.views = {}, {}
        self._prepare_lock = threading.Lock()
        for family in FAMILIES:
            root = self.root / family
            if not root.is_dir():
                continue
            count = 0
            # Do not stat every entry or read thousands of COMPLETE files at startup.
            with os.scandir(root) as entries:
                for entry in entries:
                    if recording and entry.name != recording:
                        continue
                    if not entry.is_dir():
                        continue
                    case_id = f'{family}__{entry.name}'
                    self.cases[case_id] = dict(id=case_id, views=[dict(id='left', name='head_left', width=640, height=480)])
                    self.views[(case_id, 'left')] = dict(case_dir=Path(entry.path))
                    count += 1
                    if limit and count >= limit:
                        break
        self.cases = dict(sorted(self.cases.items()))

    def discover(self, family, recording):
        """Open a known UUID outside the initial bounded browsing list."""
        import re
        if family not in FAMILIES or not re.fullmatch(r'[a-fA-F0-9-]{36}', recording):
            raise HTTPException(422, '请选择数据源并填写完整 recording UUID')
        case_id = f'{family}__{recording}'
        path = self.root / family / recording
        if not (path / 'COMPLETE.json').is_file():
            raise HTTPException(404, '该数据源中没有找到这个 recording')
        self.cases[case_id] = dict(id=case_id, views=[dict(id='left', name='head_left', width=640, height=480)])
        self.views.setdefault((case_id, 'left'), dict(case_dir=path))
        return self.cases[case_id]

    def locate(self, complete, recording):
        original = Path(complete['source']['source_path'])
        candidates = []
        if self.media_root:
            # Accept a direct mounted Lightwheel root, task tree, or UUID video folder.
            family = complete['family']
            candidates.extend([
                self.media_root / family / family / 'mcap' / original.parent.name / original.name,
                self.media_root / family / 'EgoProStandard' / 'mcap' / original.parent.name / original.name,
                self.media_root / family / 'mcap' / original.parent.name / original.name,
                self.media_root / original.parent.name / original.name,
                self.media_root / original.name,
                self.media_root / family / f'{recording}.mp4',
                self.media_root / f'{recording}.mp4',
            ])
        candidates.append(original)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise HTTPException(422, '找不到原始 RGB。当前目录仅包含训练 latent；请通过 --media-root 指定原始 MCAP / MP4 目录。原始记录路径：' + str(original))

    def prepare(self, case_id, view_id):
        row = super().view(case_id, view_id)
        with self._prepare_lock:
            if 'path' in row:
                return row
            complete_path = row['case_dir'] / 'COMPLETE.json'
            complete = json.loads(complete_path.read_text())
            recording = row['case_dir'].name
            source = self.locate(complete, recording)
            if source.suffix.lower() == '.mp4':
                video = source
                sidecar = source.with_suffix('.timestamps.json')
                if not sidecar.is_file():
                    raise HTTPException(422, f'MP4 需要同名 .timestamps.json，包含 sensor_timestamp_ns 数组与 source_time_origin_ns：{sidecar}')
                timing = json.loads(sidecar.read_text())
            else:
                video, timing = self.extract(source, complete, case_id)
            row.update(path=video, timing=timing, complete=complete, original_source=source)
            return row

    def view(self, case_id, view_id):
        return self.prepare(case_id, view_id)

    def extract(self, source, complete, case_id):
        from mcap.reader import make_reader
        fingerprint = hashlib.sha256(f'{source}:{source.stat().st_size}:{source.stat().st_mtime_ns}'.encode()).hexdigest()[:16]
        folder = self.cache / case_id / fingerprint
        video, mapping = folder / 'left.mp4', folder / 'timestamps.json'
        if video.is_file() and mapping.is_file():
            return video, json.loads(mapping.read_text())
        folder.mkdir(parents=True, exist_ok=True)
        raw_path = folder / 'left.h264'
        timestamps, classes = [], {}
        try:
            with source.open('rb') as handle, raw_path.open('wb') as raw:
                for schema, channel, message in make_reader(handle).iter_messages(topics=[TOPIC]):
                    if schema is None:
                        raise ValueError('RGB topic has no protobuf schema')
                    if schema.id not in classes:
                        classes[schema.id] = message_class(schema)
                    payload = classes[schema.id]()
                    payload.ParseFromString(message.data)
                    if payload.format.lower() != 'h264':
                        raise ValueError('Only native H264 camera messages are supported')
                    raw.write(payload.data)
                    timestamps.append(int(message.log_time))
            if len(timestamps) < 2 or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
                raise ValueError('RGB timestamps are missing or non-increasing')
            devices = complete['source'].get('source_metadata', {}).get('devices', [])
            fps = next((d['fps'] for d in devices if d['device_type'] == 'head_left_camera'), '30')
            temp = folder / 'left.tmp.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-r', str(fps), '-i', str(raw_path),
                            '-an', '-vf', 'scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2',
                            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-threads', '2',
                            '-fps_mode', 'passthrough', '-movflags', '+faststart', str(temp)],
                           check=True, capture_output=True)
            with av.open(str(temp)) as container:
                count = sum(1 for _ in container.decode(video=0))
            if count != len(timestamps):
                raise ValueError(f'RGB message/frame mismatch: {len(timestamps)} / {count}; refusing guessed mapping')
            origin = int(complete['source'].get('source_time_origin_ns', timestamps[0]))
            timing = dict(sensor_timestamp_ns=[t-origin for t in timestamps], source_time_origin_ns=origin)
            os.replace(temp, video)
            atomic_json(mapping, timing)
            return video, timing
        except (ValueError, subprocess.CalledProcessError) as exc:
            raise HTTPException(422, f'RGB 提取失败：{exc}') from exc
        finally:
            raw_path.unlink(missing_ok=True)

    @lru_cache(maxsize=32)
    def metadata(self, case_id, view_id):
        row = self.prepare(case_id, view_id)
        with av.open(str(row['path'])) as container:
            stream = container.streams.video[0]
            tb = stream.time_base
            pts = [frame.pts for frame in container.decode(stream)]
            width, height = stream.width, stream.height
            duration = float(stream.duration * tb) if stream.duration else None
        sensor = row['timing']['sensor_timestamp_ns']
        if len(pts) < 2 or len(pts) != len(sensor) or any(b <= a for a,b in zip(pts,pts[1:])) or any(b <= a for a,b in zip(sensor,sensor[1:])):
            raise HTTPException(422, '视频帧与传感器时间戳不匹配，拒绝猜测对齐')
        media = [float(p * tb) for p in pts]
        end = max(duration or 0, media[-1] + media[-1]-media[-2])
        return dict(case_id=case_id, view_id=view_id, file_id=case_id, stream_id='head_left',
                    camera_frame_id='head_left_camera', pixel_domain_id='lightwheel_head_left',
                    relative_path=str(row['path']), video_path=str(row['path']), case_path=str(row['case_dir']),
                    checksum=None, width=width, height=height, frame_count=len(pts),
                    time_base=[tb.numerator,tb.denominator], media_pts=pts, media_seconds=media,
                    sensor_timestamp_ns=sensor, media_end_seconds=end, media_duration_seconds=end-media[0],
                    sensor_duration_seconds=(sensor[-1]-sensor[0]+sensor[-1]-sensor[-2])/1e9,
                    sensor_end_timestamp_ns_estimate=sensor[-1]+sensor[-1]-sensor[-2],
                    interval_convention='[start_frame, end_frame_exclusive)',
                    lightwheel_source=dict(training_root=str(row['case_dir']), original_source=str(row['original_source']),
                                           source_time_origin_ns=row['timing']['source_time_origin_ns'],
                                           task=row['complete']['source'].get('source_metadata',{}).get('task_name')))


def build_app(data_root=DEFAULT_ROOT, output_root=DEFAULT_OUTPUT, media_root=None, limit=100, recording=None):
    catalog = LightwheelCatalog(data_root, output_root, media_root, limit, recording)
    app = create_app(output_root=output_root, annotation_mode='prefix_generation', catalog=catalog)

    @app.post('/api/lightwheel/recordings/{family}/{recording}')
    def discover(family: str, recording: str):
        return catalog.discover(family, recording)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--media-root', type=Path)
    parser.add_argument('--limit', type=int, default=100, help='Initial recordings per family; 0 = all (slow)')
    parser.add_argument('--recording', help='Open a specific recording UUID')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8892)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(build_app(args.data_root,args.output_root,args.media_root,args.limit,args.recording), host=args.host,port=args.port,timeout_graceful_shutdown=5)


if __name__ == '__main__':
    main()
