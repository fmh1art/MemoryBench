"""Export a complete native rollout on the GT frame timeline, using timestamps only."""
import json
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def align(output):
    output = Path(output)
    spec = json.loads((output / 'input/adaptation.json').read_text())
    if not spec['duration_complete']:
        return
    case = Path(spec['case'])
    meta = json.loads((case / 'case.json').read_text())
    source = np.load(case / 'input/aligned_conditions.npz')
    indices = source['source_frame_indices']
    selected = (indices >= meta['marks']['t']) & (indices < meta['marks']['eval_end'])
    target_times = source['sensor_timestamp_ns'][selected]
    native_times = np.asarray(spec['future_timestamp_ns'], dtype=np.int64)
    assert len(target_times) == meta['generation_frame_count']
    assert len(native_times) == spec['prediction_frames']
    assert np.all(np.diff(native_times) > 0)
    assert target_times[0] >= native_times[0]
    assert target_times[-1] <= native_times[-1] + 1e9 / spec['fps']
    native = output / 'prediction_native.mp4'
    if not native.exists():
        (output / 'prediction.mp4').rename(native)
    with av.open(str(native)) as container:
        frames = list(container.decode(video=0))
    if len(frames) != len(native_times):
        raise ValueError(f'Incomplete native rollout: {len(frames)} frames, expected {len(native_times)}; refusing to pad a short prediction')
    right = np.searchsorted(native_times, target_times).clip(0, len(native_times)-1)
    left = (right-1).clip(0)
    mapping = np.where(abs(native_times[left]-target_times) <= abs(native_times[right]-target_times), left, right)
    fps = Fraction(str(meta['fps']))
    temp = output / 'prediction.aligned.tmp.mp4'
    with av.open(str(temp), 'w') as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height = frames[0].width, frames[0].height
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast'}
        for index in mapping:
            frame = av.VideoFrame.from_ndarray(frames[index].to_ndarray(format='rgb24'), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(temp)) as container:
        actual_fps = container.streams.video[0].average_rate
        count = sum(1 for _ in container.decode(video=0))
    # GT is inspected for duration/count validation only, never read into the model.
    with av.open(str(case / 'ground_truth/future.mp4')) as container:
        gt_fps = container.streams.video[0].average_rate
        gt_count = sum(1 for _ in container.decode(video=0))
    assert count == gt_count == len(target_times), (count, gt_count, len(target_times))
    assert actual_fps == gt_fps == fps, (actual_fps, gt_fps, fps)
    temp.replace(output / 'prediction.mp4')
    report = dict(status='full_length_verified', model=spec['model'], case=str(case),
                  native_frames=len(frames), native_fps=spec['fps'], frames=count, fps=float(fps),
                  duration_seconds=count/float(fps), gt_frames=gt_count, gt_fps=float(gt_fps),
                  method='nearest generated sensor timestamp per GT source frame; no GT pixels used for generation',
                  output_to_native_frame=mapping.tolist(), action_semantics_verified=False)
    (output / 'output_alignment.json').write_text(json.dumps(report, indent=2))
    print(f'{spec["model"]}: verified {count} frames at {fps} fps, matching GT future', flush=True)


if __name__ == '__main__':
    align(sys.argv[1])
