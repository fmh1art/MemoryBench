"""Export completed Lightwheel ranges as RGB case drafts, outside the V1/V2 indexes."""
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from .server import atomic_json


def export_episode(row, output):
    if row['status'] != 'complete' or 'lightwheel' not in row['source']:
        raise ValueError('A completed Lightwheel annotation is required')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', row['id']):
        raise ValueError('Invalid annotation ID')
    marks = row['marks']
    p, pe, t, end = [marks[k] for k in ('prefix_start','prefix_end','t','eval_end')]
    if not (0 <= p < pe <= t < end <= row['source']['frame_count']):
        raise ValueError('MemoryBench export requires ordered, non-overlapping prefix/future ranges')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / ('lightwheel_' + row['id'])
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    source = row['source']['video_path']
    with tempfile.TemporaryDirectory(prefix='.lightwheel-', dir=output) as temp:
        case = Path(temp)
        (case/'input').mkdir(); (case/'ground_truth').mkdir()
        for start, stop, path in [(p,pe,case/'input/rgb_prefix.mp4'),(t,end,case/'ground_truth/future.mp4')]:
            subprocess.run(['ffmpeg','-v','error','-i',source,'-an','-vf',
                            rf'select=gte(n\,{start})*lt(n\,{stop}),setpts=N/(30*TB)',
                            '-r','30','-c:v','libx264','-preset','fast','-crf','18','-threads','2',
                            '-movflags','+faststart',str(path)],check=True)
        subprocess.run(['ffmpeg','-v','error','-i',source,'-vf',rf'select=eq(n\,{pe-1})',
                        '-frames:v','1','-threads','1',str(case/'input/anchor.png')],check=True)
        atomic_json(case/'annotation.json',row)
        atomic_json(case/'case.json',dict(
            evaluation_case_id=destination.name,
            status='rgb_draft_requires_conditions_and_semantic_review',
            dataset_name=row['case_id'].split('__')[0],
            source=row['source'], source_marks=marks, fps=30,
            prefix_frame_count=pe-p,generation_frame_count=end-t,
            input_contract=dict(rgb_prefix='input/rgb_prefix.mp4',rgb_anchor='input/anchor.png',
                                future_gt_rgb_is_model_condition=False),
            ground_truth=dict(future='ground_truth/future.mp4'),
            pending=['hand/head-pose condition extraction and alignment','future_action.txt human review',
                     'memory objects and evaluation metadata','V2 manifest registration and validation']))
        case.rename(destination)
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',required=True,type=Path,help='Completed evaluation JSONL export')
    p.add_argument('--output-root',required=True,type=Path,help='Staging folder; do not use V1')
    args=p.parse_args()
    with args.annotations.open() as f:
        for line in f:
            if line.strip(): print(export_episode(json.loads(line),args.output_root))


if __name__=='__main__':
    main()
