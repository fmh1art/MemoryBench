"""Validate actual released metadata and required evaluation assets."""
import argparse
import json
from pathlib import Path
import jsonschema


def validate(root):
    schema = json.loads(Path(__file__).with_name('eval_case.schema.json').read_text())
    cases = sorted(root.rglob('case.json'))
    if not cases:
        raise ValueError('No case.json records found')
    for path in cases:
        case = json.loads(path.read_text())
        jsonschema.validate(case, schema)
        assert path.parent.name == case['evaluation_case_id'], path
        marks = case['marks']
        start, end = marks['prefix_start'], marks.get('prefix_end', marks['t'])
        assert 0 <= start < end <= marks['t'] < marks['eval_end'], path
        assert case['prefix_frame_count'] == end - start, path
        assert case['generation_frame_count'] == marks['eval_end'] - marks['t'], path
        for name, count in [('prediction_future_slice', 'generation_frame_count'), ('prediction_evaluation_slice', 'evaluation_frame_count')]:
            a, b = case[name]
            assert 1 <= a < b and b - a == case[count], (path, name)
        a, b = case['prediction_future_slice']
        c, d = case['prediction_evaluation_slice']
        assert a <= c < d <= b, path
        assert case['input_contract'].get('future_gt_rgb_is_model_condition') is False, path
        for asset in ('annotation.json', 'memory_annotation.json', 'input', 'ground_truth/future.mp4', 'ground_truth/evaluation.mp4', 'packed/index.json', 'frame_mapping.npz'):
            assert (path.parent / asset).exists(), (path, asset)
    return len(cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(f'Validated {validate(args.root)} cases')
