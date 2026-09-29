"""Regression checks for disjoint source spans and existing five-boundary cases."""
import pytest
from evaluation.case_definition.build_evaluation_cases import layout
from evaluation.case_definition.assets import action_frame_count


def test_splice_skips_intermediate_reveal_and_anchors_last_observation():
    plan = layout(dict(prefix_start=5, prefix_end=38, t=90, eval_end=130))
    frames = plan['model_source_frame_indices']
    anchor = plan['model_anchor_frame']
    assert frames[anchor:anchor + 2] == [37, 90]
    assert not set(range(38, 90)).intersection(frames)
    assert plan['prefix_frame_count'] == 33
    assert plan['generation_frame_count'] == 40
    assert plan['prediction_evaluation_slice'] == [1, 41]
    assert anchor % 32 == 0 and (len(frames) - 1) % 32 == 0


def test_overlapping_future_is_rejected():
    with pytest.raises(ValueError, match='disjoint'):
        layout(dict(prefix_start=0, prefix_end=100, t=90, eval_end=130))


def test_legacy_case_layout_unchanged():
    plan = layout(dict(prefix_start=1110, t=8653, action_end=8833, eval_start=8833, eval_end=8851))
    assert plan['prefix_frame_count'] == 7543
    assert plan['inference_start_clip'] == 236
    assert plan['generation_clips'] == 7
    assert plan['prediction_evaluation_slice'] == [181, 199]


def test_full_generation_action_review_requires_explicit_policy():
    case = dict(marks=dict(t=90, action_end=None), generation_frame_count=40)
    with pytest.raises(ValueError):
        action_frame_count(case)
    case['evaluation_scope'] = 'full_annotated_generation'
    assert action_frame_count(case) == 40
