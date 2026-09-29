import json
from pathlib import Path
import sys

import numpy as np
import pytest

from evaluation.case_definition.build_evaluation_cases import layout


@pytest.mark.parametrize(
    "p,t,end",
    [
        (1668, 11171, 11455),
        (2504, 3014, 3236),
        (1110, 8653, 8851),
        (4619, 9614, 9853),
        (10704, 14389, 14629),
        (0, 33, 65),
        (0, 1, 3),
    ],
)
def test_observed_prefix_and_generated_interval_never_shift(p, t, end):
    plan = layout(
        dict(prefix_start=p, t=t, action_end=t + 1, eval_start=t + 1, eval_end=end)
    )
    mapping = np.array(plan["model_source_frame_indices"])
    anchor = plan["model_anchor_frame"]
    assert anchor % 32 == 0 and (len(mapping) - 1) % 32 == 0
    assert np.all((mapping[: anchor + 1] >= p) & (mapping[: anchor + 1] < t))
    assert mapping[anchor] == t - 1
    a, b = plan["prediction_future_slice"]
    assert np.array_equal(mapping[anchor + a : anchor + b], np.arange(t, end))
    a, b = plan["prediction_evaluation_slice"]
    assert np.array_equal(mapping[anchor + a : anchor + b], np.arange(t + 1, end))
    assert np.all(mapping[anchor + 1 + end - t :] == end - 1)


def test_explicit_inference_start_keeps_history_index_offsets():
    import os
    if not os.environ.get("MEMWORLD_ROOT"):
        pytest.skip("Set MEMWORLD_ROOT for the optional MemWorld integration test")
    sys.path.insert(0, os.environ["MEMWORLD_ROOT"])
    from script.infer.infer_sample_pt_eval import limit_episode_clip_entries
    entries = [(i, {}, Path(f"{i}.pt")) for i in range(306)]
    future, window = limit_episode_clip_entries(entries, None, 9, start_clip=297)
    assert [r[0] for r in future] == list(range(297, 306))
    assert window["start_entry_index"] == 297
    assert window["source_clip_count"] == 306
    assert [r[0] for r in entries[: window["start_entry_index"] + 1]] == list(
        range(298)
    )
    # Missing exact boundaries must not silently start from a neighbouring clip.
    with pytest.raises(ValueError):
        limit_episode_clip_entries(entries[:296], None, 9, start_clip=297)
    with pytest.raises(ValueError):
        limit_episode_clip_entries(
            entries, None, 9, window_position="middle", start_clip=297
        )


def test_exported_prediction_excludes_anchor_tail_and_action(tmp_path):
    import av
    import imageio.v2 as imageio
    from evaluation.pipeline.infer_evaluation_cases import (
        trim_prediction,
    )

    source = tmp_path / "raw.mp4"
    with imageio.get_writer(
        str(source), fps=30, codec="libx264", quality=9, macro_block_size=1
    ) as writer:
        for i in range(9):
            writer.append_data(np.full((48, 64, 3), i * 25, dtype=np.uint8))
    counts = trim_prediction(
        source,
        tmp_path,
        {"prediction_future_slice": [1, 7], "prediction_evaluation_slice": [4, 7]},
    )
    assert counts == {"future.mp4": 6, "evaluation.mp4": 3}
    for name, expected in [
        ("future.mp4", [25, 50, 75, 100, 125, 150]),
        ("evaluation.mp4", [100, 125, 150]),
    ]:
        with av.open(str(tmp_path / name)) as video:
            observed = [
                frame.to_ndarray(format="rgb24").mean()
                for frame in video.decode(video=0)
            ]
        assert observed == pytest.approx(expected, abs=3)
