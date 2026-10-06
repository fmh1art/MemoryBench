"""Generated unreadability is zero; reference invisibility remains excluded."""

import copy

import pytest
from pydantic import ValidationError

from evaluation.inventory.schema import Judgment as Inventory
from evaluation.inventory.scoring import score_inventory
from evaluation.review_nulls import Review, apply_review, unresolved_metrics
from evaluation.scoring import score_memory
from evaluation.spatial.schema import Judgment as Spatial

E = [
    {
        "frame_indices": [1],
        "observation": "GT exposes a red ring; Gen exposed cup contains unreadable fused surfaces.",
    }
]


def inventory(failure=False):
    return Inventory.model_validate(
        {
            "region_assessable": False,
            "gt_reference_clear": failure,
            "gen_region_exposed": failure,
            "generated_quality_failure": failure,
            "objects": [
                {
                    "object_id": "ring",
                    "description": "red ring",
                    "gt_count": 1,
                    "gen_count": None,
                    "evidence": E,
                }
            ],
            "evidence": E,
        }
    )


def metric(score=None):
    return {
        "status": "insufficient_evidence" if score is None else "scored",
        "score": score,
        "evidence": E,
    }


def test_inventory_quality_failure_zero_does_not_fabricate_counts():
    annotation = inventory(True)
    result = score_inventory(annotation)
    assert result["status"] == "scored"
    assert (
        result["object_presence"] == result["object_count"] == result["normalized"] == 0
    )
    assert annotation.objects[0].gen_count is None
    assert (
        result["missing_objects"]
        is result["spurious_objects"]
        is result["missing_rate"]
        is None
    )
    assert score_memory({"inventory": annotation})["memory_score"] == 0


def test_gt_not_exposed_is_still_excluded():
    result = score_inventory(inventory())
    assert result["status"] == "insufficient_evidence"
    assert result["normalized"] is None


def test_failure_flag_requires_clear_gt_and_exposed_gen():
    data = inventory(True).model_dump()
    data["gen_region_exposed"] = False
    with pytest.raises(ValidationError):
        Inventory.model_validate(data)


def test_review_changes_only_null_and_excludes_unmatched_appearance():
    inv = inventory()
    spatial = Spatial.model_validate(
        {"object_localization": metric(), "relative_spatial_relation": metric(4)}
    )
    frame = {
        "judgments": {"inventory": inv.model_dump(), "spatial": spatial.model_dump()},
        "scores": score_memory({"inventory": inv, "spatial": spatial}),
    }
    keys = unresolved_metrics(frame)
    assert keys == ["inventory", "spatial.object_localization"]
    original = copy.deepcopy(frame)
    review = Review.model_validate(
        {
            "decisions": [
                {
                    "metric": key,
                    "decision": "zero" if key.startswith("spatial") else "keep_null",
                    "gt_reference_clear": True,
                    "gen_region_exposed": True,
                    "generated_quality_failure": True,
                    "evidence": E,
                }
                for key in keys
            ]
        }
    )
    apply_review(frame, review, keys)
    assert frame["judgments"]["inventory"] == original["judgments"]["inventory"]
    assert (
        frame["judgments"]["spatial"]["relative_spatial_relation"]
        == original["judgments"]["spatial"]["relative_spatial_relation"]
    )
    assert frame["judgments"]["spatial"]["object_localization"]["score"] == 0
    with pytest.raises(ValueError):
        apply_review(frame, review, keys)


def test_review_zero_requires_visual_failure_confirmation():
    with pytest.raises(ValidationError):
        Review.model_validate(
            {
                "decisions": [
                    {
                        "metric": "inventory",
                        "decision": "zero",
                        "gt_reference_clear": True,
                        "gen_region_exposed": False,
                        "generated_quality_failure": True,
                        "evidence": E,
                    }
                ]
            }
        )


def test_unknown_gt_counts_cannot_be_scored_as_generated_failure():
    data = inventory(True).model_dump()
    data["objects"][0]["gt_count"] = None
    with pytest.raises(ValidationError):
        Inventory.model_validate(data)
    data["generated_quality_failure"] = False
    annotation = Inventory.model_validate(data)
    frame = {
        "judgments": {"inventory": data},
        "scores": score_memory({"inventory": annotation}),
    }
    assert unresolved_metrics(frame) == []


def test_empty_gt_presence_remains_inapplicable_under_gen_failure():
    data = inventory(True).model_dump()
    data["objects"][0]["gt_count"] = 0
    result = score_memory({"inventory": Inventory.model_validate(data)})
    assert result["metrics"]["inventory.object_presence"]["status"] == "not_applicable"
    assert result["metrics"]["inventory.object_count"]["normalized"] == 0


def test_recompute_entire_saved_case_preserves_gate_and_original_files(tmp_path):
    import asyncio
    import json

    from evaluation.review_nulls import review_results
    from evaluation.scoring import average_memory_frames

    source = tmp_path / "source"
    (source / "sample").mkdir(parents=True)
    inv = inventory()
    spatial = Spatial.model_validate(
        {"object_localization": metric(4), "relative_spatial_relation": metric()}
    )
    frames = []
    for index in (1, 2):
        scores = score_memory({"inventory": inv, "spatial": spatial})
        frames.append(
            {
                "frame_index": index,
                "time_seconds": float(index),
                "status": scores["status"],
                "memory_score": scores["memory_score"],
                "judgments": {
                    "inventory": inv.model_dump(),
                    "spatial": spatial.model_dump(),
                },
                "scores": scores,
            }
        )
    result = {
        "case_id": "sample",
        "case": {"lighting_style": False},
        "gate": {"observability": {"label": 1}, "action_following": {"label": 1}},
        "status": "scored",
        "selected_metrics": [
            "inventory.object_presence",
            "inventory.object_count",
            "spatial.object_localization",
            "spatial.relative_spatial_relation",
        ],
        "memory_frames": frames,
        "memory_sampling": {"frames": [{"frame_index": 1, "time_seconds": 1.0}]},
        "scores": average_memory_frames(frames),
    }
    raw = json.dumps(result)
    (source / "sample/result.json").write_text(raw)
    (source / "run.json").write_text("{}")
    (source / "summary.json").write_text("{}")

    class Service:
        async def ask(self, dimension, context, images, folder, **kwargs):
            assert dimension == "null_review"
            assert len(images) == 2
            index = context["memory_frame"]["frame_index"]
            return Review.model_validate(
                {
                    "decisions": [
                        {
                            "metric": key,
                            "decision": "zero" if key == "inventory" else "keep_null",
                            "gt_reference_clear": True,
                            "gen_region_exposed": True,
                            "generated_quality_failure": key == "inventory",
                            "evidence": [
                                {
                                    "frame_indices": [index],
                                    "observation": E[0]["observation"],
                                }
                            ],
                        }
                        for key in context["requested_unresolved_metrics"]
                    ]
                }
            )

    output = tmp_path / "updated"
    asyncio.run(review_results(source, output, Service()))
    updated = json.loads((output / "sample/result.json").read_text())
    assert updated["gate"] == result["gate"]
    assert updated["memory_score"] == 0
    assert all(
        f["judgments"]["spatial"] == spatial.model_dump()
        for f in updated["memory_frames"]
    )
    assert all(
        f["judgments"]["inventory"]["objects"][0]["gen_count"] is None
        for f in updated["memory_frames"]
    )
    assert (source / "sample/result.json").read_text() == raw
    assert json.loads((output / "summary.json").read_text())["memory_score"] == 0
