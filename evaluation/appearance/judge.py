"""Condition Appearance on Inventory matches; derive null when no match exists."""

from pydantic import create_model

from ..inventory.scoring import (
    matched_objects,
    score_inventory,
    unmatched_appearance_reason,
)
from ..judge import write_json
from ..schema import Metric, Record


async def judge_appearance(judge, inventory, context, images, output):
    matching = matched_objects(inventory)
    context = {
        **context,
        "appearance_matching": {
            "objects": matching,
            "source": "inventory",
            "unmatched_targets_excluded": True,
        },
    }
    if matching:
        return await judge.ask("appearance", context, images, output)
    reliable = (
        score_inventory(inventory)["status"] == "scored"
        and not inventory.generated_quality_failure
    )
    status = "not_applicable" if reliable else "insufficient_evidence"
    null_reason = unmatched_appearance_reason(inventory)
    evidence = [
        {
            "frame_indices": [context["memory_frame"]["frame_index"]],
            "observation": "Inventory found no matched GT/Gen target instances; appearance is not evaluated."
            if reliable
            else "Inventory could not establish reliable matched targets; appearance is unresolved.",
        }
    ]
    fields = {
        key.split(".")[1]: (Metric, ...)
        for key in context["selected_metrics"]
        if key.startswith("appearance.")
    }
    schema = create_model("AppearanceJudgment", __base__=Record, **fields)
    judgment = schema.model_validate(
        {
            key: {
                "status": status,
                "score": None,
                "evidence": evidence,
                "null_reason": null_reason,
            }
            for key in fields
        }
    )
    write_json(
        output / "appearance.derived.json",
        {
            "origin": "inventory_matching",
            "source_record": str(output / "inventory.json"),
            "context": context,
            "judgment": judgment.model_dump(),
            "reason": "No matched objects; no VLM query needed",
        },
    )
    return judgment
