"""Instance recall and total-count agreement; missing/spurious rates are diagnostics."""

from .schema import Judgment

DIAGNOSTICS = (
    "gt_objects",
    "gen_objects",
    "true_positive",
    "missing_objects",
    "spurious_objects",
    "missing_rate",
    "spurious_rate",
    "precision",
    "recall",
    "f1",
)


def matched_objects(judgment: Judgment) -> list[dict]:
    """Give Appearance only semantically matched, reliably counted instances."""
    return [
        {**obj.model_dump(), "matched_count": min(obj.gt_count, obj.gen_count)}
        for obj in judgment.objects
        if obj.gt_count is not None
        and obj.gen_count is not None
        and obj.gt_count > 0
        and obj.gen_count > 0
    ]


def score_inventory(judgment: Judgment) -> dict:
    unresolved = any(
        obj.gt_count is None or obj.gen_count is None for obj in judgment.objects
    )
    if not judgment.region_assessable or unresolved:
        return {
            "status": "insufficient_evidence",
            "score": None,
            "normalized": None,
            **dict.fromkeys(("object_presence", "object_count", *DIAGNOSTICS)),
        }
    gt = sum(obj.gt_count for obj in judgment.objects)
    gen = sum(obj.gen_count for obj in judgment.objects)
    tp = sum(min(obj.gt_count, obj.gen_count) for obj in judgment.objects)
    fn, fp = gt - tp, gen - tp
    presence = tp / gt if gt else None
    count = 1 - abs(gen - gt) / max(gen, gt, 1)
    normalized = 0.7 * presence + 0.3 * count if presence is not None else count
    return {
        "status": "scored",
        "score": 5 * normalized,
        "normalized": normalized,
        "object_presence": presence,
        "object_count": count,
        "gt_objects": gt,
        "gen_objects": gen,
        "true_positive": tp,
        "missing_objects": fn,
        "spurious_objects": fp,
        "missing_rate": fn / gt if gt else None,
        "spurious_rate": fp / max(gen, 1),
        "precision": tp / gen if gen else float(gt == 0),
        "recall": presence,
        "f1": 2 * tp / (gt + gen) if gt + gen else 1.0,
    }
