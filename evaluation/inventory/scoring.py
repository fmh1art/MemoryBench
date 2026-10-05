"""Counts come from semantic annotations; the VLM never calculates metric values."""

from .schema import Judgment


def score_inventory(judgment: Judgment) -> dict:
    unresolved = any(
        obj.gt_count is None or obj.gen_count is None for obj in judgment.objects
    )
    if not judgment.region_assessable or unresolved:
        return {
            "status": "insufficient_evidence",
            "score": None,
            "normalized": None,
            "object_presence": None,
            "object_count": None,
            "missing_objects": None,
            "spurious_objects": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    pairs = [(obj.gt_count, obj.gen_count) for obj in judgment.objects]
    gt_total = sum(gt for gt, gen in pairs)
    gen_total = sum(gen for gt, gen in pairs)
    matched = sum(min(gt, gen) for gt, gen in pairs)
    missing = gt_total - matched
    spurious = gen_total - matched
    reference_groups = [(gt, gen) for gt, gen in pairs if gt > 0]
    presence = (
        sum(gen > 0 for gt, gen in reference_groups) / len(reference_groups)
        if reference_groups
        else float(gen_total == 0)
    )
    count = (
        sum(1 - abs(gt - gen) / max(gt, gen) for gt, gen in pairs) / len(pairs)
        if pairs
        else 1.0
    )
    precision = matched / gen_total if gen_total else float(gt_total == 0)
    recall = matched / gt_total if gt_total else 1.0
    f1 = 2 * matched / (gt_total + gen_total) if gt_total + gen_total else 1.0
    normalized = (presence + count + f1) / 3
    return {
        "status": "scored",
        "score": 5 * normalized,
        "normalized": normalized,
        "object_presence": presence,
        "object_count": count,
        "missing_objects": missing,
        "spurious_objects": spurious,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
