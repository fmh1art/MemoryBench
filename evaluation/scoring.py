"""Normalize scored metrics and average applicable dimensions with equal weights."""

from statistics import mean

from .inventory.scoring import score_inventory

MEMORY_DIMENSIONS = ("inventory", "appearance", "spatial", "environment")


def score_memory(judgments: dict) -> dict:
    dimensions = {"inventory": score_inventory(judgments["inventory"])}
    metrics = {}
    incomplete = dimensions["inventory"]["status"] != "scored"
    for dimension in MEMORY_DIMENSIONS[1:]:
        values = []
        statuses = []
        for name, metric in judgments[dimension]:
            normalized = metric.score / 5 if metric.score is not None else None
            metrics[f"{dimension}.{name}"] = {
                "status": metric.status,
                "score": metric.score,
                "normalized": normalized,
            }
            statuses.append(metric.status)
            if normalized is not None:
                values.append(normalized)
        if "insufficient_evidence" in statuses:
            incomplete = True
            status, normalized = "insufficient_evidence", None
        elif values:
            status, normalized = "scored", mean(values)
        else:
            status, normalized = "not_applicable", None
        dimensions[dimension] = {"status": status, "normalized": normalized}
    applicable = [
        item["normalized"] for item in dimensions.values() if item["status"] == "scored"
    ]
    return {
        "status": "insufficient_evidence" if incomplete else "scored",
        "memory_score": None if incomplete else mean(applicable),
        "dimensions": dimensions,
        "metrics": metrics,
    }


def summarize(results: list[dict]) -> dict:
    valid_gates = [item["gate"] for item in results if item.get("gate") is not None]
    gate_passes = sum(
        gate["observability"]["label"] == 1 and gate["action_following"]["label"] == 1
        for gate in valid_gates
    )
    scored = [item for item in results if item["status"] == "scored"]
    dimensions = {}
    for dimension in MEMORY_DIMENSIONS:
        values = [
            item["scores"]["dimensions"][dimension]["normalized"]
            for item in scored
            if item["scores"]["dimensions"][dimension]["status"] == "scored"
        ]
        dimensions[dimension] = {
            "mean": mean(values) if values else None,
            "n": len(values),
        }
    # Per-metric summaries use complete scored cases, matching the headline denominator.
    metric_names = sorted(
        {name for item in scored for name in item["scores"]["metrics"]}
    )
    metrics = {}
    for name in metric_names:
        values = [
            item["scores"]["metrics"][name]["normalized"]
            for item in scored
            if item["scores"]["metrics"][name]["status"] == "scored"
        ]
        metrics[name] = {"mean": mean(values) if values else None, "n": len(values)}
    inventory = {}
    for name in ("object_presence", "object_count", "precision", "recall", "f1"):
        values = [item["scores"]["dimensions"]["inventory"][name] for item in scored]
        inventory[name] = {"mean": mean(values) if values else None, "n": len(values)}
    for name in ("missing_objects", "spurious_objects"):
        inventory[name] = {
            "total": sum(
                item["scores"]["dimensions"]["inventory"][name] for item in scored
            ),
            "n": len(scored),
        }
    return {
        "total_cases": len(results),
        "gate_evaluated": len(valid_gates),
        "gate_passed": gate_passes,
        "gate_pass_rate": gate_passes / len(valid_gates) if valid_gates else None,
        "observability_pass_rate": (
            mean(g["observability"]["label"] for g in valid_gates)
            if valid_gates
            else None
        ),
        "action_following_pass_rate": (
            mean(g["action_following"]["label"] for g in valid_gates)
            if valid_gates
            else None
        ),
        "scored_cases": len(scored),
        "gate_failed_cases": sum(item["status"] == "gate_failed" for item in results),
        "insufficient_evidence_cases": sum(
            item["status"] == "insufficient_evidence" for item in results
        ),
        "error_cases": sum(item["status"] == "error" for item in results),
        "memory_score": mean(item["scores"]["memory_score"] for item in scored)
        if scored
        else None,
        "dimensions": dimensions,
        "metrics": metrics,
        "inventory": inventory,
    }
