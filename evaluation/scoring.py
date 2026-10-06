"""Gate eligibility, weighted memory dimensions, temporal means and geometric case scores."""

from math import exp, log
from statistics import mean

from .inventory.scoring import DIAGNOSTICS, score_inventory

MEMORY_DIMENSIONS = ("inventory", "appearance", "spatial", "environment")
DEFAULT_DIMENSIONS = ("inventory", "appearance", "spatial")
DIMENSION_METRICS = {
    "inventory": ("object_presence", "object_count"),
    "appearance": ("visual_appearance", "geometry_shape"),
    "spatial": ("object_localization", "relative_spatial_relation"),
    "environment": ("scene_structure", "background_appearance", "lighting_style"),
}
WEIGHTS = {
    "inventory.object_presence": 0.7,
    "inventory.object_count": 0.3,
    "appearance.visual_appearance": 0.5,
    "appearance.geometry_shape": 0.5,
    "spatial.object_localization": 0.4,
    "spatial.relative_spatial_relation": 0.6,
}
SCORING_VERSION = "memory.v3.weighted_geometric_matched_appearance"
METRIC_NAMES = tuple(
    f"{dim}.{name}" for dim, names in DIMENSION_METRICS.items() for name in names
)


def select_metrics(dimensions=None, metrics=None):
    selected = (
        tuple(dict.fromkeys(metrics))
        if metrics is not None
        else tuple(
            f"{dim}.{name}"
            for dim in (dimensions or DEFAULT_DIMENSIONS)
            for name in DIMENSION_METRICS[dim]
        )
    )
    if not selected or not set(selected) <= set(METRIC_NAMES):
        raise ValueError("Select at least one known memory metric")
    return selected


def average_records(records, weights=None):
    """Skip unreliable entries and renormalize the remaining metric weights."""
    weights = weights if weights is not None else [1.0] * len(records)
    scored = [
        (item, weight)
        for item, weight in zip(records, weights)
        if item["status"] == "scored"
    ]
    status = (
        "scored"
        if scored
        else (
            "insufficient_evidence"
            if any(item["status"] == "insufficient_evidence" for item in records)
            else "not_applicable"
        )
    )
    normalized = (
        sum(item["normalized"] * weight for item, weight in scored)
        / sum(weight for _, weight in scored)
        if scored
        else None
    )
    return {
        "status": status,
        "normalized": normalized,
        "score": 5 * normalized if normalized is not None else None,
        "scored_samples": len(scored),
        "total_samples": len(records),
    }


def memory_from_dimensions(dimensions):
    """Environment is diagnostic only; exact zeros remain zero without a log(0)."""
    core = [dimensions[name] for name in DEFAULT_DIMENSIONS if name in dimensions]
    values = [item["normalized"] for item in core if item["status"] == "scored"]
    status = (
        "scored"
        if values
        else (
            "insufficient_evidence"
            if any(item["status"] == "insufficient_evidence" for item in core)
            else "not_applicable"
        )
    )
    score = (
        (0.0 if 0.0 in values else exp(mean(log(value) for value in values)))
        if values
        else None
    )
    return {
        "status": status,
        "memory_score": score,
        "memory_score_100": 100 * score if score is not None else None,
        "scored_dimensions": len(values),
        "total_dimensions": len(core),
    }


def score_memory(judgments: dict, selected_metrics=None) -> dict:
    selected = set(selected_metrics or select_metrics(tuple(judgments)))
    dimensions, metrics = {}, {}
    for dimension, judgment in judgments.items():
        records, weights = [], []
        inventory = score_inventory(judgment) if dimension == "inventory" else None
        for name in DIMENSION_METRICS[dimension]:
            key = f"{dimension}.{name}"
            if key not in selected:
                continue
            if inventory is not None:
                normalized = inventory[name]
                status = (
                    inventory["status"]
                    if normalized is not None
                    else (
                        "not_applicable"
                        if inventory["status"] == "scored"
                        else inventory["status"]
                    )
                )
                record = {
                    "status": status,
                    "normalized": normalized,
                    "score": 5 * normalized if normalized is not None else None,
                }
            else:
                metric = getattr(judgment, name)
                record = {
                    "status": metric.status,
                    "score": metric.score,
                    "normalized": metric.score / 5
                    if metric.score is not None
                    else None,
                }
            metrics[key] = record
            records.append(record)
            weights.append(WEIGHTS.get(key, 1.0))
        if not records:
            continue
        item = average_records(records, weights)
        if inventory is not None:
            item.update(
                {
                    key: inventory[key]
                    for key in (*DIMENSION_METRICS["inventory"], *DIAGNOSTICS)
                }
            )
        dimensions[dimension] = item
    return {
        **memory_from_dimensions(dimensions),
        "dimensions": dimensions,
        "metrics": metrics,
        "scoring_version": SCORING_VERSION,
    }


def average_memory_frames(frames):
    """Average each dimension across reliable frames BEFORE the case geometric mean."""
    if not frames or any(frame["status"] == "error" for frame in frames):
        raise ValueError("A nonempty set of successful frame evaluations is required")
    dimensions, metrics = {}, {}
    for group, destination in [("dimensions", dimensions), ("metrics", metrics)]:
        for name in frames[0]["scores"][group]:
            records = [frame["scores"][group][name] for frame in frames]
            item = average_records(records)
            item["scored_frames"] = item["scored_samples"]
            if group == "dimensions" and name == "inventory":
                for key in (*DIMENSION_METRICS["inventory"], *DIAGNOSTICS):
                    values = [
                        record[key]
                        for record in records
                        if record["status"] == "scored" and record[key] is not None
                    ]
                    item[key] = mean(values) if values else None
            destination[name] = item
    valid_frames = sum(frame["status"] == "scored" for frame in frames)
    return {
        **memory_from_dimensions(dimensions),
        "dimensions": dimensions,
        "metrics": metrics,
        "aggregation": "geometric_mean_of_temporally_averaged_core_dimensions",
        "scoring_version": SCORING_VERSION,
        "scored_frames": valid_frames,
        "excluded_frames": len(frames) - valid_frames,
        "frame_count": len(frames),
    }


def summarize(results: list[dict]) -> dict:
    gates = [item["gate"] for item in results if item.get("gate") is not None]

    def passed(item):
        gate = item.get("gate")
        return (
            gate is not None
            and gate["observability"]["label"] == gate["action_following"]["label"] == 1
        )

    eligible = [item for item in results if passed(item)]
    assessed = [
        item for item in eligible if "scores" in item and item["status"] != "error"
    ]
    scored = [item for item in assessed if item["status"] == "scored"]

    def macro(values):
        return {"mean": mean(values) if values else None, "n": len(values)}

    dimensions = {}
    for name in MEMORY_DIMENSIONS:
        values = [
            item["scores"]["dimensions"][name]["normalized"]
            for item in assessed
            if item["scores"]["dimensions"].get(name, {}).get("status") == "scored"
        ]
        dimensions[name] = macro(values)
    metric_names = sorted(
        {name for item in assessed for name in item["scores"]["metrics"]}
    )
    metrics = {
        name: macro(
            [
                item["scores"]["metrics"][name]["normalized"]
                for item in assessed
                if item["scores"]["metrics"].get(name, {}).get("status") == "scored"
            ]
        )
        for name in metric_names
    }
    inventory_records = [
        item["scores"]["dimensions"]["inventory"]
        for item in assessed
        if item["scores"]["dimensions"].get("inventory", {}).get("status") == "scored"
    ]
    inventory = {
        key: macro([item[key] for item in inventory_records if item[key] is not None])
        for key in (*DIMENSION_METRICS["inventory"], *DIAGNOSTICS)
    }
    for key in ("missing_objects", "spurious_objects"):
        values = [item[key] for item in inventory_records if item[key] is not None]
        inventory[key].update(
            total=sum(values), unit="sum of per-case mean counts across frames"
        )
    memory = macro([item["scores"]["memory_score"] for item in scored])
    coverage = len(eligible) / len(results) if results else None
    return {
        "scoring_version": SCORING_VERSION,
        "case_aggregation_policy": "gate_valid_memory_only",
        "total_cases": len(results),
        "gate_evaluated": len(gates),
        "gate_passed": len(eligible),
        "gate_coverage": coverage,
        "gate_pass_rate": len(eligible) / len(gates) if gates else None,
        "observability_pass_rate": mean(g["observability"]["label"] for g in gates)
        if gates
        else None,
        "action_following_pass_rate": mean(
            g["action_following"]["label"] for g in gates
        )
        if gates
        else None,
        **{
            f"{status}_cases": sum(item["status"] == status for item in results)
            for status in (
                "scored",
                "gate_failed",
                "not_applicable",
                "insufficient_evidence",
                "error",
            )
        },
        "memory_score": memory["mean"],
        "memory_score_n": memory["n"],
        "memory_score_100": 100 * memory["mean"]
        if memory["mean"] is not None
        else None,
        "dimensions": dimensions,
        "metrics": metrics,
        "inventory": inventory,
        "report": {
            "gate_coverage_percent": 100 * coverage if coverage is not None else None,
            **{
                name: 100 * dimensions[name]["mean"]
                if dimensions[name]["mean"] is not None
                else None
                for name in DEFAULT_DIMENSIONS
            },
            "memory_score": 100 * memory["mean"]
            if memory["mean"] is not None
            else None,
        },
    }
