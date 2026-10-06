"""Normalize scored metrics and average applicable dimensions with equal weights."""

from statistics import mean

from .inventory.scoring import score_inventory

MEMORY_DIMENSIONS = ("inventory", "appearance", "spatial", "environment")


DEFAULT_DIMENSIONS = ("inventory", "appearance", "spatial")
DIMENSION_METRICS = {
    "inventory": ("object_presence", "object_count", "f1"),
    "appearance": ("visual_appearance", "geometry_shape"),
    "spatial": ("object_localization", "relative_spatial_relation"),
    "environment": ("scene_structure", "background_appearance", "lighting_style"),
}
METRIC_NAMES = tuple(
    f"{dim}.{name}" for dim, names in DIMENSION_METRICS.items() for name in names
)


def select_metrics(dimensions=None, metrics=None):
    if metrics is not None:
        selected = tuple(dict.fromkeys(metrics))
    else:
        selected = tuple(
            f"{dim}.{name}"
            for dim in (dimensions or DEFAULT_DIMENSIONS)
            for name in DIMENSION_METRICS[dim]
        )
    if not selected or not set(selected) <= set(METRIC_NAMES):
        raise ValueError("Select at least one known memory metric")
    return selected


def score_memory(judgments: dict, selected_metrics=None) -> dict:
    selected = set(selected_metrics or select_metrics(tuple(judgments)))
    dimensions, metrics = {}, {}
    for dimension, judgment in judgments.items():
        if dimension == "inventory":
            item = score_inventory(judgment)
            values = [
                item[name]
                for name in DIMENSION_METRICS[dimension]
                if f"{dimension}.{name}" in selected
            ]
            item["normalized"] = mean(values) if item["status"] == "scored" else None
            item["score"] = (
                5 * item["normalized"] if item["normalized"] is not None else None
            )
            for name in DIMENSION_METRICS[dimension]:
                if f"{dimension}.{name}" in selected:
                    metrics[f"{dimension}.{name}"] = {
                        "status": item["status"],
                        "score": None if item[name] is None else 5 * item[name],
                        "normalized": item[name],
                    }
        else:
            records = []
            for name, metric in judgment:
                if f"{dimension}.{name}" not in selected:
                    continue
                normalized = metric.score / 5 if metric.score is not None else None
                record = {
                    "status": metric.status,
                    "score": metric.score,
                    "normalized": normalized,
                }
                metrics[f"{dimension}.{name}"] = record
                records.append(record)
            item = average_records(records)
        dimensions[dimension] = item
    applicable = [
        item["normalized"] for item in dimensions.values() if item["status"] == "scored"
    ]
    status = (
        "scored"
        if applicable
        else (
            "insufficient_evidence"
            if any(
                item["status"] == "insufficient_evidence"
                for item in dimensions.values()
            )
            else "not_applicable"
        )
    )
    return {
        "status": status,
        "memory_score": mean(applicable) if status == "scored" else None,
        "dimensions": dimensions,
        "metrics": metrics,
        "scored_dimensions": len(applicable),
        "total_dimensions": len(dimensions),
    }


def average_records(records):
    statuses = [item["status"] for item in records]
    scored = [item for item in records if item["status"] == "scored"]
    if scored:
        status, normalized = "scored", mean(item["normalized"] for item in scored)
    else:
        status = (
            "insufficient_evidence"
            if "insufficient_evidence" in statuses
            else "not_applicable"
        )
        normalized = None
    return {
        "status": status,
        "normalized": normalized,
        "score": 5 * normalized if normalized is not None else None,
        "scored_samples": len(scored),
        "total_samples": len(records),
    }


def average_memory_frames(frames):
    """Average reliable scores only; retain null results and explicit denominators."""
    if not frames or any(frame["status"] == "error" for frame in frames):
        raise ValueError("A nonempty set of successful frame evaluations is required")
    dimensions, metrics = {}, {}
    for group, destination in [("dimensions", dimensions), ("metrics", metrics)]:
        names = frames[0]["scores"][group]
        for name in names:
            records = [frame["scores"][group][name] for frame in frames]
            item = average_records(records)
            item["scored_frames"] = sum(
                record["status"] == "scored" for record in records
            )
            if group == "dimensions" and name == "inventory":
                for key in (
                    "object_presence",
                    "object_count",
                    "precision",
                    "recall",
                    "f1",
                    "missing_objects",
                    "spurious_objects",
                ):
                    values = [
                        record[key]
                        for record in records
                        if record["status"] == "scored" and record[key] is not None
                    ]
                    item[key] = mean(values) if values else None
            destination[name] = item
    scores = [
        frame["memory_score"]
        for frame in frames
        if frame["status"] == "scored" and frame["memory_score"] is not None
    ]
    status = (
        "scored"
        if scores
        else (
            "insufficient_evidence"
            if any(frame["status"] == "insufficient_evidence" for frame in frames)
            else "not_applicable"
        )
    )
    return {
        "status": status,
        "memory_score": mean(scores) if status == "scored" else None,
        "dimensions": dimensions,
        "metrics": metrics,
        "aggregation": "mean_of_reliable_frame_scores",
        "scored_frames": len(scores),
        "excluded_frames": len(frames) - len(scores),
        "frame_count": len(frames),
    }


def summarize(results: list[dict]) -> dict:
    valid_gates = [item["gate"] for item in results if item.get("gate") is not None]
    gate_passes = sum(
        gate["observability"]["label"] == 1 and gate["action_following"]["label"] == 1
        for gate in valid_gates
    )
    scored = [item for item in results if item["status"] == "scored"]
    gate_failed = [item for item in results if item["status"] == "gate_failed"]

    def case_mean(values, rejected):
        combined = values + [0.0] * rejected
        return {
            "mean": mean(combined) if combined else None,
            "n": len(combined),
            "gate_passed_mean": mean(values) if values else None,
            "gate_passed_n": len(values),
            "gate_failed_zero_n": rejected,
        }

    def enabled(item, name):
        selected = item.get("selected_metrics", select_metrics())
        return any(key == name or key.startswith(name + ".") for key in selected)

    dimensions = {}
    for dimension in MEMORY_DIMENSIONS:
        values = [
            item["scores"]["dimensions"][dimension]["normalized"]
            for item in scored
            if item["scores"]["dimensions"].get(dimension, {}).get("status") == "scored"
        ]
        dimensions[dimension] = case_mean(
            values, sum(enabled(item, dimension) for item in gate_failed)
        )
    # Gate failures contribute zero; other unreliable scores keep their null status.
    metric_names = sorted(
        {name for item in scored for name in item["scores"]["metrics"]}
        | {name for item in gate_failed for name in item.get("selected_metrics", ())}
    )
    metrics = {}
    for name in metric_names:
        values = [
            item["scores"]["metrics"][name]["normalized"]
            for item in scored
            if item["scores"]["metrics"].get(name, {}).get("status") == "scored"
        ]
        metrics[name] = case_mean(
            values, sum(enabled(item, name) for item in gate_failed)
        )
    inventory = {}
    for name in ("object_presence", "object_count", "precision", "recall", "f1"):
        values = [
            item["scores"]["dimensions"]["inventory"][name]
            for item in scored
            if item["scores"]["dimensions"].get("inventory", {}).get("status")
            == "scored"
        ]
        inventory[name] = case_mean(
            values, sum(enabled(item, "inventory") for item in gate_failed)
        )
    for name in ("missing_objects", "spurious_objects"):
        inventory[name] = {
            "total": sum(
                item["scores"]["dimensions"]["inventory"][name]
                for item in scored
                if item["scores"]["dimensions"].get("inventory", {}).get("status")
                == "scored"
            ),
            "n": sum(
                item["scores"]["dimensions"].get("inventory", {}).get("status")
                == "scored"
                for item in scored
            ),
            "unit": "sum of per-case mean counts across frames",
        }
    memory = case_mean(
        [item["scores"]["memory_score"] for item in scored], len(gate_failed)
    )
    return {
        "case_aggregation_policy": "gate_failed_zero_skip_unreliable",
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
        "not_applicable_cases": sum(
            item["status"] == "not_applicable" for item in results
        ),
        "insufficient_evidence_cases": sum(
            item["status"] == "insufficient_evidence" for item in results
        ),
        "error_cases": sum(item["status"] == "error" for item in results),
        "memory_score": memory["mean"],
        "memory_score_n": memory["n"],
        "gate_passed_memory_score": memory["gate_passed_mean"],
        "dimensions": dimensions,
        "metrics": metrics,
        "inventory": inventory,
    }
