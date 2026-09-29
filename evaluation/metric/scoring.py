"""Aggregate direct VLM pair judgments with configurable dimension weights."""

from statistics import mean
from .schemas import METRICS

LABELS = {
    "inventory": "物体清单",
    "appearance": "外观",
    "spatial": "空间关系",
}


def score_pair(judgment):
    """VLM supplies visual judgments; bookkeeping never depends on LLM arithmetic."""
    n, m = len(judgment.gt_objects), len(judgment.generated_objects)
    matches = {p.gt_id for p in judgment.verified_matches}
    scores = {k: getattr(judgment.dimensions, k).score for k in METRICS}
    reported = dict(scores)
    scores["inventory"] = (
        (1.0 if n + m == 0 else 2 * len(matches) / (n + m))
        if judgment.generated_readable
        else 0.0
    )
    scores["appearance"] = (
        sum(
            item.score if item.gt_id in matches and judgment.generated_readable else 0.0
            for item in judgment.appearance_items
        )
        / n
        if n
        else None
    )
    if not judgment.gt_spatial_applicable:
        scores["spatial"] = None
    if not judgment.generated_readable or (n and not matches):
        scores = {k: 0.0 if value is not None else None for k, value in scores.items()}
    adjustments = [
        {
            "metric": k,
            "vlm_reported": reported[k],
            "computed": scores[k],
            "reason": "Apply verified-identity constraints, GT applicability and exact counting/averaging",
        }
        for k in METRICS
        if reported[k] != scores[k]
    ]
    return scores, adjustments


def aggregate(frame_results, gate, weights=None):
    weights = weights or dict.fromkeys(METRICS, 1.0)
    if gate not in (0, 1):
        raise ValueError("Gate must be binary")
    if not gate:
        return {
            "gate": 0,
            "gate_pass_rate": 0.0,
            "metrics": dict.fromkeys(METRICS),
            "applicable_frame_counts": dict.fromkeys(METRICS, 0),
            "effective_weights": dict.fromkeys(METRICS, 0),
            "content_mean": None,
            "memory_score": 0.0,
            "content_evaluated": False,
        }
    if not frame_results:
        raise ValueError("Passing Gate requires image-pair judgments")
    values = {
        k: [r["scores"][k] for r in frame_results if r["scores"][k] is not None]
        for k in METRICS
    }
    scores = {k: mean(v) if v else None for k, v in values.items()}
    denominator = sum(weights[k] for k in METRICS if scores[k] is not None)
    if denominator <= 0:
        raise ValueError("No positive configured weight for any GT-applicable metric")
    effective = {
        k: weights[k] / denominator if scores[k] is not None else 0.0 for k in METRICS
    }
    total = sum(scores[k] * effective[k] for k in METRICS if scores[k] is not None)
    return {
        "gate": 1,
        "gate_pass_rate": 1.0,
        "metrics": scores,
        "applicable_frame_counts": {k: len(v) for k, v in values.items()},
        "effective_weights": effective,
        "content_mean": total,
        "memory_score": total,
        "content_evaluated": True,
        "scoring_policy": "mean direct pair score per GT-applicable dimension, then normalized configured weighted mean; Gate failure is zero",
    }


def metric_table(result):
    lines = [
        "| Metric | Score | 有效权重 |",
        "| --- | ---: | ---: |",
        f"| Gate | {result['gate']} | — |",
    ]
    for key in METRICS:
        value = result["metrics"][key]
        lines.append(
            f"| {LABELS[key]} | {'N/A' if value is None else f'{value:.4f}'} | {result['effective_weights'][key]:.3f} |"
        )
    lines.append(f"| Memory | {result['memory_score']:.4f} | — |")
    return "\n".join(lines) + "\n"
