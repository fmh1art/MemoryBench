"""One A/B/Tie outcome per judgment; WROP §4.4 / Table 2 statistics."""

import argparse
import csv
import html
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from . import ANNOTATION_PROTOCOL, VERSION
from .manifest import load_study
from .store import Store, normalize_vote


def comparisons(records):
    rows = []
    for record in records:
        if record["state"] != "voted":
            continue
        outcome = normalize_vote(record["vote"])["choice"]
        a, b = sorted((record["model_a"], record["model_b"]))
        score = 0.5 if outcome == "tie" else float(record["model_a" if outcome == "A" else "model_b"] == a)
        rows.append({"a": a, "b": b, "score": score, "rater": record["rater"],
                     "case_id": record["case_id"], "task": record["task"]})
    return rows


def connected_components(rows):
    graph = defaultdict(set)
    for r in rows:
        graph[r["a"]].add(r["b"])
        graph[r["b"]].add(r["a"])
    components = []
    remaining = set(graph)
    while remaining:
        seen, pending = set(), [min(remaining)]
        while pending:
            item = pending.pop()
            if item in seen:
                continue
            seen.add(item)
            pending.extend(graph[item] - seen)
        remaining -= seen
        components.append(sorted(seen))
    return components


def fit_bt(rows, models):
    """Each model receives one virtual draw, spread across all its opponents.

    Every unordered pair gets 1/(M-1) virtual games, each contributing .5 wins
    to either side. This gives exactly one virtual game per model, not M-1.
    """
    n = len(models)
    if n < 2:
        return np.full(n, 1500.0)
    index = {name: i for i, name in enumerate(models)}
    wins = np.zeros((n, n), dtype=float)
    for row in rows:
        i, j = index[row["a"]], index[row["b"]]
        wins[i, j] += row["score"]
        wins[j, i] += 1 - row["score"]
    wins += (np.ones((n, n)) - np.eye(n)) * 0.5 / (n - 1)
    i, j = np.triu_indices(n, 1)
    games, successes = (wins + wins.T)[i, j], wins[i, j]

    def objective(strength):
        diff = strength[i] - strength[j]
        loss = np.sum(games * np.logaddexp(0, diff) - successes * diff)
        residual = games * expit(diff) - successes
        gradient = np.bincount(i, weights=residual, minlength=n) - np.bincount(j, weights=residual, minlength=n)
        return loss, gradient

    result = minimize(objective, np.zeros(n), jac=True, method="L-BFGS-B",
                      options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000})
    if not result.success:
        raise RuntimeError(f"Bradley–Terry fit did not converge: {result.message}")
    centered = result.x - result.x.mean()
    return 1500 + centered * 400 / math.log(10)


def leaderboard(rows, all_models, replicates=1000, seed=42):
    components = connected_components(rows)
    entries = []
    rng = np.random.default_rng(seed)
    for component_id, models in enumerate(components, 1):
        subset = [r for r in rows if r["a"] in models and r["b"] in models]
        ratings = fit_bt(subset, models)
        raters = sorted({r["rater"] for r in subset})
        clusters = {r: [v for v in subset if v["rater"] == r] for r in raters}
        draws = []
        if len(raters) >= 2:
            for _ in range(replicates):
                sample = [row for r in rng.choice(raters, size=len(raters), replace=True) for row in clusters[r]]
                draws.append(fit_bt(sample, models))
        draws = np.asarray(draws)
        top_probabilities = None
        if len(draws):
            tied = np.isclose(draws, draws.max(axis=1, keepdims=True), atol=1e-6, rtol=0)
            top_probabilities = (tied / tied.sum(axis=1, keepdims=True)).mean(axis=0)
        for i, model in enumerate(models):
            appearances = [r for r in subset if model in (r["a"], r["b"])]
            scores = [r["score"] if model == r["a"] else 1 - r["score"] for r in appearances]
            by_task = defaultdict(list)
            for row, score in zip(appearances, scores):
                by_task[row["case_id"]].append(score)
            rank = int(1 + sum(x > ratings[i] + 1e-6 for x in ratings))
            entries.append({
                "model": model, "component": component_id, "rank": rank if len(components) == 1 else None,
                "rank_in_component": rank, "elo": float(ratings[i]),
                "ci95": np.percentile(draws[:, i], [2.5, 97.5]).tolist() if len(draws) else None,
                "top1_probability_in_component": float(top_probabilities[i]) if top_probabilities is not None else None,
                "games": len(scores), "cases": len(by_task), "raters": len({r["rater"] for r in appearances}),
                "score_rate": float(np.mean(scores)),
            })
    active = {e["model"] for e in entries}
    return {"components": components, "global_ranking_available": len(components) == 1,
            "unrated_models": sorted(set(all_models) - active),
            "leaderboard": sorted(entries, key=lambda e: (e["component"], -e["elo"], e["model"]))}


def pair_statistics(records, models):
    rows = comparisons(records)
    result = []
    for a, b in itertools.combinations(sorted(models), 2):
        subset = [r for r in rows if (r["a"], r["b"]) == (a, b)]
        n = len(subset)
        wins = sum(r["score"] == 1 for r in subset)
        ties = sum(r["score"] == 0.5 for r in subset)
        losses = n - wins - ties
        result.append({
            "model_a": a, "model_b": b, "wins": wins, "ties": ties, "losses": losses,
            "games": n, "cases": len({r["case_id"] for r in subset}),
            "a_win_rate": wins / n if n else None, "tie_rate": ties / n if n else None,
            "a_loss_rate": losses / n if n else None,
            "a_score_rate": (wins + .5 * ties) / n if n else None,
        })
    return result


def summarize(records, study, replicates=1000, seed=42):
    groups = {}
    for group in ("Overall", "IID", "OOD"):
        selected = [r for r in records if group == "Overall" or r["split"] == group]
        voted = [r for r in selected if r["state"] == "voted"]
        rows = comparisons(selected)
        non_ties = [r for r in voted if r["vote"]["choice"] != "tie"]
        groups[group] = leaderboard(rows, study.models, replicates, seed) | {
            "completed_judgments": len(voted), "media_issues": sum(r["state"] == "issue" for r in selected),
            "pairs": pair_statistics(selected, study.models),
            "position_a_preference_rate": sum(r["vote"]["choice"] == "A" for r in non_ties) / len(non_ties) if non_ties else None,
            "position_non_tie_n": len(non_ties),
        }
    counts = Counter(r["task"] for r in records if r["state"] == "voted")
    selected_tasks = study.provenance.get("selected_task_ids")
    expected = len(selected_tasks) if selected_tasks is not None else sum(
        len(case.predictions) - 1 if study.baseline and study.baseline in case.predictions
        else 0 if study.baseline else math.comb(len(case.predictions), 2) for case in study.cases)
    return {
        "version": VERSION, "title": study.title,
        "protocol": {
            "id": ANNOTATION_PROTOCOL,
            "source": "Training Object Permanence in World Models, §4.4 and Table 2",
            "url": "https://arxiv.org/html/2609.28654v1#S4.SS4",
            "annotation": "One direct A/B/Tie " + ("model" if study.provenance.get("evaluator") else "human") + " choice, jointly considering text alignment, natural motion/physical plausibility and object permanence",
            **({"evaluator": study.provenance["evaluator"],
                "input_protocol": study.provenance.get("input_protocol")} if study.provenance.get("evaluator") else {}),
            "fit": "Bradley–Terry maximum likelihood; ties .5/.5; one virtual draw per model distributed evenly over opponents",
            "elo": "1500 + (400 / ln(10)) * centered log strength; global mean 1500 when the observed comparison graph is connected",
            "weighting": "One unit per actual judgment; no majority voting, per-case reweighting or Gate-derived outcomes",
            "score_rate": "(wins + 0.5 * ties) / actual games; virtual draws excluded",
            "bootstrap": "rater-clustered, with replacement; all a sampled rater's votes retained together",
            "bootstrap_replicates": replicates, "bootstrap_seed": seed,
            "confidence": "95% percentile intervals; null with fewer than two raters or zero replicates; conditional on cases and observed comparison graph",
            "top1": "Fractional credit for equal-strength maxima, within each connected component",
            "disconnected_graph": "Separate fits and mean 1500 per observed connected component; no cross-component rank",
            "split": "case_id containing literal mine => IID; otherwise OOD",
            "exclusions": "Media issues are not votes. No self comparisons or repeated items are scheduled.",
        },
        "coverage": {"expected_tasks": expected, "target_judgments": expected * study.raters_per_pair,
                     "completed_tasks": sum(n >= study.raters_per_pair for n in counts.values()),
                     "started_tasks": len(counts), "remaining_judgments": expected * study.raters_per_pair - sum(counts.values()),
                     "complete": len(counts) == expected and all(n >= study.raters_per_pair for n in counts.values())},
        "groups": groups,
    }


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            # Prevent spreadsheet formulas in free-text cells.
            writer.writerow({k: ("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v) for k, v in row.items()})


def export(output, records, summary):
    previous = output / "summary.json"
    if previous.is_file() and json.loads(previous.read_text())["protocol"].get("id") != ANNOTATION_PROTOCOL:
        raise ValueError("Old annotation protocol in output folder; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("votes.json", records), ("summary.json", summary)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    raw, issues = [], []
    for record in records:
        flat = {k: v for k, v in record.items() if k != "vote"}
        if record["state"] == "issue":
            issues.append(flat)
        else:
            flat.update(normalize_vote(record["vote"]))
            raw.append(flat)
    write_csv(output / "votes.csv", raw)
    write_csv(output / "issues.csv", issues)
    boards, pairs, sections = [], [], []
    for group, value in summary["groups"].items():
        sections.append(f"<h2>{group}</h2><p>投票 {value['completed_judgments']} · 播放问题 {value['media_issues']}</p>")
        if len(value["components"]) > 1:
            sections.append("<p>比较图不连通：仅可在同一分量内比较名次。</p>")
        sections.append("<table><tr><th>模型</th><th>分量</th><th>名次（分量内）</th><th>Elo</th><th>95% CI</th><th>第一名概率（分量内）</th><th>得分率</th><th>投票数</th></tr>")
        for entry in value["leaderboard"]:
            entry = dict(entry)
            interval = entry.pop("ci95")
            boards.append(dict(group=group, **entry, ci95_low=interval[0] if interval else None, ci95_high=interval[1] if interval else None))
            ci = f"{interval[0]:.1f}–{interval[1]:.1f}" if interval else "N/A"
            top = entry["top1_probability_in_component"]
            top_text = f"{top:.1%}" if top is not None else "N/A"
            sections.append(f"<tr><td>{html.escape(entry['model'])}</td><td>{entry['component']}</td><td>{entry['rank_in_component']}</td><td>{entry['elo']:.1f}</td><td>{ci}</td><td>{top_text}</td><td>{entry['score_rate']:.1%}</td><td>{entry['games']}</td></tr>")
        sections.append("</table><details><summary>模型对胜 / 平 / 负</summary><table><tr><th>模型 A</th><th>模型 B</th><th>A 胜</th><th>平</th><th>B 胜</th><th>A 得分率</th></tr>")
        for entry in value["pairs"]:
            pairs.append(dict(group=group, **entry))
            if not entry["games"]:
                continue
            sections.append(f"<tr><td>{html.escape(entry['model_a'])}</td><td>{html.escape(entry['model_b'])}</td><td>{entry['wins']}</td><td>{entry['ties']}</td><td>{entry['losses']}</td><td>{entry['a_score_rate']:.1%}</td></tr>")
        sections.append("</table></details>")
    write_csv(output / "leaderboard.csv", boards)
    write_csv(output / "pairs.csv", pairs)
    document = """<!doctype html><meta charset="utf-8"><title>MemoryBench Pairwise Report</title>
<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#172b35}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}th{background:#eff5f5}h2{margin-top:50px}</style>"""
    document += "<h1>" + html.escape(summary["title"]) + "</h1>"
    document += "<p>每题一个 A / B / Tie 选择。Bradley–Terry · 平局半胜 · 连通比较图均值 1500 · 按评估者 bootstrap。得分率 =（胜 + 0.5 × 平）/ 实际投票数。</p>"
    document += "<p>" + ("已达到目标标注量。" if summary["coverage"]["complete"] else "标注尚未完成；以下为阶段性结果。") + "</p>"
    document += "".join(sections)
    (output / "report.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be >= 0")
    if not args.db.is_file():
        parser.error("Database does not exist")
    study, signature, tasks = load_study(args.study)
    records = Store(args.db, study, signature, tasks).records()
    summary = summarize(records, study, args.bootstrap, args.seed)
    summary["study_signature"] = signature
    export(args.output, records, summary)
    print(json.dumps({"output": str(args.output.resolve()), "coverage": summary["coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
