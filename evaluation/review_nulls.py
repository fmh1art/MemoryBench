"""Review only old nulls, preserve Gate/numeric grades, then recompute Python scores.

Each affected frame uses one GT/Gen image query for all unresolved submetrics.
Original judge logs remain intact; review logs and effective annotations are separate.
"""

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Literal

from pydantic import Field, create_model, model_validator

from .inventory.schema import Judgment as Inventory
from .inventory.scoring import matched_objects, score_inventory
from .judge import Judge, write_json
from .schema import Evidence, Metric, Record
from .scoring import SCORING_VERSION, average_memory_frames, score_memory, summarize

RUBRIC = """Audit ONLY the requested unresolved metrics in this ONE GT/Gen pair.
I_gt is the GT reference; I_gen is generated. Never compare aesthetic quality.
Existing annotations/evidence are clues, not visual proof. Inspect the two images.
Return exactly one decision for each requested metric; do not rescore existing numbers or Gate.

keep_null: GT does not expose the target/fact (including before opening a container), GT itself
is unreadable, the fact is inapplicable, or natural occlusion/viewpoint/projection prevents
comparison. A clear GT plus a camera view that hides the Gen target is not proof of corruption.
zero: GT clearly establishes the requested fact AND the corresponding Gen region is exposed AND
generated blur, smearing, fusion, collapse or incoherent geometry makes that fact unreadable.
All three flags must be true. Do not penalize merely lower sharpness when the fact is readable.
If corruption cannot be visually established, keep_null and explain the remaining uncertainty.

Inventory: decide whether generated corruption prevents identity/counting of scoped targets;
do not invent unknown counts, missing objects or extra objects. A zero is a quality-failure
override for Presence/Count; diagnostics remain unknown. If GT inventory itself is not clear,
keep_null. Appearance: only supplied Inventory-matched groups; an absent/unmatched object is
excluded, never an Appearance zero. Spatial: tolerate reasonable camera differences and natural
occlusion; require reference anchors/relations visible in GT before attributing failure to Gen.
For every decision, cite this frame and explicitly describe GT visibility, Gen visibility and
whether an actual generated defect is present. Output the supplied JSON schema only.
"""


class Decision(Record):
    metric: str
    decision: Literal["keep_null", "zero"]
    gt_reference_clear: bool
    gen_region_exposed: bool
    generated_quality_failure: bool
    evidence: list[Evidence] = Field(min_length=1)

    @model_validator(mode="after")
    def require_failure_evidence(self):
        if self.decision == "zero" and not (
            self.gt_reference_clear
            and self.gen_region_exposed
            and self.generated_quality_failure
        ):
            raise ValueError(
                "Zero requires clear GT, exposed Gen and generated degradation"
            )
        return self


class Review(Record):
    decisions: list[Decision] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_metrics(self):
        names = [item.metric for item in self.decisions]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate metric decisions")
        return self


def unresolved_metrics(frame):
    """Derived Appearance with no matches stays excluded; avoid absence double penalties."""
    keys = []
    inventory_data = frame["judgments"].get("inventory")
    inventory = Inventory.model_validate(inventory_data) if inventory_data else None
    if (
        inventory is not None
        and frame["scores"]["dimensions"].get("inventory", {}).get("status")
        == "insufficient_evidence"
        and inventory.objects
        and all(obj.gt_count is not None for obj in inventory.objects)
    ):
        keys.append("inventory")
    for key, item in frame["scores"]["metrics"].items():
        if key.startswith("inventory.") or item["status"] != "insufficient_evidence":
            continue
        if key.startswith("appearance.") and (
            inventory is None or not matched_objects(inventory)
        ):
            continue
        keys.append(key)
    return keys


def apply_review(frame, review, expected):
    """Only null→zero is allowed. Never change a scored grade or invent counts."""
    decisions = {item.metric: item for item in review.decisions}
    if set(decisions) != set(expected):
        raise ValueError("Review must cover exactly the requested unresolved metrics")
    if not set(expected) <= set(unresolved_metrics(frame)):
        raise ValueError("Cannot review an existing scored or derived unmatched metric")
    for key in expected:
        if key == "inventory":
            status = score_inventory(Inventory.model_validate(frame["judgments"][key]))[
                "status"
            ]
        else:
            dimension, metric = key.split(".")
            status = frame["judgments"][dimension][metric]["status"]
        if status != "insufficient_evidence":
            raise ValueError("Only unresolved original annotations can be reviewed")
    for key, item in decisions.items():
        if item.decision != "zero":
            continue
        if key == "inventory":
            judgment = frame["judgments"]["inventory"]
            judgment.update(
                gt_reference_clear=True,
                gen_region_exposed=True,
                generated_quality_failure=True,
            )
            judgment["evidence"] += [e.model_dump() for e in item.evidence]
        else:
            dimension, metric = key.split(".")
            frame["judgments"][dimension][metric] = {
                "status": "scored",
                "score": 0,
                "evidence": [e.model_dump() for e in item.evidence],
            }
    frame["null_policy_review"] = review.model_dump()


def recompute(result):
    if result["status"] == "gate_failed":
        return result
    for frame in result["memory_frames"]:
        judgments = {}
        for dim, data in frame["judgments"].items():
            schema = (
                Inventory
                if dim == "inventory"
                else create_model(
                    dim.title() + "Judgment",
                    __base__=Record,
                    **{name: (Metric, ...) for name in data},
                )
            )
            judgments[dim] = schema.model_validate(data)
        frame["scores"] = score_memory(judgments, result["selected_metrics"])
        frame["status"] = frame["scores"]["status"]
        frame["memory_score"] = frame["scores"]["memory_score"]
    result["scores"] = average_memory_frames(result["memory_frames"])
    result["status"] = result["scores"]["status"]
    result["memory_score"] = result["scores"]["memory_score"]
    result["scoring_version"] = SCORING_VERSION
    result["aggregation_policy"] = (
        "skip_reference_unavailable_score_generated_failure_zero"
    )
    return result


async def review_results(source, output, judge):
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError(
            "Use a separate output directory to preserve the original audit"
        )
    if not output.exists():
        shutil.copytree(source, output, copy_function=os.link)
        (output / "summary.json").unlink(missing_ok=True)
        (output / "verification.json").unlink(missing_ok=True)
    paths = sorted(source.glob("*/result.json"))
    work = []
    for path in paths:
        result = json.loads(path.read_text())
        if result["status"] == "gate_failed":
            continue
        for frame in result.get("memory_frames", []):
            keys = unresolved_metrics(frame)
            if keys:
                work.append((path, result, frame, keys))
    completed, zeros, errors = 0, 0, []
    lock = asyncio.Lock()

    def progress(running):
        return {
            "running": running,
            "completed_frame_reviews": completed,
            "total_frame_reviews": len(work),
            "zero_decisions": zeros,
            "errors": errors,
            "source": str(source),
            "output": str(output),
        }

    async def one(path, result, frame, keys):
        nonlocal completed, zeros
        folder = output / result["case_id"] / "judges" / f"frame_{frame['frame_index']}"
        log = folder / "null_review.json"
        context = {
            "case": result["case"],
            "memory_frame": {
                k: v
                for k, v in frame.items()
                if k in result["memory_sampling"]["frames"][0]
            },
            "sampling": {"read_frame_indices": [frame["frame_index"]]},
            "requested_unresolved_metrics": keys,
            "original_judgments": frame["judgments"],
            "appearance_matching": {
                "objects": matched_objects(
                    Inventory.model_validate(frame["judgments"]["inventory"])
                )
                if "inventory" in frame["judgments"]
                else []
            },
        }
        images = [
            path.parent / "memory_frames" / f"{frame['frame_index']}_{label}.jpg"
            for label in ("gt", "gen")
        ]
        try:
            old = json.loads(log.read_text()) if log.exists() else None
            if (
                old
                and "parsed" in old["attempts"][-1]
                and "error" not in old["attempts"][-1]
            ):
                if old["context"] != context or old["rubric"] != RUBRIC:
                    raise ValueError("Cached review context/rubric mismatch")
                review = Review.model_validate(old["attempts"][-1]["parsed"])
            else:
                # Required top-level fields are dynamic: a missing/extra metric triggers validation retry.
                def exact_coverage(self):
                    if {d.metric for d in self.decisions} != set(keys):
                        raise ValueError(
                            "Return exactly the requested metric decisions"
                        )
                    return self

                schema = create_model(
                    "FrameNullReview",
                    __base__=Review,
                    __validators__={
                        "coverage": model_validator(mode="after")(exact_coverage)
                    },
                )
                review = await judge.ask(
                    "null_review",
                    context,
                    images,
                    folder,
                    schema_override=schema,
                    rubric_override=RUBRIC,
                )
            apply_review(frame, review, keys)
            zeros += sum(d.decision == "zero" for d in review.decisions)
        except (RuntimeError, ValueError, OSError, KeyError) as exc:
            errors.append(
                {
                    "case_id": result["case_id"],
                    "frame": frame["frame_index"],
                    "error": str(exc)[:1000],
                }
            )
        async with lock:
            completed += 1
            write_json(output / "review_progress.json", progress(True))
            if completed % 20 == 0 or completed == len(work):
                print(
                    f"Null review {completed}/{len(work)} | zero decisions {zeros} | errors {len(errors)}",
                    flush=True,
                )

    print(
        f"Reviewing {len(work)} affected frame pairs; Gate and existing numeric grades retained",
        flush=True,
    )
    # Shared result instances ensure independent frame updates survive case recomputation.
    by_case = {}
    for path, result, frame, keys in work:
        if result["case_id"] not in by_case:
            by_case[result["case_id"]] = (path, result)
    await asyncio.gather(*(one(*item) for item in work))
    write_json(output / "review_progress.json", progress(False))
    if errors:
        raise RuntimeError(
            f"{len(errors)} reviews failed; rerun to resume saved successful reviews"
        )
    # Each source result was loaded once, so all its frame edits belong to that result.
    results = []
    for path in paths:
        case_id = path.parent.name
        result = (
            by_case[case_id][1] if case_id in by_case else json.loads(path.read_text())
        )
        result = recompute(result)
        result["null_policy_source"] = str(path)
        write_json(output / case_id / "result.json", result)
        results.append(result)
    write_json(output / "summary.json", summarize(results))
    run = json.loads((source / "run.json").read_text())
    run.update(
        scoring_version=SCORING_VERSION,
        null_policy_source=str(source),
        null_review=progress(False),
    )
    write_json(output / "run.json", run)
    print(json.dumps(summarize(results)["report"], ensure_ascii=False), flush=True)


async def main_async(args):
    import httpx

    judge = Judge(
        args.endpoint,
        args.model,
        os.environ[args.api_key_env],
        concurrency=32,
        timeout=240,
        retries=3,
        temperature=None,
        max_tokens=4096,
        qpm=240,
        tpm=2187000,
        tokens_per_request=16000,
        transport=httpx.AsyncHTTPTransport(
            local_address="0.0.0.0",
            limits=httpx.Limits(
                max_connections=32, max_keepalive_connections=32, keepalive_expiry=60
            ),
        ),
    )
    try:
        await review_results(Path(args.source), Path(args.output), judge)
    finally:
        await judge.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "endpoint", "model"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--api-key-env", default="MEM_BENCH_API_KEY")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
