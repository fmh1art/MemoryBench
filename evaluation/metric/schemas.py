"""Contracts for manual annotations, action evidence and direct two-image judging."""

from typing import Literal
from pydantic import Field, StrictBool, model_validator
from ..common.schema import Record

METRICS = ("inventory", "appearance", "spatial")


class GateJudgment(Record):
    action_correct: StrictBool
    target_visible: StrictBool
    evidence_frame_ids: list[str]
    reason: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def evidence_required(self):
        if self.action_correct and self.target_visible and not self.evidence_frame_ids:
            raise ValueError("Passing Gate requires frame evidence")
        return self


class VisibleObject(Record):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=500)
    identity_features: str = Field(min_length=1, max_length=1000)


class VerifiedMatch(Record):
    gt_id: str
    gen_id: str
    evidence: str = Field(min_length=1, max_length=1000)


class Dimension(Record):
    score: float | None = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1500)


class Dimensions(Record):
    inventory: Dimension
    appearance: Dimension
    spatial: Dimension


class ObjectAppearance(Record):
    gt_id: str
    score: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1000)


class PairJudgment(Record):
    gt_readable: StrictBool
    generated_readable: StrictBool
    gt_objects: list[VisibleObject] = Field(max_length=64)
    generated_objects: list[VisibleObject] = Field(max_length=64)
    verified_matches: list[VerifiedMatch] = Field(max_length=64)
    appearance_items: list[ObjectAppearance] = Field(max_length=64)
    gt_spatial_applicable: StrictBool
    dimensions: Dimensions
    summary: str = Field(min_length=1, max_length=1500)

    @model_validator(mode="after")
    def consistent_identity_and_scores(self):
        gt_ids = [o.id for o in self.gt_objects]
        gen_ids = [o.id for o in self.generated_objects]
        if len(set(gt_ids)) != len(gt_ids) or len(set(gen_ids)) != len(gen_ids):
            raise ValueError("Object IDs must be unique within each image")
        used_gt, used_gen = set(), set()
        for pair in self.verified_matches:
            if pair.gt_id not in gt_ids or pair.gen_id not in gen_ids:
                raise ValueError(
                    "Verified matches must reference visible listed objects"
                )
            if pair.gt_id in used_gt or pair.gen_id in used_gen:
                raise ValueError("Verified object identity matching must be one-to-one")
            used_gt.add(pair.gt_id)
            used_gen.add(pair.gen_id)
        if not self.generated_readable and (
            self.generated_objects or self.verified_matches
        ):
            raise ValueError(
                "Unreadable generated target cannot have invented objects or matches"
            )
        if not self.gt_readable:
            return self  # The pipeline reports invalid GT as an error, never as model zero.
        appearance_ids = [item.gt_id for item in self.appearance_items]
        if set(appearance_ids) != set(gt_ids) or len(appearance_ids) != len(gt_ids):
            raise ValueError(
                "appearance_items must contain exactly one item for every GT object ID"
            )
        if not gt_ids and self.gt_spatial_applicable:
            raise ValueError("No GT objects means no object spatial facts")
        for name, applies in [
            ("spatial", self.gt_spatial_applicable),
        ]:
            if applies and getattr(self.dimensions, name).score is None:
                raise ValueError(
                    f"{name}: a GT-applicable fact needs a numeric score; missing generated content is zero, not N/A"
                )
        return self
