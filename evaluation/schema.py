"""Shared input and evidence contracts. Dimension outputs live beside their rubrics."""

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Case(Record):
    case_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    gt_video: str = Field(min_length=1)
    gen_video: str = Field(min_length=1)
    memory_target: str = Field(min_length=1)
    memory_objects: list[Annotated[str, Field(min_length=1)]] = Field(
        default_factory=list
    )
    memory_objects_source: str | None = None
    action_description: str = Field(min_length=1)
    success_criterion: str = Field(min_length=1)
    read_start_seconds: float = Field(ge=0)
    read_end_seconds: float | None = Field(default=None, ge=0)
    # Video-local times corresponding to experiment t=0. No duration warping.
    gt_start_seconds: float = Field(default=0.0, ge=0)
    gen_start_seconds: float = Field(default=0.0, ge=0)
    lighting_style: bool = False

    @model_validator(mode="after")
    def check_interval(self):
        if (
            self.read_end_seconds is not None
            and self.read_end_seconds < self.read_start_seconds
        ):
            raise ValueError("read_end_seconds must be >= read_start_seconds")
        return self

    def resolve_videos(self, base: Path):
        data = self.model_dump()
        for key in ("gt_video", "gen_video"):
            data[key] = str((base / getattr(self, key)).resolve())
        # Benchmark references carry their Read-specific scope next to ground_truth/.
        # Explicit caller-provided targets take precedence over dataset annotations.
        reference = Path(data["gt_video"])
        if not data["memory_objects"] and reference.parent.name == "ground_truth":
            for name in ("case.json", "description.json"):
                annotation_path = reference.parent.parent / name
                if not annotation_path.is_file():
                    continue
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                if annotation.get("evaluation_case_id") != self.case_id:
                    raise ValueError(
                        f"GT annotation identity mismatch in {annotation_path}"
                    )
                objects = (annotation.get("metadata") or {}).get("memory_objects")
                if objects:
                    data["memory_objects"] = objects
                    data["memory_objects_source"] = (
                        str(annotation_path) + "#metadata.memory_objects"
                    )
                    break
        if data["memory_objects"] and data["memory_objects_source"] is None:
            data["memory_objects_source"] = "explicit_case_input"
        return type(self).model_validate(data)


class Evidence(Record):
    frame_indices: list[Annotated[int, Field(ge=1, le=12)]] = Field(min_length=1)
    observation: str = Field(min_length=1)


class Metric(Record):
    status: Literal["scored", "not_applicable", "insufficient_evidence"]
    score: Annotated[int, Field(ge=0, le=5)] | None
    evidence: list[Evidence] = Field(min_length=1)

    @model_validator(mode="after")
    def check_score(self):
        if (self.status == "scored") != (self.score is not None):
            raise ValueError(
                "score is an integer 0..5 only when status=scored; otherwise null"
            )
        return self
