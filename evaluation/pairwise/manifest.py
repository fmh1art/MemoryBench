"""Strict study contract. Only explicit, registered local files can be served."""

import hashlib
import itertools
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import VERSION


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Clip(Record):
    path: str
    start_seconds: float = Field(default=0, ge=0)
    duration_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def interval(self):
        if self.start_seconds >= self.duration_seconds:
            raise ValueError("Video start must precede its end")
        return self


class Prediction(Clip):
    # Omitted = unrecorded; explicit null = verified no text conditioning.
    prompt: str | None = None
    prompt_recorded: bool = False


class Case(Record):
    case_id: str = Field(min_length=1)
    history: Clip
    gt: Clip
    predictions: dict[str, Prediction]
    action_description: str = ""
    memory_target: str = ""
    success_criterion: str = ""
    read_start_seconds: float = Field(default=0, ge=0)
    read_end_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def read_interval(self):
        end = self.gt.duration_seconds - self.gt.start_seconds
        if self.read_start_seconds > end + 0.001:
            raise ValueError("Read start is outside the GT future")
        if self.read_end_seconds is not None and not (
            self.read_start_seconds <= self.read_end_seconds <= end + 0.001
        ):
            raise ValueError("Read interval is outside the GT future")
        return self


class Study(Record):
    version: str = VERSION
    title: str = "MemoryBench · 人工记忆评估"
    seed: int = 42
    raters_per_pair: int = Field(default=3, ge=1, le=100)
    baseline: str | None = None
    models: list[str] = Field(min_length=2)
    cases: list[Case] = Field(min_length=1)
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def identities(self):
        if self.version != VERSION:
            raise ValueError("Unsupported study version")
        if len(set(self.models)) != len(self.models) or any(not m.strip() for m in self.models):
            raise ValueError("Model names must be unique and nonempty")
        if self.baseline is not None and self.baseline not in self.models:
            raise ValueError("Baseline must be a study model")
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("case_id must be globally unique; qualify duplicate batch IDs")
        for case in self.cases:
            if set(case.predictions) != set(self.models):
                raise ValueError(f"All models must cover the common cohort: {case.case_id}")
        return self


def split(case_id):
    return "IID" if "mine" in case_id else "OOD"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_study(path):
    path = Path(path).resolve()
    study = Study.model_validate_json(path.read_text(encoding="utf-8"))
    fingerprints = []
    for case in study.cases:
        for clip in [case.history, case.gt, *case.predictions.values()]:
            clip.path = str((path.parent / clip.path).resolve())
            file = Path(clip.path)
            if not file.is_file():
                raise ValueError(f"Missing video for {case.case_id}: {file}")
            stat = file.stat()
            fingerprints.append((clip.path, stat.st_size, stat.st_mtime_ns))
    signature = digest([study.model_dump(), fingerprints])
    tasks = {}
    for case in study.cases:
        for left, right in itertools.combinations(sorted(study.models), 2):
            if study.baseline and study.baseline not in (left, right):
                continue
            key = digest([case.case_id, left, right])[:24]
            tasks[key] = {"id": key, "case": case, "left": left, "right": right}
    selected = study.provenance.get("selected_task_ids")
    if selected is not None:
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(key, str) for key in selected)
                or len(set(selected)) != len(selected)
                or set(selected) - tasks.keys()):
            raise ValueError("selected_task_ids must be a nonempty list of distinct known tasks")
        tasks = {key: tasks[key] for key in selected}
    return study, signature, tasks
