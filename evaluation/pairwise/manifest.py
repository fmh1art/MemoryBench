"""Strict study contract. Only explicit, registered local files can be served."""

import hashlib
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
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


class ModelReference(Record):
    clip: Clip
    action_description: str = ""
    memory_target: str = ""
    success_criterion: str = ""


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
            if self.provenance.get("allow_partial_models") is True:
                if len(case.predictions) < 2 or set(case.predictions) - set(self.models):
                    raise ValueError(f"Cases need at least two known models: {case.case_id}")
            elif set(case.predictions) != set(self.models):
                raise ValueError(f"All models must cover the common cohort: {case.case_id}")
        return self


def split(case_id):
    return "IID" if "mine" in case_id else "OOD"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def model_reference(study, case_id, model):
    value = study.provenance.get("model_references", {}).get(case_id, {}).get(model)
    return ModelReference.model_validate(value) if value is not None else None


def load_study(path):
    path = Path(path).resolve()
    study = Study.model_validate_json(path.read_text(encoding="utf-8"))
    def fingerprint(item):
        cid, clip = item
        clip.path = str((path.parent / clip.path).resolve())
        file = Path(clip.path)
        if not file.is_file():
            raise ValueError(f"Missing video for {cid}: {file}")
        stat = file.stat()
        return (clip.path, stat.st_size, stat.st_mtime_ns)
    clips = [(case.case_id, clip) for case in study.cases
             for clip in [case.history, case.gt, *case.predictions.values()]]
    # BOS metadata latency dominates a full human study. map retains the original
    # ordering, so signatures and existing task/session identities do not change.
    with ThreadPoolExecutor(max_workers=12) as pool:
        fingerprints = list(pool.map(fingerprint, clips))
    references = study.provenance.get("model_references", {})
    cases = {case.case_id: case for case in study.cases}
    required = study.provenance.get("reference_models", [])
    if not isinstance(references, dict) or set(references) - cases.keys():
        raise ValueError("Model references must name existing cases")
    if not isinstance(required, list) or set(required) - set(study.models):
        raise ValueError("reference_models must name study models")
    for cid, entries in references.items():
        if not isinstance(entries, dict) or set(entries) - cases[cid].predictions.keys():
            raise ValueError("Model references must name available case predictions")
        for model in entries:
            reference = model_reference(study, cid, model)
            clip = reference.clip
            clip.path = str((path.parent / clip.path).resolve())
            file = Path(clip.path)
            if not file.is_file():
                raise ValueError(f"Missing model reference for {cid}/{model}: {file}")
            stat = file.stat()
            fingerprints.append((clip.path, stat.st_size, stat.st_mtime_ns))
            entries[model] = reference.model_dump()
    for case in study.cases:
        if (set(required) & case.predictions.keys()) - references.get(case.case_id, {}).keys():
            raise ValueError(f"Missing required model reference for {case.case_id}")
    signature = digest([study.model_dump(), fingerprints])
    tasks = {}
    for case in study.cases:
        for left, right in itertools.combinations(sorted(case.predictions), 2):
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
