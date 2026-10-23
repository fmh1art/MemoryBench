"""Validated, coarse source annotations; no human-verified case boundaries."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt


class Record(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Segment(Record):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    description: str = Field(min_length=1, max_length=3000)
    evidence_frames: list[StrictInt] = Field(min_length=1, max_length=24)


class MemoryEvent(Record):
    kind: Literal['write', 'occlusion', 'distractor', 'read', 'update']
    description: str = Field(min_length=1, max_length=2000)
    evidence_frames: list[StrictInt] = Field(min_length=1, max_length=24)


class Candidate(Record):
    prefix_start_seconds: float = Field(ge=0)
    prefix_end_seconds: float = Field(gt=0)
    future_start_seconds: float = Field(ge=0)
    future_end_seconds: float = Field(gt=0)
    memory_target: str = Field(min_length=1, max_length=2000)
    why_requires_memory: str = Field(min_length=1, max_length=2000)
    write_evidence_frames: list[StrictInt] = Field(min_length=1, max_length=24)
    read_evidence_frames: list[StrictInt] = Field(min_length=1, max_length=24)


class Suitability(Record):
    decision: Literal['suitable', 'potential', 'unsuitable', 'uncertain']
    confidence: Literal['high', 'medium', 'low']
    rationale: str = Field(min_length=1, max_length=4000)
    memory_events: list[MemoryEvent] = Field(max_length=24)
    candidate_cases: list[Candidate] = Field(max_length=3)
    limitations: list[str] = Field(min_length=1, max_length=12)


class VideoAnnotation(Record):
    overall_description: str = Field(min_length=1, max_length=4000)
    segments: list[Segment] = Field(min_length=1, max_length=12)
    main_objects: list[str] = Field(max_length=40)
    memorybench: Suitability

    def validate_timeline(self, duration, frame_count=24):
        previous = 0.0
        for segment in self.segments:
            if not 0 <= segment.start_seconds < segment.end_seconds <= duration + 0.05:
                raise ValueError('Segment outside source duration or reversed')
            if segment.start_seconds < previous - 0.05:
                raise ValueError('Segments must be ordered and non-overlapping')
            previous = segment.end_seconds
        references = [s.evidence_frames for s in self.segments]
        references += [e.evidence_frames for e in self.memorybench.memory_events]
        for case in self.memorybench.candidate_cases:
            if not (0 <= case.prefix_start_seconds < case.prefix_end_seconds <= case.future_start_seconds < case.future_end_seconds <= duration + 0.05):
                raise ValueError('Candidate prefix/future intervals must be ordered and non-overlapping')
            references.extend([case.write_evidence_frames,case.read_evidence_frames])
        if any(not 1 <= index <= frame_count for group in references for index in group):
            raise ValueError('Evidence frame index outside contact sheet')
        if self.memorybench.decision == 'suitable' and not self.memorybench.candidate_cases:
            raise ValueError('Suitable requires at least one evidence-backed candidate')
        return self
