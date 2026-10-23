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
    occlusion_evidence_frames: list[StrictInt] = Field(default_factory=list, max_length=24)
    occlusion_description: str = Field(default='', max_length=2000)
    reappearance_description: str = Field(default='', max_length=2000)
    same_target_evidence: str = Field(default='', max_length=2000)


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

    def validate_timeline(self, duration, frame_count=24, actual_seconds=None):
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
            references.extend([case.write_evidence_frames,case.occlusion_evidence_frames,case.read_evidence_frames])
        if any(not 1 <= index <= frame_count for group in references for index in group):
            raise ValueError('Evidence frame index outside contact sheet')
        if self.memorybench.decision == 'suitable' and not self.memorybench.candidate_cases:
            raise ValueError('Suitable requires at least one evidence-backed candidate')
        if self.memorybench.decision == 'suitable':
            kinds={event.kind for event in self.memorybench.memory_events}
            if not {'write','occlusion','read'}.issubset(kinds):
                raise ValueError('Suitable requires explicit write, physical occlusion and reappearance/read events')
            for case in self.memorybench.candidate_cases:
                if not case.occlusion_evidence_frames or not all(text.strip() for text in
                        [case.occlusion_description,case.reappearance_description,case.same_target_evidence]):
                    raise ValueError('Suitable requires physical-occlusion evidence and same-target reappearance proof')
                if not (max(case.write_evidence_frames)<min(case.occlusion_evidence_frames)
                        and max(case.occlusion_evidence_frames)<min(case.read_evidence_frames)):
                    raise ValueError('Evidence must be strictly ordered: visible -> occluded -> reappeared')
                if actual_seconds is not None:
                    writes=[actual_seconds[i-1] for i in case.write_evidence_frames]
                    hidden=[actual_seconds[i-1] for i in case.occlusion_evidence_frames]
                    reads=[actual_seconds[i-1] for i in case.read_evidence_frames]
                    if not all(case.prefix_start_seconds-0.05<=t<=case.prefix_end_seconds+0.05 for t in writes):
                        raise ValueError('Write evidence must fall in prefix')
                    if not all(case.future_start_seconds-0.05<=t<=case.future_end_seconds+0.05 for t in reads):
                        raise ValueError('Reappearance evidence must fall in future')
                    if not min(hidden)<=case.future_start_seconds+0.05<min(reads):
                        raise ValueError('Future must start after observed occlusion and before reappearance')
        return self
