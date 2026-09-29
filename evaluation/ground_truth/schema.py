"""Manually specified memory target and probe success condition."""

from typing import Literal
from pydantic import Field
from ..common.schema import Record


class MemoryAnnotation(Record):
    schema_version: Literal["memory_annotation.v1"] = "memory_annotation.v1"
    evaluation_case_id: str = Field(min_length=1)
    memory_target: str = Field(min_length=1)
    read_action: Literal["open_drawer", "open_box", "turn_view", "other"]
    action_description: str = Field(min_length=1)
    success_criterion: str = Field(min_length=1)
    annotation_source: str = "manual"
