from typing import Literal

from pydantic import Field, field_validator

from ..schema import Evidence, Record


class BinaryDecision(Record):
    label: Literal[0, 1]
    evidence: list[Evidence] = Field(min_length=1)

    @field_validator("label", mode="before")
    @classmethod
    def strict_binary(cls, value):
        if type(value) is not int:
            raise ValueError(
                "Binary label must be integer 0 or 1, never bool/float/string"
            )
        return value


GATE_VERSION = "interaction_physics.v1"


class Judgment(Record):
    observability: BinaryDecision
    interaction_physics: BinaryDecision
