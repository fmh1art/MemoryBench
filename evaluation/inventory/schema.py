from pydantic import Field, model_validator

from ..schema import Evidence, Record


class ObjectGroup(Record):
    object_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    gt_count: int | None = Field(ge=0)
    gen_count: int | None = Field(ge=0)
    evidence: list[Evidence] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_empty_group(self):
        if self.gt_count == 0 and self.gen_count == 0:
            raise ValueError("Do not list object groups absent in both videos")
        return self


class Judgment(Record):
    region_assessable: bool
    # A generated failure is scored, even when counts cannot be recovered.
    # Defaults preserve compatibility with earlier saved annotations.
    gt_reference_clear: bool = False
    gen_region_exposed: bool = False
    generated_quality_failure: bool = False
    objects: list[ObjectGroup]
    evidence: list[Evidence] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_groups(self):
        if self.generated_quality_failure and not (
            self.gt_reference_clear and self.gen_region_exposed
        ):
            raise ValueError(
                "Generated quality failure requires clear GT and exposed Gen region"
            )
        if self.generated_quality_failure and (
            not self.objects or any(obj.gt_count is None for obj in self.objects)
        ):
            raise ValueError(
                "Generated quality failure requires known scoped GT counts"
            )
        ids = [obj.object_id for obj in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError(
                "object_id must be unique; do not double-count an object group"
            )
        return self
