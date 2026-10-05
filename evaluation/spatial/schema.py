from ..schema import Metric, Record


class Judgment(Record):
    object_localization: Metric
    relative_spatial_relation: Metric
