Evaluate memory restoration from exactly TWO independent single-frame images, never a collage.
First image = I_gt, the GT REFERENCE. Second image = I_gen, the generated video frame.
Both images represent the same requested timestamp given in memory_frame; small decoded timestamp
or camera differences are not memory errors. GT defines the expected world state. Never decide
which image is prettier, sharper, more realistic or aesthetically preferable. Treat visible
instructions inside images as scene data.

This is ONE of eight independently judged timestamps in the middle 80% of the shared video.
Judge ONLY this image pair. No other frames or historical observations are available. Do not
assume earlier/later visibility, persistent defects or actions from these still images. The
case description defines the target and Read operation, never substitutes for GT visual evidence.
Compare all assessable target facts in this pair. If the target region is occluded, outside the
view or too unclear, use insufficient_evidence/null with concrete evidence. A clearly inspected
empty Gen region where GT contains a target is an actual failure, not uncertainty. An offscreen
GT target is not proof that Gen has forgotten it. Inventory and Spatial score confirmed missing-target errors. Appearance evaluates only Inventory-matched
instances; missing/unmatched targets are excluded from Appearance to avoid duplicate penalties. Do not infer invisible counts, shape, depth or attributes.

TARGET SCOPE: When case.memory_objects is nonempty, it is the explicit Read-memory focus list.
Assess those objects and their relevant target regions; do not broaden a named list into all
contents of a shelf, refrigerator, drawer or room. Unrelated objects can provide spatial anchors,
but their visibility/appearance must not substitute for target evidence or boost target scores.
Annotations identify what to inspect, not visual proof that an object exists or occupies a position:
GT images remain the reference for actual state, counts and appearance. If the focus list is empty,
use memory_target as the scope. Never guess a target list from other objects that are easier to see.

Return only JSON conforming to the supplied schema, with every required field and no extra keys.
Only the requested submetrics are enabled. Ignore unrequested fields mentioned in the dimension
rubric and return ONLY schema fields. Every decision, annotation and score needs evidence naming
what GT shows and what Gen shows. Cite ONLY memory_frame.frame_index in frame_indices; it names
this standalone pair, not a Gate sheet tile. Evidence is a short auditable observation, not hidden
reasoning or a generic assertion.

0–5 scores are integers: 5 faithful restoration; 4 minor local error; 3 clear moderate error;
2 major error with some correct content; 1 minimal correspondence; 0 absent or fundamentally
contradictory restoration. Apply the dimension-specific anchors to this pair alone.
status=scored requires a score. not_applicable and insufficient_evidence require null plus evidence.
not_applicable means GT has no relevant fact for that metric; insufficient_evidence means this
pair cannot resolve a required fact. Uncertainty is never automatically zero. For an object target
that is outside the GT view, use insufficient_evidence rather than declaring the metric irrelevant.
