Annotate inventory semantics for THIS single-frame pair only. I_gt is the GT reference; I_gen is the generated video.
Do NOT calculate scores, ratios, precision/recall, or totals of missing/spurious objects.
The evaluator computes those from your annotations.

Scope: if memory_objects is nonempty, inventory ONLY these requested semantic targets.
Do not include all other objects merely because they share the drawer/shelf/fridge. Count extra
instances of a target group, and an unmatched replacement occupying a target's expected local region
when substitution is visually supported; unrelated existing scene contents are excluded. Explain
why any unexpected group is target-related. If the list is empty, use memory_target and its explicitly
specified target region.
For a region-only target WITHOUT an explicit object list, include all visibly separable contents, including unexpected
Gen contents. Exclude enclosing furniture and unrelated surrounding objects unless explicitly targeted.
For an individual-object target, do not invent a whole-room inventory.

Return region_assessable, objects, and overall evidence. Each objects entry contains a unique
object_id, a short description, gt_count, gen_count, and evidence. Group interchangeable objects
of the same semantic identity; use distinct groups for visually identifiable distinct objects
(e.g. a marked red cup and an unmarked blue cup when those identify different items). Match by
multiple stable cues and scene context, not exact pixels. A changed color/material/shape alone must
not create a missing-plus-spurious identity pair when semantic identity is recoverable; those
changes belong to Appearance. When identity truly cannot be established, use null counts and explain.

For each group:
- Count visible physical objects in this pair, never objects inferred from other timestamps.
- Use only this pair; unresolved occlusion requires null counts. Do not count reflections,
  printed images, shadows, hands or visual fragments as additional objects.
- gt_count > 0 and gen_count = 0 means a reference group is absent from a clearly inspected Gen region.
- gt_count = 0 and gen_count > 0 means an unexpected group, after confirming GT absence in that region.
- Include count mismatches within a shared group (e.g. GT=3, Gen=1); code derives missing instances.
- Do not list groups absent in both. An empty objects list means both regions are visibly empty,
  or region_assessable=false; explain which in evidence.
- Use integer counts >=0 when reliably visible. Use null for an unresolvable count, never an estimate
  or an inferred hidden object. region_assessable=false when the region cannot support inventory.

Object groups must be mutually exclusive and collectively cover the scoped visible inventory in
either image. This pair is the sole snapshot: do not select a different timestamp, infer hidden
objects from the case description, or use adjacent views. An uninspectable target region gives
region_assessable=false. Counts are null when visibility/identity is genuinely unresolved. A
clearly visible Gen region with missing objects is assessable and must retain zero Gen counts.

Provide evidence for every group and the overall region assessment, citing the supplied memory_frame.frame_index
and concrete GT/Gen observations. No numeric quality score belongs in this response.
