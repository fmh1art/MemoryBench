Annotate inventory semantics once. I_gt is the GT reference; I_gen is the generated video.
Do NOT calculate scores, ratios, precision/recall, or totals of missing/spurious objects.
The evaluator computes those from your annotations.

Scope: objects named by memory_target, and objects in its explicitly specified target region.
For a region target (drawer/shelf/box), include all visibly separable contents, including unexpected
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
- Count physical objects in the Read state, never appearances summed over frames.
- Use multiple Read views to disambiguate occlusion and duplicates. Do not count reflections,
  printed images, shadows, hands or visual fragments as additional objects.
- gt_count > 0 and gen_count = 0 means a reference group is absent from a clearly inspected Gen region.
- gt_count = 0 and gen_count > 0 means an unexpected group, after confirming GT absence in that region.
- Include count mismatches within a shared group (e.g. GT=3, Gen=1); code derives missing instances.
- Do not list groups absent in both. An empty objects list means both regions are visibly empty,
  or region_assessable=false; explain which in evidence.
- Use integer counts >=0 when reliably visible. Use null for an unresolvable count, never an estimate
  or an inferred hidden object. region_assessable=false when the region cannot support inventory.

Object groups must be mutually exclusive and collectively cover the scoped inventory in either
video. Use the latest corresponding Read frame pair where the scoped region is reliably inspectable
in both videos as the inventory snapshot; identify that frame in overall evidence. Adjacent views
may disambiguate identities/occlusion, but do not mix counts from different world states. If visibility
requires multiple views, establish the latest recoverable state and cite those supporting views.
Keep reference identities stable through time; describe any genuine GT changes and any clear Gen
object disappearance/duplication across Read. Observable Gen changes are memory errors, not grounds
for region_assessable=false: use the selected snapshot counts and document temporal variation.
Only actual visibility/identity/count uncertainty warrants null counts or an unassessable region.

Provide evidence for every group and the overall region assessment, citing Read frame numbers
and concrete GT/Gen observations. No numeric quality score belongs in this response.
