Judge evaluation conditions once, returning two independent binary decisions. Do not score memory.
I_gt is the GT reference; I_gen is the generated video, not an aesthetic competitor.

**Observability (0/1): reliable visibility of the requested MEMORY TARGETS, not the room.**
Use memory_objects as the explicit focus list when supplied; otherwise use memory_target.
Assess GT and Gen at corresponding Read views. Return 1 only when the essential target facts
visible in GT can be reliably assessed in Gen: separable objects/counts, identity-bearing visual
features and target placement relative to relevant anchors. The comparison may reveal a clear
match, a clearly wrong object/state, or confirmed absence. It need not reveal correct memory.
Clear walls, shelf grids, container outlines, hands or unrelated objects do NOT establish that
the listed memory targets are observable. Never substitute an easier non-target for a requested target.

Return 0 when sustained blur, translucent/opaque obstruction, washed-out detail, ghosting, darkness,
small scale or missing coverage prevents reliable target inspection. Recognizing 'a refrigerator',
'a shelf', 'a bottle-like blob' or a rough object location is insufficient when the target's boundaries,
count or distinguishing features cannot be resolved. For example, if the requested targets are a
white-capped small bottle, a red-capped large bottle and a purple sealed box, visible shelf lines alone
cannot compensate for blur that prevents inspecting those target regions and distinguishing these
items from background or smears. Do not infer their identity or visibility from the annotation.

**Distinguish confirmed missing content from unobservable content.**
- Clear, unobstructed expected region with resolvable background/object boundaries and no target:
  observable=1; confirmed missing memory is a later Memory failure.
- Blurred, veiled, smeared or obstructed expected region where a target could be hidden or merged:
  observable=0; 'I cannot see the object' is NOT evidence that the region is visibly empty.
- A clearly visible wrong/replaced/deformed object is still observable if the concrete mismatch
  can be resolved. A memory error by itself must not fail this Gate.
Do not use the visible-empty exception unless the expected region is genuinely inspectable at the
resolution needed for the requested targets. A vague low-detail patch does not prove absence.

Allow brief motion blur or partial occlusion only when later/other Read views provide reliable
comparison of the essential targets or their clearly inspected expected regions. Clear exterior
or pre-Read frames cannot rescue unreadable target views. Do not require every sampled frame to
be sharp; require adequate target-specific Read evidence. If only coarse layout is recoverable,
or a required target region remains unassessable across Read, return 0.

Observability evidence must name the requested targets/expected regions, cite corresponding Read
frames, and describe the concrete visibility: boundaries, distinguishing features, or why an absence
can be established. On failure, identify which requested facts are blocked and by what degradation.
Do not merely say 'useful scale' or 'the region is visible'. Evaluate legibility, never visual beauty.

**Action Following (0/1).** Return 1 when Gen visibly achieves action_description and
success_criterion during/after the Read operation: e.g. the intended drawer opens, the intended
container is revealed, or the camera turns to the specified target region. Use the temporal sequence
and GT as reference for what the successful read reveals. Allow small timing/trajectory/viewpoint
differences if the same semantic endpoint and access are achieved. Return 0 for wrong region/container,
wrong operation, no meaningful action, or evidence too weak to establish success. A generated clip
already at the requested final view can establish success only when the requested criterion is the
final view itself; it cannot prove a required opening or motion happened.

Keep labels independent: an incorrect action may still produce a clear view; a correct action may
produce an unreadable view. Evaluate action success without requiring correct object contents.
Each decision must include specific GT/Gen frame evidence. Read frames support Observability;
any frame can support Action Following. If uncertain, conservatively return 0 and describe the
limitation. Gate passes only when both labels are 1.
