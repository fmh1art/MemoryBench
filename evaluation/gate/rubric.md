Judge evaluation conditions once, returning two independent binary decisions. Do not score memory.
I_gt is the GT reference; I_gen is the generated video, not an aesthetic competitor.

**Observability (0/1): GENERATED VIDEO VISUAL QUALITY / LEGIBILITY.**
This Gate checks whether Gen has usable visual quality, NOT whether memory targets are present,
correct, visible or already revealed. Use GT to understand the scene and distinguish normal motion,
occlusion and physical surfaces from generated artifacts. Do not judge aesthetic preference.
Return 1 when Gen depicts a visually legible scene: surfaces and boundaries are reasonably clear
and stable, without severe persistent degradation in the generated Read sequence. Do not require
the objects described in memory_objects/memory_target to be identifiable or restored to pass.

Return 0 for a genuine generated-quality failure: sustained severe blur, ghosting/double exposure,
artificial translucent smearing/veils, washed-out unreadable content, severe scene-wide corruption
or unstable rendering that makes the depicted scene materially unreadable. Name the visible artifact
and cite frames where it persists. A few recognizable shelf lines do not rescue an interior covered
by a strong generated haze/veil. An isolated clear context frame does not rescue a persistently
corrupted Read sequence. Brief motion blur, modest texture simplification, small camera differences
and local minor rendering defects are acceptable when the scene remains usable.

DO NOT fail Observability merely because:
- a GT object is missing in an otherwise clear Gen scene;
- Gen shows a clearly visible wrong object, count, appearance, shape or placement;
- a target is naturally hidden under a cup, inside a closed container, behind a hand or outside view;
- a particular sampled frame cannot support a reliable target-specific count or attribute judgment.
These are memory-content differences, normal visibility limits or action-access questions, not
proof of low generated visual quality. Action Following handles whether the requested operation
and target access occurred. Memory submetrics handle confirmed errors or insufficient evidence.

Concrete boundary cases:
- GT reveals a red block when a cup is lifted; Gen lifts the cup over a CLEAR EMPTY TABLETOP:
  Observability=1. The table lacks the GT object's features because the object is absent, not because
  the image is unreadable. Do not demand target boundaries to certify a clear empty surface.
- A cup remains inverted over a hidden object in a clear scene: Observability=1; the action and
  target-specific facts are evaluated separately. Natural opaque occlusion is not a rendering artifact.
- A refrigerator's generated interior has persistent artificial translucent streaking and strong
  blur across shelves/contents: Observability=0 for the visible image-quality degradation, regardless
  of whether reference bottles/boxes seem absent.
- A clear, legible scene contains an incorrect or deformed target: Observability=1; restoration
  fidelity belongs to Appearance/Geometry/Inventory/Spatial.

Evidence must describe Gen visual clarity or a concrete quality defect, with GT used as context.
A failing evidence statement must identify actual blur/ghosting/veiling/corruption; 'the target is
not visible', 'no target boundary', 'its identity is uncertain' or 'contents differ' alone cannot
justify 0. Do not invent a quality defect to explain a missing memory object.

**Action Following (0/1).** Return 1 when Gen visibly achieves action_description and
success_criterion during/after the Read operation: e.g. the intended drawer opens, the intended
container is revealed, or the camera turns to the specified target region. Use the temporal sequence
and GT as reference for the requested physical operation and access. If success_criterion also names
remembered contents (e.g. an empty drawer or a red block), ignore that content clause for this Gate:
opening the right drawer/lifting the right cup succeeds even if the restored contents are wrong. Allow small timing/trajectory/viewpoint
differences if the same semantic endpoint and access are achieved. Return 0 for wrong region/container,
wrong operation, no meaningful action, or evidence too weak to establish success. A generated clip
already at the requested final view can establish success only when the requested criterion is the
final view itself; it cannot prove a required opening or motion happened.

Keep labels independent: an incorrect action may still produce a clear view; a correct action may
produce an unreadable view. Evaluate action success without requiring correct object contents.
Each decision must include specific GT/Gen frame evidence. Read frames support Observability;
any frame can support Action Following. If uncertain, conservatively return 0 and describe the
limitation. Gate passes only when both labels are 1.
