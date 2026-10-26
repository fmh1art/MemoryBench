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
proof of low generated visual quality. Interaction Physics handles whether the requested operation
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

**Interaction Physics (0/1).** Check whether the requested Read interaction produces a
physically plausible response of the manipulated objects. Return 1 only when Gen visibly
performs the intended operation on the correct object/region, achieves the semantic access goal,
and shows a response consistent with the action: a pulled drawer moves along its rails, a lifted
cup moves with the grasp, or a pushed object moves plausibly after contact. Use the temporal
sequence and GT to establish the interaction context, contact, motion and constraints.
Return 0 for a hand moving through a stationary manipulated object, implausible interpenetration,
teleportation or motion disconnected from the operation, as well as a wrong operation/target,
no meaningful action, or evidence too weak to establish the interaction and its response.
For camera-only Read actions, require coherent viewpoint change and access to the specified
region; do not require an object manipulation that was not requested.

Use action_description and the operational part of success_criterion. Ignore remembered-content
clauses (e.g. an empty drawer or a red block): opening the right drawer/lifting the right cup can
pass even if the revealed contents are missing or wrong. Content restoration belongs to Memory.
Allow small timing, trajectory and viewpoint differences when the same interaction and access
are achieved with plausible physical response. Do not require exact GT motion or infer forces,
friction or material parameters from sparse frames. A clip already at the final view can establish
success only when that view itself is the requested criterion; it cannot prove required motion.

Keep labels independent: an incorrect action may still produce a clear view; a correct action may
produce an unreadable view. Evaluate interaction physics without requiring correct remembered contents.
Each decision must include specific GT/Gen frame evidence. Read frames support Observability;
any frame can support Interaction Physics. If uncertain, conservatively return 0 and describe the
limitation. Gate passes only when both labels are 1.
