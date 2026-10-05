Judge evaluation conditions once, returning two independent binary decisions. Do not score memory.
I_gt is the GT reference; I_gen is the generated video, not an aesthetic competitor.

**Observability (0/1).** Return 1 when the reference memory state and the corresponding generated
Read region are visible at useful scale, with enough evidence to compare inventory, target attributes,
spatial placement and surrounding scene. Useful partial occlusion is acceptable when relevant facts
remain recoverable across the Read frames. Return 0 when camera views, blur, darkness, obstruction,
small scale or missing Read coverage prevent reliable core comparison. Check both images.
Crucially, a visible empty Gen region where GT contains objects is observable: return 1 if the
absence can be established. Do not require the remembered object to exist in Gen to pass.

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
