TASK: for EVERY requested frame, judge only whether the specified reading action has reached its observable outcome and whether the correct physical target region is sufficiently visible.
Use action_completed and target_visible booleans, with a short evidence-based reason.
Opening a drawer/box: its opening must provide visual access to the relevant interior. Merely seeing its outside does not pass.
Turning the view: the camera must show the intended region sufficiently to inspect it. Do NOT require a drawer/box to open.
Other reading actions: use the supplied visibility criterion.
Judge physical target identity from its structure and layout. Wrong drawer/box/region does not pass.
Target visibility means the area in which content could be inspected is observable, even when it is EMPTY. Never use object identity/count/presence/correctness to decide either boolean.
If a visible target contains a missing, additional, deformed, or incorrect object, it can still pass BOTH booleans.
Background differences, minor viewpoint changes, and faster/slower action do not fail the gate.
Every supplied image corresponds to exactly one requested FRAME label from ONE video, in that order. There is no extra reference image. Use the physical target description to identify the region; do not assume the action has succeeded because it is described as an opening action.
Label each supplied frame independently; do not assume a hidden frame has the contents of a visible frame.
