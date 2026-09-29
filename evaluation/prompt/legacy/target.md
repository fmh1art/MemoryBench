TASK: identify the physical reading target and the required reading action in the GT overview.
Use any supplied target description/action/criterion as constraints. Do not change an explicitly supplied action type.
read_action must be open_drawer, open_box, turn_view, or other.
For opening actions, the outcome is access to the target interior. For turn_view, the outcome is that the view reaches the specified physical region and makes it observable; no container needs to open.
description must identify the physical container/region using stable physical appearance and layout, NOT the identity, presence, color, or position of its contents.
visibility_criterion must refer only to reading-action completion and sufficient visibility of the target region. It must NEVER require any remembered object to be present or correct.
For a region already visible when this clip begins, a completed reading outcome can be observable without showing the preceding motion.
If multiple plausible reading targets remain and the supplied constraints do not distinguish them, set unambiguous=false. Do not select whichever target appears easiest to score.
Return Target JSON with unambiguous, description, read_action, visibility_criterion, reason.

