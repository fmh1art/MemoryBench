Score target placement and relations in this single-frame pair. I_gt is the GT world-state reference;
I_gen is the generated video. Do not score aesthetic composition. Do not over-penalize camera error.

Reason in scene coordinates using stable anchors: drawer interior, shelf compartment, tabletop,
room landmarks. Small translation, rotation, zoom, focal-length and timing differences are acceptable
when the same world placement is preserved. Use absolute/normalized image location only for genuinely
comparable camera views; otherwise compare location relative to the same stable scene anchors.
Do not align independently moving targets to hide placement errors, infer hidden depth from one
ambiguous image, or interpret camera motion as object motion. A gross failure to reach the Read view
belongs to Gate; residual camera differences alone should not lower spatial scores.

**Object Localization: target position relative to its expected region/anchors.**
- 5: Correct region and anchor-relative position, allowing plausible camera/projection differences.
- 4: Small local displacement within the correct region; reference placement is essentially preserved.
- 3: Clear moderate displacement within the correct compartment/surface; broad location is retained.
- 2: Large displacement or wrong nearby compartment/surface, with some coarse location correspondence.
- 1: Only the general scene vicinity matches; target is placed in a substantially wrong location.
- 0: Target absent from a clearly visible expected region, or no meaningful reference location is restored.
Normalize displacement to the relevant container/surface extent or nearby stable entities, not arbitrary
pixel distance. Distinguish evidence of an incorrect location from an unseen expected location.

**Relative Spatial Relation: inside/on, left/right, front/behind, containment, support and ordering.**
Evaluate only reference relations involving the memory target and a clearly identified other entity.
Separate world relations from view-relative relations; left/right requires a comparable view or a
specified stable reference orientation. Use all relevant visible relations, not a single easy pair.
- 5: All assessable reference relations preserved.
- 4: Nearly all preserved; a minor offset weakens one noncritical relation without reversing support/containment.
- 3: A salient relation is wrong/ambiguous from displacement, or several minor relations drift; most survive.
- 2: Multiple salient relations wrong, including major ordering/support/containment error; some survive.
- 1: Almost all reference relations fail; only a weak incidental relation survives.
- 0: No required relation restored, or all required targets are absent from visible expected regions.
Prioritize categorical topology (inside vs outside, on vs floating, correct compartment) over tiny
metric offsets. Contradicting one essential containment/support relation can justify a major penalty
when it is the defining memory fact; explain its significance in evidence.

Return object_localization and relative_spatial_relation as Metric records with single-pair evidence naming
targets, anchors and relations. If GT defines no assessable relation, use not_applicable/null for that
metric. If an existing fact cannot be resolved because of visibility/projection, use
insufficient_evidence/null. Missing targets in observable regions are scored failures, not exclusions.
