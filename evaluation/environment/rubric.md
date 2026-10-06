Evaluate surrounding world preservation in this single-frame pair. I_gt is the GT reference; I_gen is the generated
video. **Explicitly ignore memory_target objects, their appearance, their counts and their placement.**
Conceptually exclude their occupied regions in both images; do not let a missing/extra target improve
or degrade environment scores. If the target is an entire region (e.g. drawer contents), exclude those
contents, while keeping enclosing furniture and room context unless they too are explicitly targeted.
Do not require literal image masks. Explain the excluded target in evidence when the boundary matters.
If target exclusion leaves no useful background, use not_applicable/null, not an invented score.

Compare common visible surroundings using stable correspondences. Do not punish small camera shifts,
perspective, framing or natural occlusions. Different unseen portions cannot establish background
hallucination. Judge state preservation rather than sharpness, beauty, detail density or realism.

**Scene Structure: room layout, major background entities, boundaries and spatial organization.**
- 5: Room boundaries, major fixtures/furniture and their organization match GT.
- 4: Minor local structural drift with the overall layout and major entities preserved.
- 3: One clear moderate layout/entity error; most room structure remains recognizable.
- 2: Several major layout/entity changes or a substantial region rebuilt incorrectly; some structure survives.
- 1: Only a few landmarks resemble GT; most background structure is lost or substituted.
- 0: No meaningful surrounding structural correspondence, or a fundamentally different environment.
Examples of structural failures: changed shelf compartments, missing doorway, moved major furniture,
new walls or merged fixtures. Assess only non-target entities in shared visible regions.

**Background Appearance: non-target visual content, surfaces and background object appearance.**
- 5: Background content, colors, textures, materials and distinctive features match GT.
- 4: Small local surface/detail differences; surrounding visual identity is intact.
- 3: Noticeable moderate content/appearance changes, with the majority of reference surroundings retained.
- 2: Broad texture/material/color substitutions or several major background-content changes; some matches survive.
- 1: Only small isolated background patches resemble GT; most surrounding visual content differs.
- 0: Surrounding appearance is effectively replaced or has no meaningful reference correspondence.
Separate actual content/surface drift from modest illumination/exposure differences. New unrelated
background entities count here and may also change Scene Structure when structurally significant.

**Lighting / Style (optional).** Score only when lighting_style=true in the case. Otherwise return
not_applicable, score=null, with evidence saying this optional metric is disabled.
- 5: Reference illumination character, exposure, palette and rendering style preserved.
- 4: Slight exposure/color-temperature/style drift without changed scene character.
- 3: Noticeable moderate illumination or style change; overall character remains recognizable.
- 2: Large lighting/palette/style shift affecting much of the background.
- 1: Barely retains reference lighting/style; predominantly a different visual presentation.
- 0: Completely contradictory illumination/style (e.g. day-to-night or realistic-to-cartoon replacement).
This is fidelity to GT, never a preference for a prettier style. Do not count geometry or target defects
as lighting/style errors. Do not infer a lighting change solely from looking toward a different light source.

Return scene_structure, background_appearance and lighting_style as Metric records. Cite the supplied memory_frame.frame_index
and specific non-target landmarks/surfaces. Use insufficient_evidence/null when framing or visibility
prevents a reliable comparison of an existing background fact; keep that distinct from real scene loss.
