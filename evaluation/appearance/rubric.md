Score two aspects of remembered target objects in this single-frame pair. I_gt is the GT reference;
I_gen is the generated video. Judge restoration fidelity, never beauty or photorealism.

Match target identities semantically before comparing attributes. For multiple targets, score
across all relevant targets, considering extent and salience; do not select only surviving good
objects. A wholly missing target in a clearly visible region receives 0 on both metrics; partial
loss reduces scores according to how much reference content is lost. Keep failures attributable:
extra unrelated objects do not directly lower Appearance unless they corrupt/replace the target.
Only this pair is available. Explain viewpoint/occlusion uncertainty without assuming other views.

**Visual Appearance: color, texture, material, pattern and distinctive markings.**
- 5: All assessable identity-bearing attributes match GT in this pair; negligible rendering noise.
- 4: Small hue/texture/detail drift on a limited area; distinctive identity remains intact.
- 3: Clear moderate attribute errors, e.g. a noticeable color shift or simplified pattern, while
  substantial target appearance remains faithful.
- 2: Several major attributes differ or a salient marking/material is lost; only coarse appearance matches.
- 1: Only a weak visual resemblance remains; dominant colors/materials/patterns mostly contradict GT.
- 0: Target absent, substituted with unrelated appearance, or effectively all assessable attributes wrong.
Compare surface properties under plausible illumination, not raw pixel brightness. Different shadows,
exposure or rendering sharpness alone are not surface-attribute failures. Material identity may use
consistent highlights in this pair, but do not infer invisible texture detail from the description.

**Geometry / Shape: shape, proportions, parts and structural integrity.**
- 5: Silhouette, proportions and visible parts/structure faithfully match GT.
- 4: Slight contour or proportion error without changed object structure.
- 3: Noticeable deformation or proportion drift, while major parts and topology remain recognizable.
- 2: Large deformation, major missing/added part, or incorrect assembly; coarse object form survives.
- 1: Severe collapse/warping; only a small fragment of expected structure remains recognizable.
- 0: Target absent or geometry bears no meaningful correspondence to the reference.
Account for foreshortening, perspective, partial occlusion and small camera differences; do not use
screen-space size differences alone as proof of geometry error. Treat fused/doubled parts as geometry
errors when object count/identity is still clear.

Return visual_appearance and geometry_shape as Metric records. Use insufficient_evidence/null for
attributes or shape genuinely unresolvable in GT/Gen, with a visibility explanation; not_applicable/null
only if GT has no target object to compare. Each record needs single-pair evidence specifying matched or
changed attributes/parts and the affected targets.
