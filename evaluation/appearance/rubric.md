Score two aspects of remembered target objects in this single-frame pair. I_gt is the GT reference;
I_gen is the generated video. Judge restoration fidelity, never beauty or photorealism.

TARGET SCOPE: appearance_matching.objects is the Inventory-derived list of semantically matched
GT/Gen target groups and matched_count instances. Evaluate ONLY these matched instances.
Exclude missing GT instances, unmatched Gen additions and unrelated objects; do not reduce
Appearance for omissions/count errors. A color/material/shape change is still an appearance error
when identity is established. Use Inventory evidence to locate pairs, then verify attributes in
these two images: semantic annotation is not proof of appearance fidelity. Do not cherry-pick good
matched objects; judge all assessable matched instances. If matching is visibly inconsistent or
GT attributes cannot be resolved, return insufficient_evidence/null with concrete evidence.
For an established matched target, if GT clearly shows the attribute but generated blur/fusion/
deformation makes that attribute unreadable in an exposed Gen region, score that submetric 0.
Natural occlusion and viewpoint limitations remain null; do not infer a defect from occlusion alone.
No matched objects means not_applicable, never zero or five. The runner derives that result without
a VLM query; if supplied an empty list, apply the same rule. Only this pair is available.

**Visual Appearance: color, texture, material, pattern and distinctive markings.**
- 5: All assessable identity-bearing attributes match GT in this pair; negligible rendering noise.
- 4: Small hue/texture/detail drift on a limited area; distinctive identity remains intact.
- 3: Clear moderate attribute errors, e.g. a noticeable color shift or simplified pattern, while
  substantial target appearance remains faithful.
- 2: Several major attributes differ or a salient marking/material is lost; only coarse appearance matches.
- 1: Only a weak visual resemblance remains; dominant colors/materials/patterns mostly contradict GT.
- 0: Matched target exists, but effectively all assessable identity-bearing attributes are wrong.
Compare surface properties under plausible illumination, not raw pixel brightness. Different shadows,
exposure or rendering sharpness alone are not surface-attribute failures. Material identity may use
consistent highlights in this pair, but do not infer invisible texture detail from the description.

**Geometry / Shape: shape, proportions, parts and structural integrity.**
- 5: Silhouette, proportions and visible parts/structure faithfully match GT.
- 4: Slight contour or proportion error without changed object structure.
- 3: Noticeable deformation or proportion drift, while major parts and topology remain recognizable.
- 2: Large deformation, major missing/added part, or incorrect assembly; coarse object form survives.
- 1: Severe collapse/warping; only a small fragment of expected structure remains recognizable.
- 0: Matched target exists, but its geometry bears no meaningful correspondence to the reference.
Account for foreshortening, perspective, partial occlusion and small camera differences; do not use
screen-space size differences alone as proof of geometry error. Treat fused/doubled parts as geometry
errors when object count/identity is still clear.

Return visual_appearance and geometry_shape as Metric records. Use insufficient_evidence/null for
attributes or shape unresolvable because of GT visibility or natural occlusion/viewpoint, with a
visibility explanation. Confirmed Gen degradation with clear GT and an exposed Gen region is 0;
not_applicable/null
when no matched target instance exists or the matched target lacks the relevant visual property. Each record needs single-pair evidence specifying matched or
changed attributes/parts and the affected targets.
