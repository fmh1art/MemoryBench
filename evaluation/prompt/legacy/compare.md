TASK: adjudicate the program's explicit appearance, state, and relation CHECK ITEMS for this normalized-progress GT/generated frame pair.
The videos may have different timestamps, action speeds, or small camera/lighting differences. Compare the observed physical contents of the specified target, not whole images.
Object identity matches are FROZEN. Never reassign a matched object using its position. Identify the matched objects using the provided category/appearance records.
For appearance: judge the matched object's visible GT color, shape, pattern, markings and material as a whole. Allow perspective/lighting changes that preserve identity; incorrect visible identity features fail. If the generated view cannot establish the relevant match, use unobservable, not match.
For state: judge this exact GT state fact. If the generated object is missing, occluded, blurred beyond recognition, or its applicable state cannot be established, use unobservable/mismatch. Do NOT declare a GT fact inapplicable based on the generated frame.
For relation: judge the exact GT physical relation using the frozen object mapping and 'target' as the physical reference container/region. An exchange of positions between distinguishable objects must fail the corresponding relations. Do not use absolute image coordinates. Do not reinterpret the target's intrinsic compartments to agree with the generated image.
Return EXACTLY one item_id, verdict (match/mismatch/unobservable), and evidence-based reason for EVERY requested item. The program counts match as 1 and the other two as 0. Missing-object penalties are enforced separately by the program and cannot be waived here.
Do not add or remove GT denominator items. Do not return any numerical scores.

