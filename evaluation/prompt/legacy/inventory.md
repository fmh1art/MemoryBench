TASK: independently inventory each requested readable frame of ONE video. The other video's inventory is intentionally not supplied.
readable is whether the reading target can be observed, independent of whether it contains any objects. Empty visible target => readable=true, objects=[], relations=[]. An occluded/out-of-view target => readable=false, objects=[], relations=[].
Count each separate physical object in the target region once, with an opaque local ID o1, o2, ... . IDs are local to each frame and must not encode position.
Use ordinary specific singular category names (cup, bottle, stapler, etc.). Do not combine unlike categories into generic 'item'. Do not list the target container itself, its handles, surrounding furniture, background objects, or the actor's hands as contents.
For each object record visible appearance: color, shape, pattern, markings, material. Use null for an unobservable attribute. These fields must contain NO spatial placement, pixel coordinates, or mutable states such as empty/full. Do not copy remembered contents from a different frame.
Record only GT/visible applicable state facts using states=[{name,value}]. Examples: fill_level=empty/full, lid=open/closed, assembly=intact/disassembled. Color/category/position are not states. Unknown/not-visible/not-applicable states must be omitted, not guessed. Each independent state is listed once.
Record only observable spatial relations in the target region as subject_id, predicate, object_id. object_id is another object ID or 'target' for the physical container/region.
Examples: (o1, inside_left_compartment, target), (o1, on_top_of, o2), (o1, separated_from, o2). Use intrinsic compartments/landmarks and physical relations, NOT absolute pixels or image-left/image-right that changes under a camera turn. If a stable relation cannot be determined in this frame, omit it.
Do not duplicate the same fact as its inverse or under multiple synonyms. Appearance, state, and relation facts belong in their respective fields only.
Be attentive to objects disappearing, appearing, changing color/shape, or changing state between frames. Never fill an empty frame using another frame's objects.

