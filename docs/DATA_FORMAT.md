# Evaluation case data contract

Release version: `memorybench-data-v1`. Existing `case.json` records remain intact;
`data/eval_case.schema.json` specifies their shared fields. A batch-qualified
`case_path` in `metadata/cases.jsonl` is the globally unique identifier: short
`evaluation_case_id` values need not be unique across batches.

## Release structure

```text
README.md
archives.json                 archive byte sizes and SHA-256
SHA256SUMS
metadata/cases.jsonl           one record per case, 38 records
metadata/cases/<case_path>/case.json
metadata/eval_case.schema.json
archives/office_and_annotations.tar.gz
archives/memory_home_yijie_0924_v1.tar.gz
archives/quest_20260921_semantic13_v1.tar.gz
```

Extract all archives into the same directory. They produce `evaluation_cases/`,
with 5 cases directly under its root, 11 under memory_home_yijie_0924_v1, and 22
under quest_20260921_semantic13_v1. Each batch retains manifest.json and annotations.

## Per-case assets

| Path | Meaning |
| --- | --- |
| case.json | IDs, boundaries, counts, input contract, status and model alignment |
| annotation.json | original annotation snapshot |
| memory_annotation.json | scoring target, gate criterion and state facts |
| input/ | observed RGB/anchor and permitted future actions/pose; home also fixed_memory.pt |
| ground_truth/future.mp4 | future RGB reference; never input conditioning |
| ground_truth/evaluation.mp4 | scored RGB interval |
| frame_mapping.npz | model/source frame correspondence and padding masks |
| context/ | model-specific preprocessing assets, including GT for packing |
| packed/index.json | model-specific sample index; not a universal model input format |
| packed/samples/ | precomputed tensors where present |
| step_annotation.json, preview.jpg | annotation evidence and review assets |
| *validation.json | existing asset/compatibility checks, not inference quality claims |

Optional assets differ by batch. In particular, fixed-memory home inputs must not
be re-encoded as ordinary prefix history. Read `input_contract`, `status` and
`inference_status` before launching inference.

## Coordinates, intervals and temporal alignment

Frame indices are zero-based; all ranges and slices are left-closed/right-open.
`marks` has prefix_start, t, eval_end and may have prefix_end, action_end, eval_start.
For `keyframes`, prefix is [prefix_start,t) (or ends at explicit prefix_end), the
future is [t,eval_end), and stable scoring is [eval_start,eval_end).
For `prefix_generation`, prefix is [prefix_start,prefix_end), future is [t,eval_end),
and a gap before t can be excluded. action_end may be null; do not invent an action
gate boundary. The full annotated generation may be the evaluation interval.
The last observed anchor is prefix_end-1, or t-1 when prefix_end is absent.

`prefix_frame_count`, `generation_frame_count`, `evaluation_frame_count` describe
exported intervals. `prediction_future_slice` and `prediction_evaluation_slice`
index a generated video **including its observed anchor**; remove anchor and tail
padding using these slices. `model_anchor_frame`, `inference_start_clip`,
`generation_clips`, `padding_left_frames`, `padding_right_frames` describe the
32-frame MemWorld section layout. Do not equate source indices with model indices.

Where present, input/aligned_conditions.npz stores source_frame_indices and
sensor_timestamp_ns; left_hand_positions_ref/right_hand_positions_ref have shape
(N,21,3), with validity masks and NaN for missing observations. c2w_opencv and
c2w_memworld have shape (N,4,4); translations are meters. Preserve the source
coordinate convention. MP4 playback FPS is not the sensor clock. Consult each
batch's frame_mapping and original README for its specific timeline fields.

Future RGB in ground_truth/context or packed targets is reference material only;
it must never become observed memory or model conditioning. Future hand actions
and head pose are permitted only as stated in input_contract.

## Portability and validation

Historical directory/packed_index/source paths can refer to the producing machine.
Use `<download_root>/evaluation_cases/<case_path>` as the operational directory
and its `packed/index.json` as the index. The batch loader already rebases these
two top-level paths. Rebuilding from raw data requires separately supplied WMData.
The release does not imply that every historical model-specific tensor embeds
only relative paths. Internal packed/context links are relative and external
selected-video links are exported as actual files.

Validate metadata using `python data/validate.py --root data/evaluation_cases`.
Archive SHA-256 verifies bytes; metadata validation checks schema, IDs, boundaries,
counts, slices, required assets and input-vs-GT conditioning. Existing validation
records are preserved rather than replaced with fabricated successful results.
