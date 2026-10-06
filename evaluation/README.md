# Memory Evaluation

A reference-based evaluation of the world state recovered after a memory Read.
Each case needs **GT video + Gen video + a short case description**. Gate uses one query with two 12-frame sheets. When both binary labels pass, Memory uses four
independent GT/Gen single-frame pairs from the middle 60% of the shared video. Each enabled
dimension gets one query per pair; four scores are averaged. Environment is disabled by default.

## Files

```text
evaluation/
  gate/          schema.py + rubric.md: Observability, Action Following (binary)
  inventory/     schema.py + rubric.md + scoring.py: semantic annotations → Python metrics
  appearance/    schema.py + rubric.md: Visual Appearance, Geometry / Shape (0–5)
  spatial/       schema.py + rubric.md: Object Localization, Relative Spatial Relation (0–5)
  environment/   schema.py + rubric.md: Scene Structure, Background Appearance,
                                       optional Lighting / Style (0–5)
  common.md      Gate contact-sheet reference/evidence rules
  memory_common.md single-pair reference/evidence rules
  schema.py      case, evidence and metric contracts
  video.py       aligned Gate sheets and four independent Memory image pairs
  judge.py       asynchronous VLM client, validation, retries and call records
  scoring.py     normalization and aggregation
  run.py         single/batch CLI
  cases.example.json
```

The rubrics are intentionally separate from orchestration code. Every judge sees the shared
rules, its dimension rubric, the case/sampling metadata, the output JSON Schema, and exactly
two separately labeled images: **I_gt is the GT reference; I_gen is the generated video**.
GT defines correctness. No judge compares which video is more attractive.

## Run

Python 3.11+. Install from the repository root:

```bash
python -m pip install -r requirements-eval.txt
# Create cases.json using evaluation/cases.example.json as the format example.
# The example paths are placeholders, not bundled test videos.
export MEM_BENCH_API_KEY='your-service-key'
python -m evaluation.run \
  --cases cases.json \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model your-vlm \
  --output evaluation_results/run_001 \
  --concurrency 8 --case-workers 2
```

`--cases` is a nonempty JSON array; a single element runs one case. Video paths are relative
to that JSON file. IDs must be unique and safe directory names. Output must be a new directory.
Credentials come from the environment and are never written into configuration snapshots.
The service must support multimodal chat-completions, inline JPEGs and `response_format=json_object`.
Responses are validated locally against Pydantic schemas, including integer scores and evidence frames.
Unselected submetrics are absent from the response schema and do not affect averages.
`--extra-body provider.json` allows provider-specific options, e.g.
`{"chat_template_kwargs":{"enable_thinking":false}}`; it cannot replace prompts, model or JSON mode.
The old TOML configuration/old evaluation commands are no longer used.

`--concurrency` bounds all in-flight VLM requests across cases, including Gate and retries.
`--case-workers` bounds simultaneous cases and video preparation. Within a passing case, the four pairs and their enabled dimension Judges run concurrently.
Defaults are Inventory, Appearance and Spatial: 1 Gate + 4×3 Memory requests per passing case;
transport, JSON or schema failures allow up to `--retries` additional attempts (default 2).
Every attempt is recorded. HTTP 4xx errors other than 408/429 fail immediately. An evaluation error
sets the case status to `error` and makes the CLI exit 1 after other cases finish; gate rejection or
insufficient visual evidence is a valid evaluation outcome and does not cause exit 1.

## Select dimensions or submetrics

Default: Inventory, Appearance and Spatial; Environment makes no calls and contributes no score.
Choose whole dimensions with `--dimensions`, or exact submetrics with `--metrics` (mutually exclusive):

```bash
# Append to the run command to enable Environment explicitly.
--dimensions inventory appearance spatial environment

# Judge only these two submetrics, one query per dimension per pair.
--metrics appearance.visual_appearance spatial.relative_spatial_relation
```

Available submetrics: inventory.object_presence, inventory.object_count, inventory.f1;
appearance.visual_appearance, appearance.geometry_shape;
spatial.object_localization, spatial.relative_spatial_relation;
environment.scene_structure, environment.background_appearance, environment.lighting_style.
Inventory always needs semantic object counts; Python computes its diagnostics and averages only
the selected Inventory submetrics. Lighting/style additionally needs `lighting_style=true` in the
case; otherwise it remains not_applicable. Selection is recorded in run.json and every result.

## Case description and timeline

Required fields: case_id, gt_video, gen_video, memory_target, action_description,
success_criterion, read_start_seconds. The target should unambiguously name the object(s) or
inventory region. Describe the future action and its visible success condition; do not disclose
expected counts/appearance as a substitute for reference-image evidence.

Optional `memory_objects` is the explicit list of objects tested by this Read. When nonempty,
Gate and Memory Judges focus only on that list; unrelated items are not an alternative source of
positive evidence. If omitted, the case loader reads `metadata.memory_objects` from the matching
GT case's adjacent `case.json`, then `description.json`. It validates the annotation's case identity,
records memory_objects_source, and preserves explicit caller overrides. For generic videos or
annotations without an object list, memory_target remains the fallback scope. Names/descriptions
identify what to inspect; GT images still establish actual presence, counts and placement.

Optional read_end_seconds defaults to the shared clip end. The inclusive Read interval is the
post-action evaluation interval, not the historical memory-writing interval. Earlier frames show
action context. Provide clips containing the Read operation if success requires proving motion.
Existing annotations can supply memory_target, action_description and success_criterion, but the
Read time must be supplied on the paired video timeline. No hidden mapping from old case assets.

Optional gt_start_seconds and gen_start_seconds identify each video's local time corresponding
to experiment t=0. Defaults are both zero. Each video's local clock is decoded PTS minus its first
frame PTS. Offsets align known clip starts; **do not independently normalize video duration**.
If the clips share the same physical start, leave both offsets zero.

## Gate: two images, twelve aligned frames

1. Decode timestamps and determine the common interval from experiment t=0 to the earlier last frame.
2. Request 12 uniformly spaced times, including both endpoints, on this shared physical timeline.
3. In each video choose the nearest actual frame to each requested time. Check requested-to-actual
   errors and GT-to-Gen actual-time skew against `--max-skew-seconds` (default 0.1 s); violations are
   input errors, not memory failures. No interpolation or time warping.
4. Compose independent I_gt.jpg and I_gen.jpg, each 2 rows × 6 columns, in chronological order.
   Every tile shows frame number, shared requested t and actual decoded frame time. Resize with
   aspect ratio preserved and center-pad; never stretch. Default content tile is 480×320.
5. Pass both images to the single Gate query. Observability evidence cites Read tiles; Action
   Following may cite any tile. Both labels must be 1 before any Memory images or calls are made.

A timestamp request need not coincide exactly with a real frame at another frame rate; both the
small tolerance and actual times are recorded. A short/sparse clip can repeat the same real frame;
unique-frame counts expose this, without pretending those are independent observations. Video tail
outside the overlap is excluded and its duration recorded; align/trim inputs deliberately. If the
Read interval lies outside overlap or uniform sampling misses it, the case fails with an input error.
With only 12 views, unobserved temporal failures cannot be assessed; use clips covering the relevant
Read operation and state, and retain this sampling protocol when comparing models.

## Memory: four independent image pairs

Request frames at **20%, 40%, 60%, 80%** of the entire shared video interval used by Gate, including
the endpoints of its middle 60%. This is not an independently stretched per-video clock or the
middle 60% of the Read interval. Record requested and actual timestamps, decoded indices and whether
each sample lies in Read. The same alignment tolerance applies as for Gate.

Save eight standalone JPEGs. Each preserves the decoded image's aspect ratio; cap its longest side
at 1024 pixels without upscaling or stretching. No sheets, borders, timestamp overlays or additional
views are included in Memory queries. Timestamp/index labels are supplied as text metadata.

Judge each pair independently against GT with the enabled dimension rubrics. Memory evidence must
cite only that pair's index (1–4), a separate index space from Gate's sheet tiles (1–12). Never use
another pair to resolve occlusion or infer a hidden object. A global Gate pass does not guarantee
that every single sampled view can resolve every submetric. A sample before Read or with an
uninspectable target can therefore return insufficient_evidence, with its images and explanation
retained for review. Samples are never silently moved to a more convenient timestamp.

## Judgments and evidence

Gate outputs independent 0/1 labels. It passes only when both Observability and Action Following
are 1. A **clear, unobstructed and inspectable** empty Gen region can pass: confirmed absence is a
memory error. Blur, veiling or unresolved object boundaries cannot establish that the region is empty.
Observability must establish legibility of the requested targets, not just walls, shelf lines or
other unrelated objects. Sustained degradation that blocks target identity/count/placement gives 0;
a clearly visible wrong object/state remains observable. Brief motion blur is allowed when other
Read frames provide adequate target-specific evidence. A failed Gate has no
memory score, and no memory judges run. Report Gate pass rates separately from conditional memory scores.

Inventory returns mutually exclusive semantic object groups, reference/generated integer counts,
and evidence for this pair. It never returns numeric quality scores. Null counts or an unassessable
region leave Inventory unscored; do not silently drop unknown groups. Counts are physical visible
objects in this single snapshot. Attribute drift alone should not manufacture missing-plus-spurious
identities; no adjacent-view evidence or counting across timestamps is allowed.

Other metrics output `{status, score, evidence}`. Each rubric defines specific anchors for every
integer 0–5. `scored` requires an integer; `not_applicable` and `insufficient_evidence` require null.
Missing targets in clearly visible expected regions are actual scored failures. Spatial judges use
stable world anchors and tolerate small camera differences. Environment explicitly excludes memory
targets. Lighting / Style is disabled unless lighting_style=true; a disabled metric is not_applicable.

Every decision and metric, every inventory group, and the overall inventory assessment provide
frame-index evidence describing concrete GT/Gen facts. Gate timestamps are saved in sampling.json; Memory timestamps in memory_sampling.json. The
response validator rejects Memory evidence citing any pair other than the one supplied.

## Deterministic scores

For inventory groups i, let g_i and p_i be GT and Gen counts, and m_i=min(g_i,p_i):

| Quantity | Definition |
|---|---|
| Object Presence | Fraction of GT groups with p_i>0; semantic group presence, not instance recall |
| Object Count | Mean of 1−abs(g_i−p_i)/max(g_i,p_i), over the union of nonempty GT/Gen groups |
| Missing Objects | sum(g_i−m_i) |
| Spurious Objects | sum(p_i−m_i) |
| Precision | sum(m_i)/sum(p_i) |
| Recall | sum(m_i)/sum(g_i) |
| F1 | 2×sum(m_i)/(sum(g_i)+sum(p_i)) |
| Inventory normalized | Mean(Object Presence, Object Count, F1) |

These are macro group presence/count scores and instance-based precision/recall/F1. Extra groups
lower Object Count and F1 even if all GT groups remain present. Missing/spurious counts are diagnostic
counts and are not normalized by five. Inventory's equivalent raw score is 5×normalized and can be
fractional; it is computed by code, not an ordinal VLM score.

If both regions are visibly empty, Presence/Count/Precision/Recall/F1 all equal 1. If GT is empty but
Gen is not, Presence/Count/F1/Precision are 0 and Recall is 1. If GT is nonempty and Gen empty,
Presence/Count/F1/Precision/Recall are 0. These conventions make correct empty restoration score 1.

Other metric scores normalize as score/5. Within each pair, average selected applicable submetrics
within a dimension, then average selected applicable dimensions with equal weights.
At every averaging step, include only reliable `scored` values: skip insufficient_evidence and
not_applicable instead of assigning zero. Within a pair, average valid submetrics into dimensions
and valid dimensions into its Memory score. Across the four pairs, independently average valid
pair scores, dimension scores and submetric scores. A real score of zero remains in the denominator.
A metric, dimension or case with no valid scores stays null; preserve insufficient_evidence when
visibility is unresolved, otherwise not_applicable. Different means can have different denominators.
Persist scored_samples/total_samples, scored_frames/excluded_frames and original per-frame statuses;
Demo shows the effective frame count for each submetric. Failed API/schema calls remain evaluation
errors and are not silently excluded. Optional lighting/style participates only when enabled.

Batch summaries are macro means over cases with valid Memory scores, with per-dimension/per-metric
sample counts. Each dimension/submetric uses its own valid case count. Inventory ratios use cases with valid Inventory. Missing/spurious counts are computed per
frame, averaged over reliable Inventory frames per case, then summed across cases with valid Inventory in the summary;
this is explicitly labeled as a sum of per-case mean counts, not a count of distinct physical objects.
Gate pass rates use cases with a valid Gate result; errors before/during Gate are separately counted.
Always report total cases, gate_evaluated, gate_passed, scored_cases, insufficient_evidence_cases,
and error_cases alongside memory_score. Never replace excluded cases with zero or compare conditional
scores without their coverage.

## Outputs

```text
run_001/
  run.json                  protocol, model, execution settings and cases
  summary.json              Gate rates, conditional memory means and denominators
  drawer_demo/
    I_gt.jpg, I_gen.jpg      Gate-only 12-frame sheets
    sampling.json           Gate timeline, Read indices and coverage
    memory_sampling.json    four aligned timestamps (absent if Gate fails)
    memory_frames/          1_gt.jpg, 1_gen.jpg, ... 4_gt.jpg, 4_gen.jpg
    judges/
      gate.json             one binary Gate call, prompt/schema/usage/all attempts
      frame_1/              enabled dimension records for pair #1
        inventory.json      absent when Inventory is disabled
        appearance.json
        spatial.json
      frame_2/ ... frame_4/  independent calls; Environment only when enabled
    result.json             status, evidence-bearing judgments, derived scores/errors
```

Rubric text and response schemas are stored in every call record. Provider error bodies and base64
image payloads are not logged. Keep the videos and output artifacts with experiment records.

Run the focused regression tests with `python -m pytest tests/test_evaluation.py -q`.
They use synthetic videos and a mocked VLM service; no remote VLM or credentials are needed.

## Service limits and default temperature

For services that only accept their model's default temperature, add
`--use-default-temperature` to omit the temperature parameter. The default remains 0 for
other services. Connection establishment is limited to the smaller of 15 seconds and
`--timeout`; response reads retain the full `--timeout` so stalled connections do not hold
Judge slots for the entire generation timeout.

`--qpm` and `--tpm` enable global pacing inside the concurrency limit, including retries.
`--tokens-per-request` reserves a conservative input-plus-output token count (default 20000).
Requests are spaced by the greater of `60/qpm` and `60*tokens_per_request/tpm`; idle time does
not accumulate a burst allowance. Set the reservation after inspecting pilot-call usage,
including both contact sheets, prompt/schema and maximum output. Usage above the reservation
is reported as a configuration error and is not retried. Leave headroom below service limits,
especially if the same model quota is shared by other jobs.

For a dedicated gpt-5.6-terra quota of QPM=300 and TPM=2430000, a conservative starting point is:

```bash
python -m evaluation.run --cases cases.json --endpoint "$VLM_ENDPOINT" \
  --model gpt-5.6-terra --output evaluation_results/terra \
  --use-default-temperature --concurrency 32 --case-workers 8 \
  --qpm 270 --tpm 2187000 --tokens-per-request 20000 --max-tokens 8192
```

This starts at most about 109 requests/minute with the 20000-token reservation, retaining 10%
quota headroom. Concurrency controls latency overlap; it does not replace rate limits. Decrease
the reservation only after measured pilot usage justifies it.

## Local result viewer

Browse saved results without making new Judge calls. The viewer needs only Python's standard
library and serves the paired videos referenced by each `result.json`.

```bash
python -m evaluation.viewer.server \
  --results /home/fanmeihao/data/acwm/eval_results/_evaluation/terra_frames4_pilot10_20261006 \
  --port 8765
```

Open <http://127.0.0.1:8765>. Search cases or targets, filter by category/status, and sort by Memory
score. Each v2 case shows the four standalone image pairs and their individual scores, the valid-frame
average scores and raw submetric means. Select a frame to inspect its Inventory counts, 0–5
Judge scores and evidence. Disabled dimensions say so. Gate uses a separate 12-frame sheet tab;
Gate evidence indices are never treated as Memory pair indices. Legacy v1 outputs remain supported. Use the video tab
for synchronized playback/seek, or expand the sheets at their original resolution. Case URLs
include a shareable `#case_id` fragment. Gate failures and insufficient evidence retain their
status and null overall score.

On a remote server, forward port 8765 through SSH (`ssh -L 8765:127.0.0.1:8765 user@server`),
then open the same local URL. `--host` changes the bind address; the default is loopback.
Artifacts are read when the server starts; restart after replacing result files. Keep the
referenced GT/Gen videos available to enable playback. Run the viewer regressions with
`python -m pytest tests/test_evaluation_viewer.py -q`.
