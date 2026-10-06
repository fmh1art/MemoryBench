# Memory Evaluation

A reference-based evaluation of the world state recovered after a memory Read.
Each case needs **GT video + Gen video + a short case description**. There are five
logical VLM judgments: one Gate, followed by four concurrent memory judgments when Gate passes.
No detectors, trackers, per-frame judge loops, separate action clip, or model inference adapters.

## Files

```text
evaluation/
  gate/          schema.py + rubric.md: Observability, Action Following (binary)
  inventory/     schema.py + rubric.md + scoring.py: semantic annotations → Python metrics
  appearance/    schema.py + rubric.md: Visual Appearance, Geometry / Shape (0–5)
  spatial/       schema.py + rubric.md: Object Localization, Relative Spatial Relation (0–5)
  environment/   schema.py + rubric.md: Scene Structure, Background Appearance,
                                       optional Lighting / Style (0–5)
  common.md      shared reference/evidence rules
  schema.py      case, evidence and metric contracts
  video.py       aligned sampling and independent 2×6 images
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
`--extra-body provider.json` allows provider-specific options, e.g.
`{"chat_template_kwargs":{"enable_thinking":false}}`; it cannot replace prompts, model or JSON mode.
The old TOML configuration/old evaluation commands are no longer used.

`--concurrency` bounds all in-flight VLM requests across cases, including Gate and retries.
`--case-workers` bounds simultaneous cases and video preparation. Within a passing case, Inventory,
Appearance, Spatial and Environment run concurrently. By default each dimension gets one request;
transport, JSON or schema failures allow up to `--retries` additional attempts (default 2).
Every attempt is recorded. HTTP 4xx errors other than 408/429 fail immediately. An evaluation error
sets the case status to `error` and makes the CLI exit 1 after other cases finish; gate rejection or
insufficient visual evidence is a valid evaluation outcome and does not cause exit 1.

## Case description and timeline

Required fields: case_id, gt_video, gen_video, memory_target, action_description,
success_criterion, read_start_seconds. The target should unambiguously name the object(s) or
inventory region. Describe the future action and its visible success condition; do not disclose
expected counts/appearance as a substitute for reference-image evidence.

Optional read_end_seconds defaults to the shared clip end. The inclusive Read interval is the
post-action evaluation interval, not the historical memory-writing interval. Earlier frames show
action context. Provide clips containing the Read operation if success requires proving motion.
Existing annotations can supply memory_target, action_description and success_criterion, but the
Read time must be supplied on the paired video timeline. No hidden mapping from old case assets.

Optional gt_start_seconds and gen_start_seconds identify each video's local time corresponding
to experiment t=0. Defaults are both zero. Each video's local clock is decoded PTS minus its first
frame PTS. Offsets align known clip starts; **do not independently normalize video duration**.
If the clips share the same physical start, leave both offsets zero.

## Two images, twelve aligned frames

1. Decode timestamps and determine the common interval from experiment t=0 to the earlier last frame.
2. Request 12 uniformly spaced times, including both endpoints, on this shared physical timeline.
3. In each video choose the nearest actual frame to each requested time. Check requested-to-actual
   errors and GT-to-Gen actual-time skew against `--max-skew-seconds` (default 0.1 s); violations are
   input errors, not memory failures. No interpolation or time warping.
4. Compose independent I_gt.jpg and I_gen.jpg, each 2 rows × 6 columns, in chronological order.
   Every tile shows frame number, shared requested t and actual decoded frame time. Resize with
   aspect ratio preserved and center-pad; never stretch. Default content tile is 480×320.
5. Pass both images independently to each judge. Explicit Read frame indices identify which tiles
   supply memory evidence. Gate's Action Following may cite any frame; all other evidence is Read-only.

A timestamp request need not coincide exactly with a real frame at another frame rate; both the
small tolerance and actual times are recorded. A short/sparse clip can repeat the same real frame;
unique-frame counts expose this, without pretending those are independent observations. Video tail
outside the overlap is excluded and its duration recorded; align/trim inputs deliberately. If the
Read interval lies outside overlap or uniform sampling misses it, the case fails with an input error.
With only 12 views, unobserved temporal failures cannot be assessed; use clips covering the relevant
Read operation and state, and retain this sampling protocol when comparing models.

## Judgments and evidence

Gate outputs independent 0/1 labels. It passes only when both Observability and Action Following
are 1. A visible empty Gen region can pass: object absence is a memory error. A failed Gate has no
memory score, and no memory judges run. Report Gate pass rates separately from conditional memory scores.

Inventory returns mutually exclusive semantic object groups, reference/generated integer counts,
and evidence. It never returns numeric quality scores. Null counts or an unassessable region leave
Inventory unscored; do not silently drop unknown groups. Counts describe the latest reliably inspectable shared Read state (identified in evidence),
with adjacent views supporting identity/occlusion; they are not summed frame appearances. Observable
Gen disappearance/duplication is documented as drift rather than dismissed as visibility uncertainty. Attribute drift alone should not manufacture missing-plus-spurious identities.

Other metrics output `{status, score, evidence}`. Each rubric defines specific anchors for every
integer 0–5. `scored` requires an integer; `not_applicable` and `insufficient_evidence` require null.
Missing targets in clearly visible expected regions are actual scored failures. Spatial judges use
stable world anchors and tolerate small camera differences. Environment explicitly excludes memory
targets. Lighting / Style is disabled unless lighting_style=true; a disabled metric is not_applicable.

Every decision and metric, every inventory group, and the overall inventory assessment provide
frame-index evidence describing concrete GT/Gen facts. The paired timestamps for those indices are
saved in sampling.json. The response validator rejects memory evidence citing pre-Read frames.

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

Other metric scores normalize as score/5. Average applicable submetrics within each dimension;
average applicable dimensions with equal weights for the case memory score. Optional lighting/style,
when enabled, participates equally within Environment. `not_applicable` is excluded with its status
retained. Any `insufficient_evidence` metric makes the case memory score null; any failed judge makes
it an evaluation error. Partial dimension results remain available for inspection, but cannot inflate
the headline score by selecting only easy observations.

Batch summaries are macro means over complete `scored` cases only, with per-dimension/per-metric
sample counts. Inventory ratios use the same cases; missing/spurious totals are also reported.
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
    I_gt.jpg, I_gen.jpg      independent image inputs
    sampling.json           shared/actual timestamps, Read indices, skew and coverage
    judges/
      gate.json             prompt/schema/context, raw response, parsed output, usage, attempts
      inventory.json        same; absent if Gate fails
      appearance.json
      spatial.json
      environment.json
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
  --results /home/fanmeihao/data/acwm/eval_results/_evaluation/terra_full_20261006 \
  --port 8765
```

Open <http://127.0.0.1:8765>. Search cases or targets, filter by category/status, and sort by Memory
score. Each case shows the two timestamp-aligned 12-frame sheets, five normalized score cards,
raw 0–5 submetric scores, Inventory counts/missing/spurious totals, Gate labels and expandable
Judge evidence. Click an evidence frame number to open the paired frame crops. Use the video tab
for synchronized playback/seek, or expand the sheets at their original resolution. Case URLs
include a shareable `#case_id` fragment. Gate failures and insufficient evidence retain their
status and null overall score.

On a remote server, forward port 8765 through SSH (`ssh -L 8765:127.0.0.1:8765 user@server`),
then open the same local URL. `--host` changes the bind address; the default is loopback.
Artifacts are read when the server starts; restart after replacing result files. Keep the
referenced GT/Gen videos available to enable playback. Run the viewer regressions with
`python -m pytest tests/test_evaluation_viewer.py -q`.
