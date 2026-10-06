# Memory Evaluation v3

See [Eval Formulation](formulation.md) for the complete Chinese formulation and null/zero rules.

Gate decides whether a case is valid. Memory measures restoration of remembered objects.
Report **Gate Coverage, Inventory, Appearance, Spatial, Memory Score** separately.
Gate rejection contributes zero to the headline case/benchmark score; memory submetrics are not
judged and stay uncomputed. Environment and inventory error breakdowns are diagnostics.

## Structure

```text
evaluation/
  gate/            binary Observability + Action Following schema and rubric
  inventory/       semantic count annotations, instance matching and Python formulas
  appearance/      matched objects only: appearance and shape rubrics
  spatial/         placement and relations rubrics, tolerant of camera differences
  environment/     optional background diagnostics, excluding memory targets
  common.md        Gate reference/evidence instructions
  memory_common.md standalone pair reference/evidence instructions
  video.py         aligned Gate sheets and eight standalone Memory pairs
  judge.py         bounded async requests, validation, retries and call records
  scoring.py       weighted dimensions, temporal averages, geometric mean, summary
  run.py           single/batch evaluation
  viewer/          read-only demo: videos, images, scores and evidence
```

## Inputs and sampling

A JSON list supplies case_id, gt_video, gen_video, memory_target, action_description,
success_criterion and read_start_seconds. Optional memory_objects names the specific Read targets;
matching GT case.json metadata is loaded automatically when that list is absent. Annotations set
scope, never replace visual evidence. GT must cover the same Read operation as Gen; the benchmark
reference is ground_truth/future.mp4, not its shorter evaluation-only clip.

Both timelines use decoded PTS and explicit gt_start_seconds/gen_start_seconds offsets. Use their
shared duration, never independently stretch time. Reject misaligned frames beyond
--max-skew-seconds (default 0.12), missing Read coverage or invalid clips.

Gate uses one query: two independent 2×6 sheets with 12 uniformly sampled, aligned frames.
Timestamp every tile. Resize and pad to preserve aspect ratio; never stretch.
Each Judge receives I_gt labeled GT REFERENCE and I_gen labeled GENERATED VIDEO.
GT defines expected state, not aesthetic preference. All judgments require frame evidence.

Memory uses **eight aligned timestamps uniformly spanning 10%–90% of the shared video**, including
both endpoints. These are not samples of a separately stretched Read interval. Each query receives
exactly two standalone images, no collage or overlays; cap the longest side at 1024 without
upscaling/stretching. Record requested/decoded timestamps and whether each lies in Read.
A frame before Read or with unresolvable target facts may be insufficient_evidence.

## Judge order and scope

1. Gate returns independent Observability and Action Following labels in one query.
2. If either label is zero, stop: status=gate_failed, memory_score=0; do not judge Memory submetrics.
3. For every Memory pair, Inventory and Spatial run concurrently.
4. Appearance waits for Inventory's semantic matching, then evaluates only matched instances.
   Missing GT targets and unmatched additions do not lower Appearance. Partial matching evaluates
   all matched instances, not just the best ones. No confirmed match gives not_applicable/null;
   unresolved matching gives insufficient_evidence/null. These empty-match results are derived
   from Inventory and recorded without an unnecessary Appearance VLM call.
5. Optional Environment runs concurrently and remains outside the core Memory score.

Observability evaluates generated visual quality: persistent severe blur, ghosting, artificial
veiling or scene-wide corruption. A clear empty/wrong scene or natural occlusion does not fail it.
Action Following checks the correct physical operation/target-region access. Ignore remembered
contents in a success criterion: opening the correct drawer still succeeds if its contents differ.

Selecting only Appearance still requires Inventory annotation for matching, but Inventory scores
are not added to that selected score. Eight pairs and separate cases run concurrently, bounded by
one shared request semaphore and QPM/TPM pacer. Every actual attempt, including retries, is recorded.

## Formulas

For case i, g_i = observability_i × action_following_i. Gate Coverage = sum(g_i)/N.
Do not multiply this coverage into the headline Memory score.

VLM identifies mutually exclusive semantic object groups with GT/Gen counts. Python matches
interchangeable instances within each group: TP=sum(min(gt_count, gen_count)). GT=sum(gt_count),
Gen=sum(gen_count), FN=GT−TP, FP=Gen−TP. Stable identity may survive changed color, shape or placement;
an unrelated substitution is an unmatched identity, not an appearance-only error.

| Dimension | Normalized score per frame |
|---|---|
| Inventory | 0.7 × presence + 0.3 × count |
| Presence | TP / (TP + FN), i.e. instance recall, not fraction of object groups |
| Count | 1 − abs(Gen − GT) / max(Gen, GT, 1), using total counts |
| Appearance | 0.5 × visual/5 + 0.5 × shape/5, matched instances only |
| Spatial | 0.4 × localization/5 + 0.6 × relation/5 |

**Diagnostics only:** Missing Rate=FN/GT, Spurious Rate=FP/max(Gen,1), raw FN/FP, precision and F1.
They never contribute to Inventory or Memory scores. Empty GT makes presence and Missing Rate
not_applicable/null; count remains defined (both empty gives 1). No matched targets does not earn
an Appearance 5 or receive a frame-level Appearance 0. When an entire case has no matched target
despite a confirmed clear GT target, its case-level Appearance dimension is zero.

For each dimension, average reliable frame scores across the eight pairs, producing Inv_i, App_i,
Spa_i. **Then** calculate M_i=(Inv_i×App_i×Spa_i)^(1/3). Do not average frame-level geometric means
instead: nonlinear aggregation would give a different case score. Environment never participates.
A real zero stays exactly zero; use log/exp only for positive inputs, avoiding log(0) without
artificially lifting failures to epsilon.

Skip insufficient_evidence/not_applicable when GT does not expose the required fact, GT is
unreadable, or natural occlusion/viewpoint prevents comparison. If GT is clear and Gen exposes the
corresponding region but generated degradation makes the fact unreadable, the affected metric is
scored zero. Inventory keeps unknown counts/diagnostics null and uses generated_quality_failure
to assign zero Presence/Count; it never invents missing objects from blur. Appearance still only
evaluates matched instances. Renormalize selected/available submetric weights. Every null Metric
has a structured null_reason: reference_unavailable, natural_occlusion, generated_failure,
no_matched_objects, not_applicable, or unknown. Legacy missing reasons stay unknown unless reliable
Inventory annotations establish confirmed absent matches; never classify free-text evidence by keywords.
If all frame scores for an enabled core dimension are null, confirmed generated failure or absent
matches makes that case dimension zero (whole_dimension_failure_zero=true). Otherwise keep it null
and flag the case incomplete. Never compute the geometric mean using only the remaining enabled
dimensions. A known zero fixes the total at zero even if other dimensions remain unresolved, without
fabricating their grades. Disabled dimensions are excluded by configuration; Environment is diagnostic.

To update saved results under the generated-failure policy:

```bash
python -m evaluation.review_nulls --source OLD_RESULTS --output NEW_RESULTS --endpoint URL --model MODEL
```

Supply the key through `MEM_BENCH_API_KEY`. This performs one targeted image query per frame
with unresolved metrics, keeps Gate and existing numeric grades, and excludes derived unmatched
Appearance. Successful reviews are resumable. Original files are preserved; `null_review.json`
records each decision and evidence, and `review_progress.json` tracks progress. Python recomputes
the same weighted frame scores, temporal dimension averages and geometric case scores afterwards.

Each dimension/submetric benchmark mean uses its own valid case count. Report these denominators;
partial scores should not be interpreted as complete three-dimension assessments.

Benchmark M_Memory=sum(g_i*M_i)/N: failed Gates contribute zero. For a completed benchmark,
N includes every case. API errors or reference-unavailable/unknown passed cases remain unscored,
are reported separately, and do not masquerade as Gate failures/zeros. If any case remains unresolved,
benchmark_complete=false and the headline is null; partial_memory_score exposes the provisional
mean with memory_score_n scored cases, never a completed all-case score. Also retain the passed-only mean in `gate_valid_memory_score`
(0–1), `gate_valid_memory_score_100`, and its denominator `gate_valid_memory_score_n`.
The paper's **Memory Score=100×M_Memory**. Summary memory_score is normalized 0–1;
memory_score_100 and report.memory_score are the 0–100 headline. report also contains
Gate Coverage percent and dimension scores on the 0–100 scale. Do not silently turn API/schema
errors into invalid visibility or zeros; report error_cases separately.

## Running

```bash
python -m evaluation.run \
  --cases cases.json --output results/new_run \
  --endpoint https://your-endpoint/chat/completions --model your-vlm \
  --api-key-env MEMORYBENCH_API_KEY \
  --concurrency 32 --case-workers 16 \
  --qpm 270 --tpm 2187000 --tokens-per-request 9000 \
  --max-tokens 4096 --timeout 240 --retries 3
```

Default dimensions: Inventory, Appearance, Spatial. --dimensions or --metrics selects enabled
scores; use --dimensions inventory appearance spatial environment to add Environment diagnostics.
Inventory.f1 is not a selectable score. Lighting/style is optional per case.
--use-default-temperature omits temperature for endpoints that require their default.
New output directories are required, preventing accidental result overwrites.

Default maximum: 1 Gate + 8×3 Memory queries per passing case. No-match Appearance derivations
reduce that count. Gate rejection produces only the Gate call. Image preparation, dependency calls,
errors, retry attempts and schema validation remain auditable.

```text
run/
  run.json              v3 protocol, formulas, weights, rubric hashes and execution settings
  summary.json          separate Gate Coverage, memory means, diagnostics and denominators
  case_id/
    result.json         status, targets, Gate, per-frame scores and aggregate scores
    I_gt.jpg, I_gen.jpg Gate sheets
    sampling.json       Gate timeline and Read indices
    memory_sampling.json eight aligned timestamps, absent for rejected cases
    memory_frames/     1_gt.jpg, 1_gen.jpg, ... 8_gt.jpg, 8_gen.jpg
    judges/
      gate.json        prompts/schema/context/images and all charged/failed attempts
      frame_1/
        inventory.json
        appearance.json or appearance.derived.json
        spatial.json
```

Rubric changes require new VLM judgments: v2 Appearance scores penalized absent targets, whereas
v3 excludes them. Never rebrand those old labels as v3 by recomputing Python formulas alone.
Within a v3 interrupted run, exact matching inputs/model/rubrics/schema may reuse successful calls;
preserve unsuccessful attempts when repairing failures.

## Demo

```bash
python -m evaluation.viewer.server --results results/new_run --port 8765
```

Open http://127.0.0.1:8765. Compare Gate sheets, eight standalone pairs or synchronized videos.
Select a pair to inspect Inventory annotations, 0–5 VLM scores and evidence. Gate-rejected cases
have no Memory score. Disabled dimensions say so. Historical v2 scores retain their version label.
