# Automated video diagnostics

This is an independent, no-VLM evaluation path. It reads GT/Gen paths and start offsets from
existing `*/result.json` files, evaluates **all cases regardless of Gate**, and writes a separate
result tree. It does not change Memory Score, Gate labels, source metadata or existing results.

## Paper and implementation audit

[Training Object Permanence in World Models, §4.5](https://arxiv.org/html/2609.28654v1#S4.SS5)
reports MSE, MAE, PSNR, SSIM, MS-SSIM, LPIPS, temporal-difference L1, final-frame MSE/SSIM/LPIPS,
CLIP similarity and FID. The paper treats these as reference-fit diagnostics, not measures of
physical reasoning correctness. Its protocol resamples predictions to the target frame count and
uses 320×192, with additional 1280×720 final-frame evaluations.

On 2026-10-09, the public [official repository](https://github.com/hokindeng/object-permanence)
at commit `824baddcdad087810d35638bf8f273b884f74975` was inspected. No evaluation script implementing
that suite was found in the released tree; the code covers Blender generators and the PWM training
stack. This module is an **independent implementation inspired by the paper**, not their official
scoring code or a claim of numerically reproducing their table.

## Implemented metrics

All pixel calculations use RGB in [0,1]. Frame scores are averaged within each video; video scores
are then averaged with equal case weights. No heterogeneous metrics are combined into one score.

| Metric | Definition / direction |
|---|---|
| MSE / MAE | Mean squared / absolute GT–Gen pixel difference; lower |
| PSNR | `-10 log10(clip MSE)` in dB; higher. This is not the mean of frame PSNRs. |
| SSIM | 11×11 Gaussian window, sigma 1.5, valid spatial window, population covariance, RGB channel mean; higher |
| MS-SSIM | Five scales, weights [0.0448,0.2856,0.3001,0.2363,0.1333], 2×2 mean downsampling with edge padding for odd dimensions, nonnegative contrast-structure factors; higher |
| Temporal-difference L1 | `mean(abs((Gen[t]-Gen[t-1])-(GT[t]-GT[t-1])))`; lower |
| Final MSE / SSIM | Last aligned frame, at the same scale as the clip metrics |
| LPIPS / Final LPIPS (optional) | Pretrained AlexNet LPIPS v0.1; lower |
| NIQE (optional) | PyIQA NIQE on generated frames, then frame mean; lower. An image-quality diagnostic, not a temporal video-quality model. |
| GT / Gen Laplacian variance | Grayscale sharpness diagnostic; noise can also increase it, so **not** a calibrated quality score |
| GT / Gen adjacent L1 | Mean adjacent-frame change; detects static output / large changes, but does not distinguish legitimate motion from flicker |

MS-SSIM is null if either effective spatial dimension is <161 (five scales must remain defined).
Temporal metrics are null for fewer than two frames. Exact identity PSNR is serialized as `"inf"`,
with `infinite_cases` reported in the summary. Errors and nulls never become zeros.

LPIPS uses the [authors' library](https://github.com/richzhang/PerceptualSimilarity) with RGB in
[-1,1], evaluation/inference mode and pretrained weights. NIQE uses
[PyIQA](https://github.com/chaofengc/IQA-PyTorch). Neither is silently replaced with random weights
or a simpler proxy if a dependency or weight is unavailable; the case records an error.

CLIP similarity, FID and FVD are **not implemented** here. FID/FVD require a specified cohort,
feature extractor and sample protocol; treating them as independent per-case scores would be
misleading. These global image metrics also cannot isolate memory targets without reliable masks.

## Sampling: explicit differences from WROP

- Evaluate the **full shared future interval**, excluding Prefix. Do not use the middle-80% eight
  frames from VLM judging for temporal quality metrics.
- Default 24 Hz, plus the last shared timestamp; match decoded PTS with the existing nearest-frame
  logic and case start offsets. No independent time normalization or stretching of clip duration.
- Reject a sample if GT/Gen skew or sampling error exceeds 0.1 s. Record actual timestamps and
  unique source-frame counts, so duplicated frames at lower source FPS remain inspectable.
- Fit GT inside a maximum 320×192 canvas **preserving aspect ratio**, then use the GT content size
  for evaluation. Fit/pad Gen to this size. This removes artificial GT letterbox pixels from scores
  while retaining any Gen aspect-ratio discrepancy. A 4:3 video is evaluated at 256×192.
- Always report `effective_size`. The paper's exact resizing, color conventions and MS-SSIM details
  are not fully specified in its public scoring code (not found), so our values are not directly
  comparable to its tables.
- `--max-frames N` optionally caps sampling uniformly over the full interval. It changes temporal
  spacing and must be held fixed when comparing runs; 0 (default) means no cap.
- Shared duration and both source endpoints are saved. A truncated output is evaluated only over
  overlap; inspect duration coverage separately instead of interpreting good overlap fit as completion.

Camera differences and background dominance can strongly affect all full-reference scores.
High similarity or sharpness does not establish object permanence. Retain VLM Memory and these
video diagnostics as separate reports.

## Run

```bash
pip install -e '.[automated]'
python -m evaluation.automated.run \
  --results /home/fanmeihao/data/acwm/eval_results/_evaluation/terra_v3_3_full_20261007 \
  --output /data01/fanmeihao-annotation/automated/20261009/yijie0924 \
  --workers 2 --resume
```

Use `--limit 10` for a smoke run. It selects sorted case IDs, **not** a stratified benchmark sample.
Use separate output roots for different models/protocols. Memory/compute grow with clip length;
reduce workers or explicitly cap frames for long clips.

Optional pretrained metrics (additional downloads, no API/VLM calls):

```bash
pip install -e '.[automated-perceptual,automated-quality]'
# Keep caches off the home filesystem if needed.
export TORCH_HOME=/data01/fanmeihao-annotation/model-cache/torch
export HF_HOME=/data01/fanmeihao-annotation/model-cache/huggingface
python -m evaluation.automated.run \
  --results /path/to/vlm_results --output /path/to/automatic_results \
  --learned lpips niqe --workers 1 --device cpu
```

`--device cuda:0` is supported where PyTorch/CUDA is available. Learned metrics run in a single
worker so weights are shared and concurrent workers do not duplicate model memory.

Outputs:

- `run.json`: metric definitions, package versions, inspected upstream commit and sampling settings.
- `<case_id>/automated.json`: raw metrics, frame metrics, aligned timestamps, input fingerprints,
  timing, missing reasons or error message.
- `summary.json`: per-metric mean/count, infinite counts and explicit failed-case list.

Resume only reuses successful results with matching case, input file size/mtime, settings and
protocol version. Progress prints after every case. Partial failures yield a nonzero CLI exit code.

Validation: identity / constant shifts, temporal mismatch, null edge cases, SSIM agreement with
scikit-image, actual MP4 decoding/alignment, and input/settings-sensitive resume behavior.
The initial real-data smoke run covers the same first 10 cases of both models. Optional LPIPS/NIQE
were not executed in that base CPU environment; install the optional dependencies before enabling.
