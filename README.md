# MemoryBench

Action-conditioned world model memory evaluation, with auditable VLM judgments and deterministic scoring (v2.1.0).

## Evaluation

The evaluation framework uses five VLM judges organized by dimension:
Gate (Observability and Action Following), Inventory, Appearance, Spatial and Environment.
GT and generated videos are uniformly sampled at 12 aligned timestamps into two independent
2×6 contact sheets. Gate runs first; passing cases receive four concurrent memory judgments.
Each judgment provides evidence; Inventory metrics are computed from semantic annotations,
and other metrics use explicit 0–5 rubrics.

```bash
python -m pip install -r requirements-eval.txt
python -m evaluation.run --help
# Prepare a JSON array following evaluation/cases.example.json.
python -m evaluation.run --cases cases.json \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model your-vlm --output evaluation_results/run_001 --concurrency 8
```

See [evaluation implementation, rubrics and scoring](evaluation/README.md).
Gate failures and insufficient evidence are excluded from the conditional memory score;
coverage and error counts are reported separately. The previous evaluation, generation,
packing and baseline-adapter code has been removed from `evaluation/`.

The annotation app is separate: `python -m mem_case_anno_demo.server --help`.
For Lightwheel prefix/future annotation, see [the source adapter and demo guide](mem_case_anno_demo/LIGHTWHEEL.md).

## Data and generation

All **38 cases** are in [fmh1art/MemoryBenchData](https://modelscope.cn/datasets/fmh1art/MemoryBenchData):
5 office, 11 home, 22 Quest. See [download instructions](data/README.md),
[data format](docs/DATA_FORMAT.md), and [migration notes](docs/MIGRATION.md).

The scoring path does not need model weights. MemWorld generation/packing needs
an external MemWorld checkout and weights; set `MEMWORLD_ROOT`, `WAN_MODEL_DIR`,
and `MEMWORLD_CHECKPOINT`. Source reconstruction additionally needs
`WMDATA_ROOT` (evophys-wmdata checkout and its hand reviewer) and source WMData.
Generation/conversion adapters are no longer bundled in `evaluation/`; prepare paired videos
with your external generation pipeline before scoring.

Home cases carry fixed cached memory and are marked `assets_ready`; their metadata
states that a fixed-memory mid-trajectory rollout adapter is required. Do not
interpret this release as a claim that every case has completed inference, or use
the generic MemWorld runner for that batch without the required adapter.

The annotation demo reads WMData packages, not the exported evaluation case folders.
Run `python -m mem_case_anno_demo.server --help` for source/output options.

No new redistribution license is granted by this migration. Preserve upstream
licenses for external model repositories and obtain the appropriate data access.
