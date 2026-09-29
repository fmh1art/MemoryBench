# MemoryBench

Action-conditioned world model memory evaluation, with auditable VLM judgments and deterministic scoring (v2.1.0).

## Structure

```text
evaluation/
  metric/             scoring and judgment schemas
  pipeline/           generation, evaluation and batch commands
  prompt/             action gate and pair judgment prompts
  common/             configuration, video sampling and VLM client
  case_definition/    build, pack and validate evaluation assets
  ground_truth/       memory annotation schema
  reporting/          reports, result validation and viewer
  baselines/          existing model adapters and input conversion helpers
configs/              example configuration (local credentials excluded)
docs/                 protocol, task definitions and data contract
data/                 ModelScope download instructions and JSON Schema
mem_case_anno_demo/    WMData annotation web app
tests/                existing protocol and annotation regression tests
```

## Install and evaluate

Python 3.11+ and FFmpeg are required. From this repository:

```bash
python -m pip install -r requirements-tools.txt
cp configs/example.toml configs/local.toml
# Edit the VLM endpoint/model in configs/local.toml.
export MEM_BENCH_API_KEY="your-service-key"
python data/download.py --output data
python -m evaluation.pipeline.evaluate --help
python -m evaluation.pipeline.reevaluate_memory_bench --help
python -m mem_case_anno_demo.server --help
python -m pytest -q
```

Single-case scoring requires GT evaluation video, generated evaluation video,
separate generated action video, and explicit memory_annotation.json:

```bash
python -m evaluation.pipeline.evaluate \
  --gt data/evaluation_cases/eval_002_10891f27/ground_truth/evaluation.mp4 \
  --generated /path/to/prediction/evaluation.mp4 \
  --action /path/to/prediction/action.mp4 \
  --annotation data/evaluation_cases/eval_002_10891f27/memory_annotation.json \
  --config configs/local.toml --output-dir evaluation_results/my_run
```

Scoring samples an action gate and stable-frame pairs, then computes identity-aware
inventory, appearance and spatial scores. A failed gate gives zero memory score.
An API or validation failure is an evaluation error, not a valid zero score.
See [evaluation protocol](evaluation/pipeline/EVALUATION.md).

## Data and generation

All **38 cases** are in [fmh1art/MemoryBenchData](https://modelscope.cn/datasets/fmh1art/MemoryBenchData):
5 office, 11 home, 22 Quest. See [download instructions](data/README.md),
[data format](docs/DATA_FORMAT.md), and [migration notes](docs/MIGRATION.md).

The scoring path does not need model weights. MemWorld generation/packing needs
an external MemWorld checkout and weights; set `MEMWORLD_ROOT`, `WAN_MODEL_DIR`,
and `MEMWORLD_CHECKPOINT`. Source reconstruction additionally needs
`WMDATA_ROOT` (evophys-wmdata checkout and its hand reviewer) and source WMData.
The local conversion helper is in `evaluation/case_definition/support/`.
Baseline adapters require their respective upstream repositories and environments;
see `evaluation/baselines/INPUT_ADAPTATION.md`. Some legacy baseline setup scripts
retain machine-specific defaults and must be configured before use.

Home cases carry fixed cached memory and are marked `assets_ready`; their metadata
states that a fixed-memory mid-trajectory rollout adapter is required. Do not
interpret this release as a claim that every case has completed inference, or use
the generic MemWorld runner for that batch without the required adapter.

The annotation demo reads WMData packages, not the exported evaluation case folders.
Run `python -m mem_case_anno_demo.server --help` for source/output options.

No new redistribution license is granted by this migration. Preserve upstream
licenses for external model repositories and obtain the appropriate data access.
