# Download MemoryBenchData

The videos, tensor caches and annotations live in
[fmh1art/MemoryBenchData](https://modelscope.cn/datasets/fmh1art/MemoryBenchData),
not GitHub. The dataset currently contains 38 cases across three batches.
An authorized ModelScope account is required while the repository is private.

```bash
python -m pip install modelscope
export MODELSCOPE_API_TOKEN="your-token"
python data/download.py --output data
python data/validate.py --root data/evaluation_cases
```

The downloader disables all proxy variables for this process, verifies archive
SHA-256 and sizes, then safely extracts to data/evaluation_cases. It does not
change shared proxy services. Set `--download-only` to retain archives without
extracting. A re-run reuses ModelScope's download cache.

[Data format](../docs/DATA_FORMAT.md) defines both annotation modes and file layout.
[JSON Schema](eval_case.schema.json) describes shared case.json metadata. Do not
confuse fixed-memory home cases with ordinary prefix-video inference inputs.
