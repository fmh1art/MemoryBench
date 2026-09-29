# Release validation

- 64 regression tests passed in the standalone repository; the external MemWorld integration test is skipped when MEMWORLD_ROOT is unset.
- The optional MemWorld integration test also passed when explicitly pointed at the existing local MemWorld checkout (65 distinct tests passed in total).
- Evaluation, re-evaluation and annotation demo CLI help commands succeeded.
- All 38 case metadata records passed the published JSON Schema, interval/count/slice checks and required-file checks.
- The three primary archives were streamed to check readability, member paths and the presence of exactly 38 distinct case.json records; results accompany the dataset under metadata/archive_validation.json.
- Archives carry SHA-256 checksums and sizes in archives.json and SHA256SUMS.
- No model inference or paid VLM evaluation was rerun for this structural migration.

Dependencies: external model repositories/checkpoints are needed for generation. The home batch's fixed-memory adapter requirement remains explicit in its metadata.

Home bank dependencies are additionally included in 11 supplemental archives with relative packed indices. The final asset validator also checks every referenced sample-bank file exists.

The extracted release passed validation for all 38 cases, including every packed sample path. All 11 home banks and their remapped memory-table, trajectory-cache and prompt-cache paths were checked; see metadata/dependency_validation.json in the dataset.
