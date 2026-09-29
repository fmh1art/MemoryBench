# Migration and scope

The source snapshot is `/mnt/vdb_1t/project/acwm/acwm_data_rsi_system/mem_bench`.
The source directory, uncommitted changes, live annotation service and historical
results were not moved or overwritten.

| Source | Release |
| --- | --- |
| metrics | evaluation/metric |
| metrics/rubrics | evaluation/prompt |
| protocol | evaluation/pipeline |
| common, case_definition, reporting | evaluation/<same name> |
| ground_truth/schema.py | evaluation/ground_truth/schema.py |
| ground_truth/annotation | mem_case_anno_demo |
| baselines | evaluation/baselines |
| design_space, task_families, Final_Goal.md | docs |
| evaluation_cases | ModelScope MemoryBenchData archives |

Additional necessary modules are common (API/video/config), case_definition
(asset construction), ground_truth (annotation schema), reporting (audit/viewer),
baselines (model adapters), and tests. The data downloader verifies checksums.

`our_method` contains downloaded model/checkpoint transfer bundles; `simulation`
is a separate generation research area. Neither is required for the benchmark
scoring path and neither is included in this release. Historical evaluation_results,
logs, caches, local.toml and annotation service working records remain in the source.
All evaluation_cases contents, including annotations and batch validation records,
are archived. External selected-video symlinks were materialized; internal relative
packed/context links are preserved. Original absolute paths in historical metadata
are provenance, not portable lookup paths: resolve assets relative to the case folder.

This migration changes module paths and configurable runtime locations, not the
v2.1.0 scoring definition or existing case labels. Not all cases carry the final
operation/state/stress taxonomy. Missing labels are not inferred from filenames.
