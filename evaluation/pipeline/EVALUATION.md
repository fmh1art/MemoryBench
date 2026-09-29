# ACWM Memory Evaluation v2.1.0

当前入口采用人工任务标注、独立动作 Gate 和稳定片段逐对图像评估；旧版自动定位、类别最大匹配流程不再运行。
最新结果：[epoch 10 / step30080 的 5 个样本](../evaluation_results/memory_eval_v2_1_epoch10_step30080_20260912/REPORT.md)。
界面：`http://127.0.0.1:8891/memory_eval_v2_1_epoch10_step30080_20260912/`，远程访问需转发 8891 端口。

## 输入与流程

每个 `../evaluation_cases/eval_*/memory_annotation.json` 保存人工定义的目标、动作和完成条件。
当前 5 个样本的目标均为抽屉内的物体；第 2 个样本为 `turn_view`，其余为 `open_drawer`。
标注过程见 [data_annotation](../ground_truth/annotation/README.md)，本程序不再让 VLM 推断目标。

1. 从生成动作段 `[t, action_end)` 按时间顺序均匀抽 6 帧，少于 6 帧时使用全部帧。
2. 从 GT 和生成的稳定评分段 `[eval_start, eval_end)` 各均匀抽 8 帧，少于 8 帧时使用全部帧。
3. Gate 的一次 VLM 请求只输入动作标注和生成动作帧，判断动作执行是否正确且目标区域可见。
4. Gate 为 0 时 Memory 为 0，并跳过物体评分；Gate 为 1 时，按相对时间进度逐对比较 GT 和生成图。
5. 每对图仅发送 2 张图片，一次 VLM judge 同时输出物体清单、外观、空间关系的判断与证据。
6. 每维先跨图片对求平均，再按配置的权重求归一化加权平均。

抽帧使用视频真实 PTS，保留首尾及真实帧号；双方抽帧数不同时配成较多侧的组数，较短侧可重复最近进度的帧，所有独立抽帧均被使用且重复有记录。
当前 5 个样本都满足 8 对图；每个 Gate 通过的样本共 9 次逻辑调用（Gate 1 次、图片对 8 次），格式修复重试单独记录。
所有 prompt 包含 ACWM 任务背景，客户端和 vLLM 服务的每次图片输入上限均为 8。

## 三维定义与权重

| 维度 | 定义 |
| --- | --- |
| 物体清单 | 根据确认的具体物体身份一对一匹配，计算 `2×匹配数/(GT数+生成数)`；两侧可见且均为空时为 1。 |
| 外观 | 形状、颜色、纹理、材质及细节保真度；逐 GT 物体打分后求均值，未匹配物体记零。 |
| 空间关系 | 物体间或物体与内部布局的有意义关系；普通“在抽屉内”不计额外分。 |

VLM 判断视觉事实及身份，Python 复算清单 F1、逐物体外观均值及权重，避免依赖 VLM 的算术；保留原始三维报告和程序复算差异。
不强制同类别匹配，只有相似颜色、类别或位置不能认定同一物体；GT 非空且没有确认身份时，所有适用维度均为 0。
GT 无适用事实的维度为 N/A 并排除；生成缺失、模糊或错误不能把适用维度变成 N/A。
默认三个配置权重均为 1；三个维度都适用时各占 1/3。
Gate 失败的内容维度显示“未评估”，API 或无效 GT 错误保留 null 且不作为模型零分。

修改 [local.toml](../configs/local.toml) 的权重即可用于下一次评估：

```toml
[weights]
inventory = 1.0
appearance = 1.0
spatial = 1.0
```

协议与边界规则见 [PROTOCOL.md](PROTOCOL.md)；V1 和 V2 分数定义不同，不能把跨协议分数变化解释为模型能力变化。

## 运行

Python 使用 `/mnt/vdb_1t/project/acwm/MemWorld/.venv/bin/python`，依赖见 [requirements-eval.txt](../requirements-eval.txt)。
VLM 部署参考 `../../../deploy/qwen3_8_vllm`（仓库根目录的 deploy），历史运行使用 `memory_bench_4gpu.config.sh`；当前服务以本机部署配置为准。
[local.toml](../configs/local.toml) 已配置本地 API、模型、采样和权重；需要密钥时可使用 `MEM_BENCH_API_KEY` 环境变量。

复用已经生成的视频，重跑全部 VLM 评估：

```bash
cd /mnt/vdb_1t/project/acwm/acwm_data_rsi_system/mem_bench/protocol
/mnt/vdb_1t/project/acwm/MemWorld/.venv/bin/python reevaluate_memory_bench.py \
  --source ../evaluation_results/memory_eval_epoch10_step30080_20260912 \
  --output ../evaluation_results/my_new_evaluation \
  --root evaluation_cases --config ../configs/local.toml --workers 4
```

同一模型、标注、配置和版本中断后可追加 `--resume`；修改配置或 prompt 时应使用新目录，避免混合口径。
单个样本的输入必须明确区分生成动作段与双方稳定段：

```bash
./run_eval.sh \
  --gt /path/to/gt_evaluation.mp4 \
  --generated /path/to/generated_evaluation.mp4 \
  --action /path/to/generated_action.mp4 \
  --annotation ../evaluation_cases/eval_001_a17f6882/memory_annotation.json \
  --config ../configs/local.toml --output-dir /path/to/new_result
```

需要重新生成 ACWM 视频时：

```bash
/mnt/vdb_1t/project/acwm/MemWorld/.venv/bin/python run_memory_bench.py \
  --root evaluation_cases --output ../evaluation_results/my_new_generation \
  --config ../configs/local.toml --gpus 0,1,2,3 --checkpoint /path/to/dit.safetensors \
  --steps 20 --cfg 5 --seed 42
```

每张 ACWM 卡独立处理一个样本，视频完成即提交 VLM 评分；当前重评估只复用 epoch 10 / step30080 的生成视频。

## 结果与审计

每个样本目录集中保存前缀、手部条件、GT/生成完整未来视频、稳定视频、动作视频、对比视频、标注、来源校验及评分。
`evaluation/frames` 保存全部真实评分帧，`evaluation/pairs/pair_*.json` 保存各对评分，`evaluation/vlm_calls` 保存完整 prompt、图片路径、原始响应和重试。
`selected_frames_with_source.json` 将评分帧映射回源视频和传感器时间；生成帧映射表示条件时间，并不证明视觉对应。
汇总为 `status.json`、`metrics.csv`、`REPORT.md`；`run_config.json`、`vlm_deployment.json`、`code_snapshot` 和 rubric 快照保存运行口径。

```bash
/mnt/vdb_1t/project/acwm/MemWorld/.venv/bin/python ../reporting/finalize_memory_bench_report.py --root /path/to/completed_run
/mnt/vdb_1t/project/acwm/MemWorld/.venv/bin/python ../reporting/validate_memory_bench_results.py --root /path/to/completed_run
```

界面提供 5 个样本切换、GT/生成同步播放、片段选择、逐对图像与评分依据、6 帧 Gate 证据、原始请求链接和移动端布局。
前端源文件为 `../reporting/static/results_viewer.html`、`results_viewer.css` 和 `results_viewer.js`，批处理复制至结果目录供静态服务展示。

## 检查代码

| 环节 | 实现 |
| --- | --- |
| 抽帧及图片配对 | [video.py](../common/video.py)、[pipeline.py](pipeline.py) |
| Gate prompt | [action_gate.md](../metric/rubrics/action_gate.md) |
| 三维 rubric | [pair_judge.md](../metric/rubrics/pair_judge.md)、[common.md](../metric/rubrics/common.md) |
| JSON 契约与评分 | [schemas.py](../metric/schemas.py)、[scoring.py](../metric/scoring.py) |
| API 请求、上限与重试 | [vlm.py](../common/vlm.py) |
| 视频复用与批处理 | [assets.py](../case_definition/assets.py)、[reevaluate_memory_bench.py](reevaluate_memory_bench.py) |

旧版 target、localize、inventory、match、compare rubric 仅为历史参考，当前 pipeline 不加载它们。
测试：`PYTHONPATH=/tmp/acwm-annotator-test-deps /mnt/vdb_1t/project/acwm/MemWorld/.venv/bin/python -m pytest -q ../tests`。
程序校验检查片段、帧数、请求和算分一致性，不能保证 VLM 识别和身份匹配正确；请在逐帧页复核视觉证据。
