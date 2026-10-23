# 光轮原始视频描述与 MemoryBench 初筛

对 EgoPro / EgoStandard 每条完整 WMData 录制的 head-left RGB 视频均匀抽 24 帧，按 **4 行 × 6 列**拼成一张带实际 PTS 时间戳的大图。用指定 Ark endpoint 生成中文整体描述、时间分段活动及 MemoryBench 候选判断。

原始视频、WMData 包、MemoryBenchDataV1 和 V2 索引均不修改。结果是稀疏抽帧下的 VLM 初筛，不能代替逐帧人工确认，也不会直接构建正式 eval case。

## 本机运行

使用 `/var/lib/fmh/venvs/memorybench-annotator/bin/python`。依赖来自项目基础依赖（httpx / pydantic / av / Pillow / numpy），SQLite 为标准库。无需 GPU。

密钥只从本机私密文件或显式 `--key-file` 读取，不进入代码、日志、数据库或结果。不要把实际 key 放在命令行。

```bash
cd /mnt/bos/tm9b1x/fmh/projects/MemoryBench
/var/lib/fmh/venvs/memorybench-annotator/bin/python -m evaluation.source_annotation.run \
  --supervise --workers 64 --prepare-workers 24 --requests-per-second 4
```

当前 tmux 会话为 `lightwheel-source-vlm-v2`，日志 `/var/lib/fmh/lightwheel-source-vlm-v2.log`。
早期试跑保存在 `ark-20261023` 与 `tasks.sqlite3`；收紧“普通清理不是记忆读取”判据后的正式任务使用 `ark-20261023-v2` 与 `tasks-v2.sqlite3`，不同 prompt 的结果不混作同一批统计。
另一个 tmux 任务 `lightwheel-duration-stats` 生成本次完整录制清单和时长表。

默认模型为 `ep-20250612104210-ss27q`。每次请求优先用 `https://ark.cn-beijing.volces.com/api/v3`。
**只有主地址连接失败**（ConnectError / ConnectTimeout）时才尝试 `https://ark-cn-beijing.bytedance.net/api/v3`；主地址正常时不请求备用地址。主地址返回 429、5xx 或读取超时会在主地址退避重试，401/403/404 等配置错误暂停队列。

先试跑少量记录：

```bash
python -m evaluation.source_annotation.run --max-new 3 --workers 3 --requests-per-second 1
```

同一数据库只能有一个 runner，进程锁避免重复调用。再次运行自动续跑；成功记录不会重复发请求。`--retry-failed` 重试失败记录。
`--supervise` 自动恢复临时 API 中断，429 后降低请求速率；全量结束后还会重试失败视频最多两轮。永久配置错误或本地队列空间不足会保留 pending 状态并退出，原因记录在 progress 中。

## 输入与范围

默认清单目录 `/var/lib/fmh/data/lightwheel-duration-stats-20261023`，包含两源独立 CSV 和 `summary.json`。
只在单源 CSV 已完整生成并通过行数校验后导入；一源未完成不会阻塞另一源处理。
按完整包目录去重，不包含 `.incomplete`。左右摄像头不重复计数。本次导入的清单就是本次处理快照，目录后续新增的录制不自动混进已经冻结的清单。

默认持久队列 `/var/lib/fmh/runtime/lightwheel-source-vlm/tasks-v2.sqlite3` 位于本地磁盘，使用 WAL。成功结果的压缩副本保存于队列，便于汇总而无需重新逐个读取远端文件。本地剩余空间不足 512 MiB 时暂停，不继续填满磁盘。队列含进度、状态与模型结果，不含 API key。

## 结果

默认输出目录：

`/mnt/bos/tm9b1x/fmh/data/MemoryBenchAnnotationsV2/lightwheel/source_vlm/ark-20261023-v2/`

每条录制按 `family/UUID前两位/包目录名/` 保存：

- `sheet.jpg`：24 帧拼图。
- `sheet.json`：抽样目标秒数、实际 PTS / 秒数、源 MP4 大小 / mtime、拼图 SHA256、布局等。不会为了统计校验读取完整 MP4 的 SHA256。
- `result.json`：整体描述、分段活动、主要物体、适合度、事件证据、最多 3 个粗候选 prefix/future 区间、模型原始响应和用量。
- `invalid_response.json`：仅在结构校验连续失败时保留最近响应及原因。

根目录 `progress.json` 每约 30 秒更新任务数量、适合度分类计数及累计 token 用量。`run_config.json` 保存模型、prompt、schema、抽帧版本与指纹；更改这些关键配置必须使用新的输出目录和队列。

`annotations.jsonl` 和 `screening_summary.json` 在 runner 本轮退出时汇总，长跑途中可能仍是上次试跑的汇总版本；运行中的最新结果以 `progress.json` 和逐条 `result.json` 为准。全部完成后获得全量 JSONL。

## 标注内容与校验

Prompt 见 [PROMPT.md](PROMPT.md)，结构校验见 [schema.py](schema.py)。

- `overall_description`：整段视频总体活动。
- `segments`：按时间排列的活动片段，每段带近似秒数及证据帧。
- `main_objects`：可辨认的主要物体。
- `memorybench.decision`：`suitable` / `potential` / `unsuitable` / `uncertain`。
- `memory_events`：可见的 write、occlusion、distractor、read、update 事件。
- `candidate_cases`：需要人工复核的粗 prefix/future 时间区间、记忆目标、写入/读取证据和需要历史记忆的理由。
- `limitations`：本视频的证据限制及均匀抽帧遗漏短事件的风险。

“有物体操作”不等于适合 MemoryBench。判断需要具体历史写入、目标不可见/遮挡以及之后重新读取的证据。无法确认完整链条时使用 potential / uncertain，不能捏造缺失过程。

结构不合规会限次重试；超出视频时长、反向/重叠区间、越界证据帧编号、没有候选却标为 suitable 等均被拒绝。所有候选仍保持 `human_verified: false`，后续须在标注 demo 中回看确认边界、补齐条件和语义后才能登记到 V2。

验证：

```bash
python -m pytest -q tests/test_source_annotation.py
```
