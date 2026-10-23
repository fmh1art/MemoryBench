# 光轮 prefix / future 人工标注

复用 MemoryBench 的逐帧区间标注界面，同时浏览 `lightwheel-egopro`、`lightwheel-egostandard`。
源训练目录只读，默认标注输出到 `/mnt/bos/tm9b1x/fmh/data/MemoryBenchAnnotationsV2/lightwheel`，不写 V1 或 V2 的索引。

## 启动

本机已准备 `/var/lib/fmh/venvs/memorybench-annotator` 环境。
其他机器需要 Python 3.11+、ffmpeg，以及 `fastapi uvicorn av numpy pandas pyarrow pillow mcap protobuf`。

```bash
cd /mnt/bos/tm9b1x/fmh/projects/MemoryBench
/var/lib/fmh/venvs/memorybench-annotator/bin/python -m mem_case_anno_demo.lightwheel \
  --data-root /mnt/bos/tm9b1x/wmdata --source-format wmdata \
  --host 127.0.0.1 --port 8892
```

转发服务器 8892 端口后，在浏览器打开 http://127.0.0.1:8892 。
仅看某条录制可加 `--recording <UUID>`。初始每个源列出 100 条，避免遍历庞大目录；页面左侧可输入完整 UUID 打开其他录制。`--limit` 可修改初始数量，`--limit 0` 显式加载全部目录名，可能较慢。搜索框只筛选已载入列表。

默认 `--data-root` 为 `/mnt/bos/tm9b1x/wmdata`，直接使用：

- `/mnt/bos/tm9b1x/wmdata/lightwheel-egopro/`
- `/mnt/bos/tm9b1x/wmdata/lightwheel-egostandard/`

每个录制包目录名为 `<UUID>__<hash>`。程序只读取初始有界列表，在打开录制时读取该包的 parquet 元数据，直接播放 head-left MP4，并以 `meta/extents.parquet` 的逐帧 PTS / 传感器时间戳定位。无需 MCAP 或转码。同 UUID 存在多个版本时，输入完整包目录名选择版本。源 WMData 包保持只读。

## 仅展示 suitable 的人工复核队列

当前 8892 实例已开启 `--suitable-only`。只展示 VLM 已完成且判为 `suitable` 的录制；potential、unsuitable、uncertain 和未完成记录不会进入列表，数量不受最初每源 100 条的浏览上限限制。

```bash
python -m mem_case_anno_demo.lightwheel --source-format wmdata --suitable-only \
  --host 0.0.0.0 --port 8892
```

默认只读监测 `/var/lib/fmh/runtime/lightwheel-source-vlm/tasks-v3.sqlite3`。只有通过新“明确物理遮挡 → 同一目标重新出现”规则的 suitable 才会展示；旧 suitable 在复核通过前不会混入列表。也可以重复传 `--screening-database <path>`，后传入的优先级更高。

页面每 30 秒刷新列表，不会自动丢弃正在编辑的样本。已经打开的视频在更新后仍可保存标注；人工标注独立保存，不会被模型重跑覆盖。UUID 打开功能也受 suitable 筛选约束。VLM 初筛仅供选择回看视频，不代表已人工确认可转成正式 case。

## 兼容旧训练缓存 / MCAP

旧训练目录可显式指定 `--data-root /mnt/bos/tm9b1x/v05-training-ready/egoquest-lightwheel-fps2-20261001-v1/data --source-format training --media-root /path/to/LightwheelAI`。
它们保存 latent 缓存，不含可直接播放的 RGB。此模式按录制读取 `COMPLETE.json` 追溯原始 MCAP。
如果原路径不可访问，必须用 `--media-root` 指向实际挂载位置。支持以下相对结构：

- `EgoPro/EgoProStandard/mcap/<task>/<UUID>.mcap`
- `EgoStandard/EgoStandard/mcap/<task>/<UUID>.mcap`
- `<family>/mcap/<task>/<UUID>.mcap`、`<task>/<UUID>.mcap`、`<UUID>.mcap`
- `<family>/<UUID>.mp4` 或 `<UUID>.mp4`

`family` 是 `EgoPro` 或 `EgoStandard`。MP4 必须附带同名 `<UUID>.timestamps.json`，内容为
`{"sensor_timestamp_ns":[...],"source_time_origin_ns":123}`；数组按显示帧顺序存相对传感器时间，长度必须与视频帧数相同。
不能把训练历史采样率 2 fps 当成原始视频帧率。

MCAP 首次打开时，从 `/sensor/camera/head_left/video` 提取 H264，CPU 转码为浏览器可播放的 640×480 MP4，保存在输出目录的 `media_cache/`。首次等待可能较长；后续复用缓存。每条 RGB 消息与解码帧数必须相等，原始采集时间戳单独保留，消息/帧计数不一致则拒绝标注。此映射沿用上游每消息一帧的来源约定；不支持缺失消息或乱序时间戳的源。

## 标注

1. 选择数据源与录制，播放视频；空格控制播放，左右方向键逐帧，Shift + 左右跳 1 秒。
2. 用 1–4 或右侧按钮标记 prefix 开始、prefix 结束、future 开始、future 结束；结束边界不包含该帧。
3. 默认 prefix 结束与 future 开始联动，可取消以跳过中间无关片段。用下方按钮循环预览两段。
4. 填写可选 case 名称，保存草稿或完成标注。同一录制可新建多个样本，保存支持版本冲突检测与历史版本。
5. 点击「导出前缀 / 生成片段标注」。已完成样本写入 `exports/evaluation.jsonl`，草稿不导出。

时间边界保存原始显示帧号、MP4 PTS 与传感器时间戳。WMData 模式保存包目录、RGB stream / file ID、校验和及像素域；旧训练模式额外保存原始 MCAP / MP4 路径及采集时间原点。
界面支持通用独立区间；用于 MemoryBench 导出时要求 `prefix_start < prefix_end <= future_start < future_end`，避免历史/未来重叠造成泄漏。

## 导出 RGB case 草稿

```bash
/var/lib/fmh/venvs/memorybench-annotator/bin/python -m mem_case_anno_demo.export_lightwheel \
  --annotations /mnt/bos/tm9b1x/fmh/data/MemoryBenchAnnotationsV2/lightwheel/exports/evaluation.jsonl \
  --output-root /mnt/bos/tm9b1x/fmh/data/MemoryBenchAnnotationsV2/lightwheel/staged_cases
```

每个样本生成独立目录：`input/rgb_prefix.mp4`、`input/anchor.png`、`ground_truth/future.mp4`、`annotation.json`、`case.json`。
按原帧号裁切，拼接时间轴统一为 30 fps，anchor 是 prefix 最后一帧。已有同名 case 拒绝覆盖。

这一步产物是 RGB 草稿，状态为 `rgb_draft_requires_conditions_and_semantic_review`。
正式加入 MemoryBenchDataV2 前还需提取/对齐手部与头姿条件，人工审核 `future_action.txt` 与记忆目标语义，完成 V2 索引登记和素材校验。导出器不伪造这些字段，也不直接更新 V1/V2 manifest。

## SSH 转发下的加载速度

首屏与暂停逐帧定位只取带原始 PTS 的 960 像素预览图，点击播放才加载 MP4；整体视频不会在打开页面时抢占转发带宽。元数据/标注/回收列表并行读取，API JSON 压缩，视频 byte-range 与 JPEG 保持原协议。预览缩放不改变源帧号、时间戳、原始视频或导出素材；不传 `width` 参数的帧接口仍返回原始分辨率。

更新后先保存正在编辑的标注，再在 Mac 按 `⌘⇧R` 强制刷新。
