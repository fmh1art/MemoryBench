# Ground Truth

最终 GT 应明确与记忆有关的存在性、唯一身份、位置、关系和状态/属性；不要求整个画面像素一致。

当前代码：

- `annotation/server.py` + `annotation/static/`：人工标记五个关键边界；支持同一视频的多个评估样本。
- `schema.py`：`MemoryAnnotation`，人工指定目标、Probe 动作和成功条件，禁止评估器自行猜测目标。
- [标注操作说明](annotation/README.md)：真实 PTS / 传感器时间和导出约定。

当前尚无逐对象的完整状态历史 GT。增加状态标注时，必须用可回溯到画面证据的身份/位置/特征区分相似物体，明确新旧状态与更新事件。
VLM 评估时提取的物体清单是自动判断，不自动视为人工真值；契约和校验放在 `metrics/schemas.py`。
