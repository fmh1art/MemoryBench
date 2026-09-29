# Design Space

每个 case 明确记录以下三项；它们是任务标签，不是 VLM 自报分数。

| 维度 | 取值 | 应记录的信息 |
| --- | --- | --- |
| Memory Operation | Write / Maintain / Update | 本 case 重点考察哪一种记忆操作 |
| State Type | Object Existence / Attribute-State / Spatial Location / Object Relation / Scene Modification | 要恢复的状态事实 |
| Stress | Delay / Memory Load / Interference / Similarity / Update Depth | 压力类型、数值和单位，例如 Delay = 30 秒 |

当前代码未自动生成这些标签。后续将它们加入 case 元数据时，需由任务设计或人工标注确定；不要由生成结果反推标签。
物体状态 GT 放在 `ground_truth/`，分数计算放在 `metrics/`，按标签分组的汇总放在 `reporting/`。
