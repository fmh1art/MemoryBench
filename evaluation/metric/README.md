# Metrics

现有 v2.1.0 已移除 State 维度：动作 Gate ×（物体清单、外观、空间关系的加权平均）。

- `schemas.py`：Gate、物体身份一一对应、逐维视觉判断的 JSON 校验。
- `scoring.py`：复算清单 F1、逐物体外观均值和适用维度的归一化权重。
- `rubrics/action_gate.md`：动作及目标可见性。
- `rubrics/pair_judge.md` + `rubrics/common.md`：逐对 GT / 生成图评分。
- `rubrics/legacy/`：旧版 prompt，仅历史参考；当前流程不加载。

API 错误、无效 GT 为 null，不能当模型零分；Gate 失败时 Memory 为零，内容维度未评估。

| Final_Goal 指标 | 当前缺口 |
| --- | --- |
| Task Success | Gate 仅验证 Probe 动作，不代表整个 Memory 任务成功；需要完整状态成功条件 |
| State Fidelity | 当前三维分可保留为基线；最终需按状态 GT 定义一致性 |
| Exact Match | 需要明确全部关键状态事实及匹配规则，不能用均分阈值冒充 |
| Error Breakdown | Forgotten / Hallucinated / Stale / Misbound / Duplicated 需要存在性、身份映射及状态历史证据 |

后续指标在本模块增加独立函数，并记录协议版本；不悄悄替换已有分数定义。
