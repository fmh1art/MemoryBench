# Case Definition

最终 case = Design Space Point + Task Family + Protocol。这里负责样本构建与模型输入，GT 内容由 `ground_truth/` 管理。

现有入口按顺序阅读：

1. `build_evaluation_cases.py`：读取人工边界，提取历史 RGB、历史/未来手部和 head pose、GT 未来及稳定片段，生成帧映射。
2. `pack_evaluation_cases.py`：用 Wan VAE / 文本编码器打包成 MemWorld 的 sample_pt_v2。
3. `validate_evaluation_assets.py`：检查导出视频、帧映射与对齐。
4. `assets.py`：重评估时复制已有生成视频、裁剪动作段、记录来源校验和。

默认产物在 `../evaluation_cases/`，保留原位置。原始视频只读；GT 未来画面不能进入模型历史条件。
当前 `case.json` 含源身份、五个边界、对齐映射和推理参数；尚未具备完整 Design Space、Task Family、四阶段标记。
增加这些元数据时应保留原字段，并由人工明确事件和压力维度，不把 `t` 等现有技术边界直接当作 Memory Event。

## 独立前缀 / 生成区间

构建器支持 `prefix_generation` 四边界标注：按 `[prefix_start,prefix_end)`、`[t,eval_end)` 拼接；要求两段按源时间排序且不重叠，anchor 为 `prefix_end-1`。先按原 sensor 时间对齐，再拼接，源时间缺口不补帧。

此模式没有单独的动作结束和稳定评估边界，明确使用完整生成段进行评估，`action_end=null`、`evaluation_scope=full_annotated_generation`；保留原始 `source_marks`，不伪造人工边界。`--include-drafts` 仅在明确选用有效草稿时传入，不改变源标注状态。五边界模式不变。

已构建示例与复现信息：[13 个语义组抽样](../evaluation_cases/quest_20260921_semantic13_v1/README.md)。
