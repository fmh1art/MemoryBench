# Evaluation Protocol

最终协议：Initialize → Memory Event → Retention / Update → Probe。
前三个阶段产生要保留的历史状态；Probe 重新暴露目标，检查模型生成的状态。

当前实现以人工边界 `prefix_start < t < action_end <= eval_start < eval_end` 执行：

- `infer_evaluation_cases.py`：历史条件输入与从 t 开始的滚动视频生成。
- `run_memory_bench.py`：多个 GPU 生成样本，生成完成后立即排队评分。
- `pipeline.py`：人工目标 → 动作 Gate → 稳定段逐对图像判断 → 确定性算分。
- `evaluate.py` / `run_eval.sh`：单样本评分入口。
- `reevaluate_memory_bench.py`：复用生成结果，批量重新评分及断点继续。
- `compare_checkpoint_videos.py`：对同一样本生成不同 checkpoint 的视频。

四阶段的概念不等于上述五个技术边界；现有历史前缀尚未细分 Initialize、Memory Event 和 Retention / Update。
详细口径见 [PROTOCOL.md](PROTOCOL.md)，使用说明见 [EVALUATION.md](EVALUATION.md)，历史长度限制见 [MEMORY_PREFIX_LIMITS.md](MEMORY_PREFIX_LIMITS.md)。
