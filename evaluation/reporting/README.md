# Reporting

当前报告和界面按样本展示，不推断缺失的任务标签。

- `report.py`：生成过程的进度报告和 v2 的 Markdown / CSV 汇总。
- `media.py`：GT / 生成视频逐帧同步拼接和对比缩略图。
- `finalize_memory_bench_report.py`：评分帧映射回源视频，输出算术审计。
- `validate_memory_bench_results.py`：源文件校验、视频帧数、抽帧、请求与分数复算；保留历史 v1 快照验证。
- `static/`：结果页面；由批处理复制到每个实验目录。
- `serve_results.py`：运行结果 demo，默认 8891。
- `build_checkpoint_demo.py`：不同 checkpoint 的视频对比页。

Final_Goal 中的 Operation Score、State-type Score、Stress Curve、Error Breakdown 需要补齐 case 标签、GT 和新指标后再汇总；典型 Case Study 可先使用现有逐帧审计页。
不能把缺失标签的样本自动归入某类，也不能把未完成或 API 失败的样本算成零分。
