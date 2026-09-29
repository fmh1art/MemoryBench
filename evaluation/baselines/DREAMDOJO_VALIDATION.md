# DreamDojo 官方样例验证（2026-09-24）

**结论：原生首帧＋真实 GR1 动作能驱动合理运动；此前的 5 帧视频历史适配有问题，全零 action 也不是有效动作适配。** 当前三个片段只用于排查推理链路，不代表复现论文全部指标。

## 测试内容

- 数据：[NVIDIA 官方 GR1 数据集](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1)，`In-lab_Eval/gr1_unified.pnp_corn_robot` 的 episode 0、1、2；版本见输出目录 `dataset_revision.txt`。
- 权重：`2B_GR1_post-train/iter_000050000` 的 EMA；使用现有转换权重和官方 `Video2WorldInference`，复用官方 `MultiVideoActionDataset` 预处理。
- 官方参数：首帧条件、空 prompt、guidance=0、每轮 12 帧、4 轮、640×480；固定相同种子比较真实 action 与全零 action。20 fps 原视频隔帧采样，输出 10 fps，共 49 帧（含首帧）。
- 首行初始 state、min-max 归一化、每 4 帧相对同一基准的 action 差分均保持官方实现；前 29 维按左臂／右臂／左手／右手／腰排列，其余为零。没有使用未来 RGB 作为推理条件。
- 环境兼容沿用现有 checkpoint 路径、VAE revision、移除无关数据配置及文本编码器 offload。日志中的缺少 LAM 权重提示不影响本次 GR1 路径：当前推理 API 的 LAM 提取代码已被上游注释，末 32 维也全为零。

## 结果

表中为 PSNR(dB) / SSIM；不计条件首帧，直接用编码前预测与原始 GT 数组计算。

| Episode | 真实动作：首轮 12 帧 | 真实动作：完整 48 帧 | 全零动作：完整 48 帧 |
|---|---|---|---|
| 000000 | 24.77 / 0.883 | 20.30 / 0.749 | 18.80 / 0.766 |
| 000001 | 24.93 / 0.874 | 21.04 / 0.778 | 18.74 / 0.756 |
| 000002 | 26.53 / 0.896 | 22.54 / 0.798 | 20.05 / 0.784 |

画面检查：真实动作下可看到抓取并向盘子搬运，后段有手臂外观漂移、额外物体幻觉等误差；episode 0 的全零动作基本停留在初始姿态。背景占比较大，静止视频也可能有较高 SSIM，不能只按像素指标判断动作成功。

## 定位到的适配问题

1. **5 帧条件超出此 checkpoint 的训练配置。** `exp_2B_action_conditioned_rectify_flow_gr00t.py` 的 GR1 release 继承 `2b_bridge_action_conditioned_oss`，明确 `min_num_conditional_frames=max_num_conditional_frames=1`。通用 API 能接受 2 个 latent，不代表此权重训练过该条件。
2. **控制和初始状态缺失。** 旧适配把整个 `[12,384]` 清零，既没有真实动作，也丢掉官方首行状态。文本无法等价替代。
3. **采样时间不同。** 原生 GR1 数据路径为 20→10 fps，旧 benchmark 路径使用 16 fps；后续真实动作重定向需同步修正时间尺度。

额外补测：episode 0，真实动作不变，仅把条件改为前 5 帧／2 latent。相同未来帧 5–12 上，解码 MP4 后计算的 PSNR/SSIM：首帧条件 **23.68 / 0.862**，5 帧条件 **16.76 / 0.687**，后者画面明显变形。结果见 `context_comparison.json`、`context5_preview.png`。

另发现官方评测拼接后续块时取 `[:12]`，会重复条件帧并丢掉块尾。本次同时保存原拼接版本 `*_official_stitch.mp4`；主结果取 `[1:]`，确保预测时间与 GT 对齐。首轮结果不受此差异影响。

## 去哪里看

根目录：`/mnt/vdc_5t/model/acwm/DreamDojo/official_validation/`。

- `episode_00000{0,1,2}/real_actions_vs_gt.mp4`：左 GT、右真实动作预测。
- 同目录 `zero_actions_vs_gt.mp4`：左 GT、右全零动作预测。
- `episode_000000/context5_real_actions_vs_gt.mp4`：5 帧历史问题对照。
- `results.json`：原生测试指标、版本与参数；`video_validation.json`：所有视频完整解码校验。

下载脚本：`_support/dreamdojo_official_download.py`（进程级公司代理、固定数据版本）。推理脚本：`_support/dreamdojo_official_validate.py`；5 帧补测：`_support/dreamdojo_official_context_check.py`。

本机复跑：`bash /mnt/vdc_5t/model/acwm/DreamDojo/official_validation/run.sh`；补测用同目录 `run_context.sh`。分别使用 GPU 0、1，执行前确认空闲。推理日志为 `run.log` 和 `context_check.log`。

后续先建立首帧＋正确 GR1 动作的参考基线；严格视频 context 方案需要另行适配/训练并验证，不能把原生单帧条件冒充视频历史。
