# Ctrl-World 官方样例验证（2026-09-24）

**结论：官方三相机＋真实机器人状态的 replay 能生成跟随轨迹的运动，三个样例都优于保持初始位姿的对照。当前 benchmark 适配还有明确的时序差异，不能把其效果直接当作原生模型效果。** 本次是推理链路检查，不是论文指标完整复现。

## 怎么验证

- 使用仓库自带 `dataset_example/droid_subset` 和官方默认样例：899（移动香蕉）、18599（从椅子向桌上搬物体）、199（移动绿色毛巾）。起始帧分别 8、14、8。
- 上游 commit：`99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d`；上游工作区无修改。
- 实际执行 `scripts/rollout_replay_traj.py` 原始 main 循环；只替换本机权重/输出路径、固定 seed=0，并截获函数返回的预测帧供分析。模型 `load_state_dict` 严格加载通过。
- 三个真实相机、真实 7 维 Cartesian pose＋gripper 状态、原始文本；50 步去噪、guidance=1、每轮 5 帧、推进 4 帧、12 轮，共 49 帧（含起始位置）。原始历史缓冲/latent 更新保持不变。
- 对照仅把机器人状态序列替换为初始状态重复，其他输入和种子相同。该对照不是归一化全零动作；模型仍可能漂移，并不保证静止。
- 未来 GT latent 在原脚本中只用于生成后的对照图解码，不参与预测条件。

## 结果

对原始 GT 计算逐帧 PSNR，再对三相机取平均；不计起始帧和超出源视频的重复补帧。

| 样例 | 有效预测帧 | 真实动作 PSNR | 保持位姿 PSNR | 静止首帧基线 PSNR |
|---|---:|---:|---:|---:|
| 899：移动香蕉 | 48 | 20.21 | 14.42 | 15.91 |
| 18599：椅子到桌上搬物体 | 48 | 20.50 | 15.41 | 16.41 |
| 199：移动绿色毛巾 | 24 | 21.28 | 16.69 | 17.55 |

抽帧检查可见，真实动作预测的机械臂运动、接近物体的位置更接近 GT；仍有背景漂移、物体外观/位置偏差。不能仅凭 PSNR 宣称抓取放置成功或物理交互准确。

官方 199 样例只有 33 帧，从第 8 帧开始最多有 25 个不同源帧。原脚本会将最后一帧和状态重复到 49 帧；原生结果完整保留，但对照视频与指标截到有效区间。

原脚本保存 4 fps，源视频实际为 5 fps；分析导出按源时间使用 5 fps。`config.fps=7` 是原生 SVD 条件参数，本次保持默认，不等同于源采样率。

## 与我们旧适配的差异

1. **块边界错位风险明确存在。** 官方 5 帧包括当前时刻，下一轮只推进 4 帧；旧 `ctrl_video.py` 每次将全部 5 帧视作新的 future，且动作窗口从下一时刻开始，与原生重叠语义不一致。
2. **历史条件时间布局不同。** 官方从“每块末尾”的缓冲取 `[0,0,-8,-6,-4,-2]`，保留起始锚点与间隔历史；旧适配只取最近 6 个连续 RGB 帧。Ctrl-World 确实训练了 6 个历史条件，问题是时间布局和内容。
3. **滚动状态表示不同。** 官方直接保留生成 latent；旧适配每轮解码成 RGB 后重新 VAE 编码，初始编码也用了 `mode()` 而官方用 `sample()`。这些差异尚未逐项消融。
4. **观测和控制仍是代理。** 旧适配单个人视角复制为三个相机，使用训练统计中点作为合成初始机器人状态，再叠加腕部位移/掌面旋转/指尖距离；不具备真实 DROID 相机和机器人基座标定。

因此应先修正块边界与原生 latent 历史机制，再单独验证手部到 DROID 的重定向；仅增加生成长度不能解决上述问题。本轮保留 benchmark 旧输出，供后续对比。

## 视频与复跑

结果根目录：`/mnt/vdc_5t/model/acwm/Ctrl-World/official_validation/`。

- `<899|18599|199>/comparison.mp4`：**上：原始 GT；中：真实动作预测；下：保持位姿对照**。每行从左到右为相机 0、1、2；已去掉超出源视频的补帧。
- 同目录 `prediction_vs_raw_gt.mp4`：上 GT、下预测。
- 同目录 `prediction.mp4`、`gt_raw.mp4`：49 帧版本；`preview.png`：相机 0 抽帧预览。
- `upstream/Rollouts_replay/video/`：未改动的官方输出拼图，GT 行是 VAE 重建而非原始 RGB，4 fps。
- `results.json`：逐相机指标；`settings.json`：参数；`run.log`：日志。保持位姿运行原始产物在平级 `official_validation_hold/`。

验证入口：`_support/ctrl_official_validate.py`；分析入口：`_support/ctrl_official_analyze.py`。

```bash
# GPU 0：原生 replay；GPU 1：保持位姿对照，运行前确认空闲。
bash /mnt/vdc_5t/model/acwm/Ctrl-World/official_validation/run.sh
bash /mnt/vdc_5t/model/acwm/Ctrl-World/official_validation_hold/run.sh
/mnt/vdc_5t/model/acwm/.tools-env/bin/python /mnt/vdb_1t/project/acwm/acwm_data_rsi_system/mem_bench/baselines/_support/ctrl_official_analyze.py
```
