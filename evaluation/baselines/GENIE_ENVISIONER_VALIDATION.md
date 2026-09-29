# GE-Sim 官方样例复现（2026-09-24）

本报告仅覆盖 GE-Sim，不能代表 GE-Base。GE-Base-slow 已单独启动下载和官方样例队列，入口 `ge_base.py`，状态 `/mnt/vdc_5t/model/acwm/GE-Base/queue_status.json`；下载/推理未结束前不计为已复现。

使用 Genie-Envisioner-V1 仓库提供的 `gesim_video_gen_examples/sample_0/1/2`，复现 GE-Sim Cosmos v0.1 原生推理。**这些输入包没有未来 GT；本次验证加载、完整生成与画面表现，不计算 PSNR/SSIM，也不宣称达到论文指标。**

## 输入与运行

- 上游 commit：`d41358f7548f2cd9fc4b79fcdf7d86d15a8d7b1e`；调用原始 `infer_gesim.infer`，不改模型、条件构造或记忆更新。
- 每个样例：head／hand_left／hand_right 各 4 张历史图像；300×16 双臂动作；各相机 300×4×4 外参及 3×3 内参。
- 权重：`ge_sim_cosmos_v0.1.safetensors`，配套 Cosmos-Predict2-2B-Video2World 的 T5、VAE、scheduler。
- 官方默认：384×512／相机、4 个历史条件、每块 25 帧、15 步采样、guidance=1、30 fps；默认通用质量 prompt，不添加任务答案。
- 只替换本机权重路径，另外显式固定 Python／NumPy／Torch seed=42（上游 `seed` 参数没有真正设置 RNG）。包装器仅截获结果并另存纯 RGB。
- 首块后按官方方式从已有历史＋预测中均匀取 4 帧作为记忆，继续使用自身预测，不注入未来真实图像。

## 本次实测

3 个样例全部完成，每个 12 个生成块，耗时约 392 秒（GPU 0、1、2 并行）。权重缺失键、额外键、形状不匹配均为空。18 个导出视频完整解码和帧数/fps 校验通过。

抽帧检查：sample_0 的壶与夹爪场景大致保留；sample_1 可见面包从烤面包机区域移向盘子；sample_2 可见夹爪接近瓶子并向垃圾桶区域移动。但有局部形状变化、人物纹理失真及跨相机物体状态不完全一致，不能据此宣称操作成功或物理过程准确。

上游末块还有一个时间处理细节：296 帧 future 不是 25 的整数倍，最后只有 21 条未来控制。原脚本的补齐数量公式会得到负数，未实际补齐；pipeline 将这 21 条条件插值为完整生成块，再裁剪总输出。本轮保留这一官方行为，末段不能视为严格逐帧控制对齐。现有 benchmark 的 `genie_video.py` 已在输入侧补齐到整块再裁剪。

## 输出说明

结果根目录：`/mnt/vdc_5t/model/acwm/Genie-Envisioner/official_validation/`。

每个 `sample_0`、`sample_1`、`sample_2` 目录：

- `prediction.mp4`：去掉最初 4 帧历史的三相机 RGB 拼图，左→右为 head／hand_left／hand_right。296 帧，30 fps，约 9.87 秒。
- `prediction_head.mp4`、`prediction_hand_left.mp4`、`prediction_hand_right.mp4`：单相机预测。
- `upstream/video.mp4`：官方原始输出，300 帧；**上为三相机 RGB，下为动作轨迹图，不是 GT**。
- `rgb_with_context.mp4`：含最初 4 帧输入的 RGB。
- `preview.png`：三相机抽帧，第一列为最后输入帧，后三列为预测。
- `manifest.json`：配置、输入 SHA256、版本及生成状态；`progress.json`：块进度；`run.log`：日志。

根目录 `results.json` 保存完整解码校验，缺少 GT 的精度指标明确标为未评估。

## 与旧 benchmark 适配的差异

1. **夹爪数值尺度有误。** 官方轨迹图使用 `gripper / 120` 映射颜色；官方样例约为 35–123。旧适配直接写入手指开合的 0–1 值（case 1 实际约 0.23–1.0），除以 120 后几乎没有变化。需要按机器人夹爪语义校准后转换，不能把 0–1 原样传入。
2. 官方为三个真实相机及随末端运动变化的腕部相机外参；旧适配复制单个人视角和相同 head pose，缺少真实腕部视角和几何关系。
3. 官方动作为机器人基座坐标系中的左右末端 xyz＋xyzw＋夹爪值；旧适配使用合成中性位姿叠加人手运动，尚无基座/末端标定。
4. 官方首组历史为连续 4 帧，之后稀疏更新记忆；旧适配起始就从整段前缀均匀取 4 帧。旧动作采样/导出为 16 fps，官方样例默认 30 fps，时间尺度需独立检查。

这些差异需要逐项验证；官方样例跑通不代表人手 benchmark 适配已经成立。

## 复跑

入口：`_support/genie_official_validate.py`；分析：`_support/genie_official_analyze.py`。本机脚本将样例编号同时作为 GPU 编号，运行前确认该卡空闲：

```bash
bash /mnt/vdc_5t/model/acwm/Genie-Envisioner/official_validation/run.sh 0
bash /mnt/vdc_5t/model/acwm/Genie-Envisioner/official_validation/run.sh 1
bash /mnt/vdc_5t/model/acwm/Genie-Envisioner/official_validation/run.sh 2
/mnt/vdc_5t/model/acwm/.tools-env/bin/python /mnt/vdb_1t/project/acwm/acwm_data_rsi_system/mem_bench/baselines/_support/genie_official_analyze.py
```
