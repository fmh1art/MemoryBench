# Benchmark 输入适配

目标：`历史视频 context + 历史/未来动作 → t 之后的视频`。

**2026-09-24 GE-Sim 核验：** 使用仓库 sample_0/1/2 做原生验证，见 [GENIE_ENVISIONER_VALIDATION.md](GENIE_ENVISIONER_VALIDATION.md)。已发现旧适配夹爪数值未转为原生尺度，三相机和机器人坐标也仍是近似；现有 benchmark 输出不代表适配正确。

**2026-09-24 Ctrl-World 核验：** 官方三样例 replay 与保持位姿对照已完成。旧适配的 5 帧块步进、连续 RGB 历史及重新编码方式与原生不同，需先修正；见 [CTRL_WORLD_VALIDATION.md](CTRL_WORLD_VALIDATION.md)。

**2026-09-24 DreamDojo 更正：** 当前 GR1 checkpoint 的训练配置只使用 1 个条件 latent（首帧）。此前 5 帧／2 latent 路径虽然能生成文件，但在官方真实动作样例上也出现明显变形，不能视为验证通过的视频历史适配。官方首帧＋真实动作对照与详细结果见 [DREAMDOJO_VALIDATION.md](DREAMDOJO_VALIDATION.md)。

使用已有 `evaluation_cases/<case>/`，不再随便取一段视频当正式测试。RGB 只来自 `[prefix_start,t)`；最后真实帧是 `t−1`。未来允许读取手部、head pose 和**动作描述**，不读取未来 RGB、物体清单或成功答案。动作文本优先采用已有 `memory_annotation.json`：例如“打开目标抽屉”，不加入“抽屉里有苹果”。

## 怎么运行

每个模型仍用原来的单文件入口。先准备输入可用 `--prepare-only`：

```bash
cd /mnt/vdb_1t/project/acwm/acwm_data_rsi_system/mem_bench/baselines
PY=/mnt/vdc_5t/model/acwm/.tools-env/bin/python
$PY oasis.py --benchmark-case ../evaluation_cases/eval_001_a17f6882 --prepare-only
$PY oasis.py --benchmark-case ../evaluation_cases/eval_001_a17f6882 --gpu 1
```

其他模型替换 `oasis.py` 即可。默认预测到 `eval_end`；`--prediction-frames 8` 只做短段检查，不能拿来评完整 case。默认结果在 `/mnt/vdc_5t/model/acwm/benchmark_runs/<模型>/<case>/`。

- `input/context.mp4`：真正提供给模型的历史；**不一定覆盖整段前缀**。
- `input/prompt.txt`：动作文本，可用 `--prompt` 指定动作说明。
- `input/game_actions.npy` / 模型专用动作文件：转换后的控制。
- `input/adaptation.json`：历史源帧、未来动作源帧、采样率、转换规则和限制。
- `prediction.mp4`：完整模式下与 `ground_truth/future.mp4` 的帧数、fps、播放时长一致；只含预测段。
- `prediction_native.mp4`：完整滚动生成的原生帧率视频，保留供检查。
- `output_alignment.json`：实际解码核验结果，以及每个输出帧对应的原生预测帧。

## 各模型怎么适配

| 模型 | 视频 context | 动作适配 | 需要知道的限制 |
|---|---|---|---|
| **Hand2World** | 默认最近 77 帧；VAE 编码后以干净 latent 预填充 AR 的 KV 缓存，再从 t 生成 | 原历史/未来手部条件图＋相机轨迹＋动作文本 | **实验性新增路径**；历史长度须为 `4k+1`，不改变原训练 KV 窗口；手骨架仍不是原生 MANO 网格 |
| **Oasis** | 默认最近 16 帧，真实视频；可 `--context-frames` 调整 | head pose → WASD/鼠标；打开/关闭抽屉 → 操作期间保持 `use` | 原模型窗口 32 帧；增加前缀不能得到无限记忆；游戏 use 不等于真实开抽屉；无文本接口 |
| **WorldMem** | 默认最近 32 帧，作为初始记忆；后续使用自身预测 | 同 Oasis 的控制＋相机位姿/时间 | Minecraft checkpoint；真实运动到游戏动作尺度未标定，丢弃相机 roll |
| **HyDRA** | 77 帧真实历史，之后每块 77 帧自回归 | 历史/未来相机轨迹＋动作文本 | 没有显式手控制；文本仅尝试表达操作。新路径直接调用单样例 API，避开旧入口默认跑官方 examples 的问题 |
| **Ctrl-World** | 最近 6 帧真实历史，之后滚动使用预测 | 右手腕位移＋掌面旋转＋拇指/食指距离 → DROID 位姿/夹爪；附动作文本 | 单视角复制成三视角；初始机器人状态取训练范围中点并限幅，没有标定机器人基座或 IK |
| **GE-Sim** | 从允许的整段前缀均匀取 4 帧；原生记忆 rollout | 双手腕位移/掌面姿态/指尖距离 → 双臂末端位姿/夹爪；附动作文本 | 稀疏帧可能漏掉关键事件；单视角复制三相机，初始臂位姿是合成的 |
| **DreamDojo** | 旧适配最近 5 帧／2 latent；已发现超出该 checkpoint 的训练条件 | 旧适配为全零 384 维＋文本，没有真实 GR1 动作与初始状态 | **旧结果仅证明生成和长度，不是有效适配**；原生首帧＋真实动作已另行验证，见专项报告 |
| **iVideoGPT** | 最近 2 帧，每次预测 10 帧，再滚动 | 右腕 xyz 增量、掌面 yaw 增量、指尖开合 → 5 维控制代理 | 无文本接口；平移以 5cm、旋转以 0.2rad 缩放并限幅，尚未机器人标定 |
| **Matrix-Game 3.0** | **目前仅使用 t−1 图像**；后续有自身生成记忆 | 真实相机运动 → WASD/鼠标，动作意图 → 文本，已去掉随机控制 | **降级基线，不能当作使用完整视频历史的同条件模型**；不支持直接手部控制 |

## 转换口径

- 所有采样按 `aligned_conditions.npz` 的 sensor 时间选择；历史选择严格限制在 t 之前。用源帧号读取已经导出的 CFR 前缀视频，记录源帧对应关系。
- 先按各模型原生时间采样率生成整个 sensor future 区间；短窗口模型用自身预测滚动，不读取未来真实图像。完整输出再按 GT 源帧的 sensor 时间戳，选择最近预测帧，按 GT 的 fps 导出。低帧率模型会重复最近预测帧，这是采样率转换，不是用短视频补齐未生成的未来；原生帧数不足直接报错。源视频导出为 CFR，播放时长可能与 sensor 时长略有差异，所以不能仅改 fps 或按比例拉长。
- WASD：将相邻 head pose 的位移变换到前一相机坐标系；前后/左右速度超过 `0.05m/s` 才按键。鼠标由相对 pitch/yaw 得到；Oasis/WorldMem 按 `20°` 归一化，Matrix 按原生 `15°` 比例换算。
- 游戏交互：当前已有标注中的 `open_drawer/close_drawer` → `use`；`turn_view` 只用相机运动。未知操作不擅自映射成攻击/放置/丢弃。文本接口仍可使用现有动作描述。
- 机器人控制：腕部位移、掌面旋转、指尖距离是**可解释的控制代理**，不是精确重定向。必须保留这项实验条件，不能宣称机器人与真人执行了相同轨迹。
- Hand2World 只在 t 之前强制写入观察到的 latent；未来段按原 AR 采样。历史超出原 KV 窗口仍会被遗忘，不改模型架构或扩大训练窗口。

## 验证与还需要什么

2026-09-23：9 个适配路径均通过第一个真实标注 case 的 8 帧生成与完整解码测试（Matrix 仍属末帧降级输入）。旧 `smoke_status.json` 不表示 benchmark 对齐。

输入适配测试独立于旧的加载测试：看 `/mnt/vdc_5t/model/acwm/benchmark_adapter_status.json`，日志和短视频在 `benchmark_adapter_tests/<模型>/`。`short_case_passed` 只说明新适配生成了 8 帧；完整 case 和动作 Gate 通过之前不算 benchmark 完成。

5 个现有标注 case × 9 个模型的 45 份完整输入包已准备并通过边界校验，见 `/mnt/vdc_5t/model/acwm/benchmark_input_validation.json`。

Oasis 另已完成第一个 case 的完整区间生成：202 帧、20fps，结果 `/mnt/vdc_5t/model/acwm/benchmark_rollout_tests/Oasis/prediction.mp4`；完整解码通过，尚未评动作 Gate / Memory 分数。

当前已有的 `case.json`、手部/head pose 和人工动作标注足够启动；无需重新提供相同标注。新操作若没有明确动作描述，需要补**动作本身**，不能用预测区间的物体答案充当输入。严格机器人控制还需本体映射/基座标定；更长历史需验证原生窗口或新增记忆接法，不能靠重复末帧冒充。

## 完整 future 批量生成

2026-09-23 启动 9 模型 × 5 case 的完整预测。结果：`/mnt/vdc_5t/model/acwm/benchmark_full_length/<模型>/<case>/prediction.mp4`。进度在同目录根部 `status.json`；只有 `full_length_verified` 表示完整原生生成及与 GT 帧数/fps 校验都通过，尚不表示动作语义或记忆评测通过。每项日志为 `run.log`。

后台 tmux：`acwm-full-future`。可续跑队列（已验证结果自动跳过）：

```bash
/mnt/vdc_5t/model/acwm/.tools-env/bin/python _support/run_full_benchmark.py
```

默认使用空闲的 GPU 0–3；每卡一项。单模型完整运行省略 `--prediction-frames`，该参数仅用于短测。
