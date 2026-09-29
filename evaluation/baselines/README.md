# World model baselines

**2026-09-23：9 个新输入适配路径已通过短段生成测试；Oasis 另完成一个完整区间。Matrix 的外部历史仍退化为末帧；动作语义和 Memory 指标尚未评估。**

**正式 benchmark 输入适配与运行方式：[INPUT_ADAPTATION.md](INPUT_ADAPTATION.md)。** 下文旧结果主要是加载/生成 smoke test；不能等同于 benchmark 对齐。HyDRA 旧四段视频是官方示例，新的 benchmark 路径使用真实 case。

权重根目录：`/mnt/vdc_5t/model/acwm`。原始仓库在 `origin_scripts/`，10 个仓库的来源和固定 commit 记录于 [`_support/repositories.json`](_support/repositories.json)。没有把待完成清单标记为“已复现”。

当前已经完成：仓库克隆、原有 10 个入口（NWM 已停用，当前计划 9 个）、真实 office case 的格式转换及 CPU 输入校验。**Ctrl-World、iVideoGPT、WorldMem、Oasis、HyDRA、GE-Sim 和 Matrix-Game 3.0 已实际生成可解码结果；其余模型继续下载、安装和验证。** 输入语义依据见 [INPUT_REFERENCES.md](INPUT_REFERENCES.md)。 后台测试会分别记录安装失败、下载受阻、推理失败和真正生成了可解码视频的成功状态。

## 后台任务和状态

```bash
tmux attach -t acwm-baselines-download  # Ctrl-b d 分离，不停止下载
cat /mnt/vdc_5t/model/acwm/download_status.json
cat /mnt/vdc_5t/model/acwm/smoke_status.json
```

- `acwm-baselines-download`：依清单顺序下载；文件级续传，某个模型失败后继续下一模型。日志 `/mnt/vdc_5t/model/acwm/download.log`。
- `acwm-baselines-worldmem`：安装 WorldMem 独立环境并运行 office 样例。
- `acwm-baselines-configure`：等待当前续传队列结束，补下 Hand2World 的 bidirectional teacher 和已获许可的 Oasis，再依次安装和测试 Hand2World、Oasis、HyDRA、Matrix-Game。只选 GPU 0–3 中空闲显存超过 70GB 且低利用率的卡，一次一个模型；日志 `smoke-<脚本名>.log`。环境安装和推理分别设 2 小时上限，失败后继续。
- 环境均在 `/mnt/vdc_5t/model/acwm/.envs/`，不改已有训练环境。辅助转换环境 `.tools-env/`。
- 代理配置参考 `/mnt/proxy.sh`。公司 HTTP 代理为 `http://100.68.170.171:3128`，已验证能访问 Hugging Face。续传任务仅设置自身进程的大小写代理环境变量，不执行脚本中会停止共享 mihomo 服务的 company 切换。ModelScope 仍直连。
- **ModelScope 例外**：GE-Sim 下载子进程清除全部大小写代理变量，设置 `NO_PROXY=no_proxy=*`，Requests `trust_env=False`；CDN 重定向同样直连。已探测直连 HTTP 200。
- **访问状态（2026-09-21）**：Oasis 已下载。DreamDojo 所需 Cosmos-Predict2.5-2B tokenizer、GE-Sim 所需 Cosmos-Predict2-2B-Video2World VAE 均已用当前凭据实测 HTTP 206，依赖下载和测试已重启（`download-permissions.log` / `configure-permissions.log`）。NWM 申请被拒，已移出计划。
- 上轮 Oasis 采样完成但视频保存因 PyAV API 不兼容失败，修复为 `av==12.3.0` 后排队重测；Matrix-Game 的强制 Flash Attention 路径已由进程内 SDPA 适配，数值对照及 office 样例推理均通过。Hand2World 的 Google Drive 公司代理请求返回 504，尚未补齐 teacher。

重新下载某一模型（先确认原下载队列已结束，避免同时写同一文件）：

```bash
/mnt/vdc_5t/model/acwm/.tools-env/bin/python _support/download_models.py Oasis
```

安装及测试队列也可在原队列结束后重新执行 `_support/finish_and_smoke.py`。`downloaded` 仅表示下载完成；`smoke_passed` 仅表示此适配样例推理成功，**不是论文指标复现**。

## 调用

从本目录执行，例如：

```bash
# 无需权重：生成样例、模型输入、适配说明和 command.json
/mnt/vdc_5t/model/acwm/.tools-env/bin/python ivideogpt.py --prepare-only

# 权重和对应环境准备好后：实际运行
/mnt/vdc_5t/model/acwm/.tools-env/bin/python ivideogpt.py --gpu 0

# 使用已有的模型专用环境
/mnt/vdc_5t/model/acwm/.tools-env/bin/python hydra.py --python /path/to/env/bin/python --gpu 1
```

所有入口支持 `--case`、`--start`（秒）、`--output`、`--python`、`--gpu`、`--prepare-only`。支持动作输入的模型另有 `--actions model_native.npy`；动作形状见下表和文件头说明。模型本体在独立子进程中执行，官方代码目录设为工作目录及 `PYTHONPATH`。

默认样例：`memory-office/QUEST-20260806-memory-office__qv-02bb3439c90d7cfc`，从 2 秒起取约 10 秒，按 16Hz 取 161 帧。RGB 使用 WMData extents 中的捕获时间及视频 PTS，绑定同一 camera_frame_id 的位姿/内参；位姿最近邻误差不超过 50ms，失效位姿直接报错。Rotation6D 按列向量恢复 optical camera-to-reference，单位米。

默认输出：`/mnt/vdc_5t/model/acwm/baseline_runs/<模型名>/`。其中 `input/manifest.json` 记录源帧和时间戳，`compatibility.txt` 记录适配，`command.json` 记录实际命令，`run_status.json` 记录执行退出码。GIF/MP4 保留各官方模型的输出约定。

| 调用文件 | 版本与输入适配 | 输出 / 限制 |
|---|---|---|
| [`hand2world.py`](hand2world.py) | AR；首帧 + 161 帧相机 DA3 JSON + 对齐的左右手骨架视频 | 骨架替代 MANO 渲染是近似条件；不是等价手部外观 |
| [`dreamdojo.py`](dreamdojo.py) | 2B GR1；首帧 + `[12,384]` 原生归一化动作，默认全零 | 13 帧；Cosmos tokenizer 已获许可，依赖下载中；使用仓库实际注册的 GR1 OSS 配置，绕开不存在的文档别名和 sudo 转换 |
| [`ctrl_world.py`](ctrl_world.py) | DROID；单视角复制三份、历史重复；`[11,7]` 笛卡尔状态，默认训练统计中点 | 5 帧三视角拼图；直接调用官方 agent，绕开硬编码 DROID case |
| [`genie_envisioner.py`](genie_envisioner.py) | GE-Sim Cosmos v0.1；四帧历史、复制三相机；`[29,16]` 双臂 xyz+xyzw+gripper，默认合成静止位姿 | 25 帧生成及条件可视化；Cosmos 依赖已获许可，下载中；没有真实腕部相机；官方输出 30fps，会将 16Hz 输入重新定时 |
| [`ivideogpt.py`](ivideogpt.py) | VP2 RoboDesk；64×64、2 历史 + 10 预测；`[12,5]`，默认零 | 官方 GT/预测并排 GIF；长序列仍需另做多窗口 rollout |
| [`worldmem.py`](worldmem.py) | Minecraft；360×640、10Hz；16 历史 + 16 预测；`[32,25]`，默认零 | 相机近似映射 xyz/pitch/yaw；丢弃 roll；进程内数据适配跳过固定 100 帧偏移 |
| [`oasis.py`](oasis.py) | 500M；首帧缩放 360×640；`[32,25]`，默认无按键 | 32 帧；已获许可，下载/测试排队中 |
| [`hydra.py`](hydra.py) | 官方 checkpoint；77 历史 + 77 未来相机；直接传相对 20×12 矩阵 | 绕开 Unreal 厘米/轴转换；模型读取 480×832 中心裁剪视频；未来 RGB 不作条件 |
| [`matrix_game.py`](matrix_game.py) | 3.0 5B distilled；首帧，官方离线随机控制，seed 42 | 一轮 3 步采样；随机控制不是 office 轨迹；A800 禁用 FA3/int8 |

机器人和 Minecraft 控制在 office case 中不存在，默认替代量只用于检查加载与生成链路；不能据此报告真实动作条件 L0/L1/L2 成绩。Hand2World 的 MANO 渲染、机器人控制重定向和游戏动作映射仍需更严格的领域适配才能用于可比测评。

CPU 验证：

```bash
/mnt/vdc_5t/model/acwm/.tools-env/bin/python _support/validate_inputs.py
```

下载来源以各原始仓库 README 为准；主模型包括 [Hand2World Google Drive](https://drive.google.com/drive/folders/1J37kycnFz8jvkyN0Wkm9vrsXmHjqBms-)、[DreamDojo](https://huggingface.co/nvidia/DreamDojo/tree/main/2B_GR1_post-train)、[Ctrl-World](https://huggingface.co/yjguo/Ctrl-World)、[GE-Sim ModelScope](https://modelscope.cn/models/agibot_world/Genie-Envisioner)、[iVideoGPT](https://huggingface.co/thuml/ivideogpt-vp2-robodesk-64-act-cond)、[WorldMem](https://huggingface.co/zeqixiao/worldmem_checkpoints)、[HyDRA](https://huggingface.co/H-EmbodVis/HyDRA)、[Matrix-Game 3.0](https://huggingface.co/Skywork/Matrix-Game-3.0)。

NWM 的已有代码和历史日志保留供追溯，不再下载、安装或测试；历史 JSON 中的失败记录不计入当前 9 个模型的进度。

## 2026-09-21 兼容修复

- Matrix-Game 3.0：官方 `--fa_version=0` 未覆盖两个直接调用 Flash Attention 的路径。`_support/matrix_infer.py` 在推理进程内改用 SDPA，保留有效长度、缩放及窗口/因果掩码；通过显式 attention 数值对照，实际生成 `baseline_runs/Matrix-Game-3.0/prediction.mp4`。
- DreamDojo：安装固定版本 PyTorch3D v0.7.9 的纯 PyTorch transforms，禁用本次不需要的 C++/CUDA ops；旋转往返测试通过。注册阶段不再下载最终被显式 GR1 checkpoint 覆盖的基础权重，直接推理移除未注册的验证数据配置；tokenizer 改用官方仓库可用的固定 revision（入口文件头有记录）。仍需 Cosmos-Reason1-7B 文本编码器；后台 `acwm-dreamdojo-fix` 正在下载并继续测试，尚未标记成功。
- Hand2World：用户自行下载 Google Drive 中的 `V_0.9/bidirectional.safetensors`，放入 `/mnt/vdc_5t/model/acwm/Hand2World/checkpoints/V_0.9/`。当前仍缺此文件。
