# 输入适配依据

以 `tobe_implemented.md` 指定的 checkpoint 为准。论文用于判断条件的物理含义，README 用于确认公开版本能力，固定 commit 的推理代码决定实际字段、维度和预处理。论文中的其他版本不能视作本次下载模型的能力。

| 模型 | 核对来源 | 对 office 适配的约束 |
|---|---|---|
| Hand2World | [论文 §3.3–3.4、附录 B](https://arxiv.org/html/2602.09600v1)、[公开版 README](https://github.com/NTUYWANG103/Hand2World/blob/4e9b0095dc9eb0731daf1e9f4a0dd26993ae8b58/README.md) | 论文手部条件是 MANO 网格投影的实心轮廓叠加线框，左右手区分颜色；当前 landmarks21 骨架缺少表面轮廓，只用于运行测试。公开 AR 版重新训练于 Wan 2.2 / ARCTIC，与论文完整系统有差异。 |
| DreamDojo | [README](https://github.com/NVIDIA/DreamDojo/blob/02f119b759d5c7f84a399fdeea3c6e82e7ed6cff/README.md)、[post-training](https://github.com/NVIDIA/DreamDojo/blob/02f119b759d5c7f84a399fdeea3c6e82e7ed6cff/docs/POSTTRAIN.md)、[evaluation](https://github.com/NVIDIA/DreamDojo/blob/02f119b759d5c7f84a399fdeea3c6e82e7ed6cff/docs/EVAL.md) | 使用指定 GR1 post-train 的原生动作接口；不能把人体手部关节点直接当作 GR1 控制，也不能把 latent-action 预训练版的输入约定套过来。默认归一化零动作是占位。 |
| Ctrl-World | [README 推理与数据准备](https://github.com/Robert-gyj/Ctrl-World/blob/99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d/readme.md) | DROID 动作 chunk 与多视角观测；单路头戴相机复制为三个视角不等同于原生多相机，统计中点状态也不是 office 的真实控制。 |
| GE-Sim | [README GE-Sim 与官方数据转换示例](https://github.com/AgibotTech/Genie-Envisioner-V1/blob/d41358f7548f2cd9fc4b79fcdf7d86d15a8d7b1e/README.md) | 需要相机内外参和双臂动作。README 的历史条件是稀疏记忆帧；当前四个相邻历史帧只作短样例，未复现训练时长范围；没有真实腕部视角。 |
| iVideoGPT | [README 模型表与推理](https://github.com/thuml/iVideoGPT/blob/d601d5cac9e96c6aa0c17cb37ed6a7c7ca1fb210/README.md) | 保持清单指定 VP2 RoboDesk action-conditioned 版；其他 OXE action-free 模型不能混用动作格式。零 RoboDesk 动作不是人体动作重定向。 |
| WorldMem | [论文 §3.2–3.5](https://arxiv.org/html/2504.12369v1)、[README](https://github.com/xizaoqu/WorldMem/blob/c1cc91bdfb2dda52fe309280e5ed046b1faba448/README.md) | Minecraft 使用 25 维游戏动作，并以位姿和时间组织记忆。零按键与 office 相机运动不构成一致的游戏控制，只能检查链路；论文 RealEstate 实验不能据此认定当前 Minecraft checkpoint 是真实场景版。 |
| Oasis | [README](https://github.com/etched-ai/open-oasis/blob/f59deef2c019c212bd0c5a3a5b986a51f3701847/README.md) | 首帧和游戏键鼠动作生成后续帧。office 没有对应游戏动作；无按键仅是可执行的占位条件。 |
| HyDRA | [论文 §4](https://arxiv.org/html/2603.25716v1)、[README](https://github.com/H-EmbodVis/HyDRA/blob/f2c666061098f24d7327968a57e253cf0a339bfa/README.md)、[相机解析代码](https://github.com/H-EmbodVis/HyDRA/blob/f2c666061098f24d7327968a57e253cf0a339bfa/infer_hydra.py) | 原生任务是历史 RGB + 历史/未来相机轨迹预测未来 RGB。相机输入为展平 R/t 的 12 维向量；适配器使用官方允许的相对矩阵入口。未来 RGB 不作为条件；office 光学坐标已是米制，不重复套 Unreal 轴变换和厘米换算。固定裁剪视野与训练视野仍有差异。 |
| Matrix-Game 3.0 | [README 推理](https://github.com/SkyworkAI/Matrix-Game/blob/71c3cd7f741311f8100f6cf9cde942b6c1378d11/Matrix-Game-3/README.md) | 官方离线例子使用随机动作，交互入口接收用户控制。当前离线样例的随机动作不代表 office 轨迹。 |

以上近似均需与每个入口文件头及生成的 `compatibility.txt` 一起阅读。`smoke_passed` 仅说明加载并生成了可解码结果，不表示输入在语义上等价，也不表示完成 L0/L1/L2 指标复现。
