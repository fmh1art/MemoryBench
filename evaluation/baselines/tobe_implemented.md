模型	推荐版本	类型	重点测评	选择原因	开源情况 / 注意事项	模型参数 / 权重链接	是否已复现
Hand2World	AR Open Version	第一人称手物交互	L0 / L1 / L2	与 Ego Video、Hand Pose、Camera Pose 和物体交互场景最接近	已公开代码及 AR 版本权重；公开版与论文完整系统存在差异	https://github.com/NTUYWANG103/Hand2World	
DreamDojo	2B GR1 Post-train	机器人操作	L0 / L1 / L2	支持真实机器人动作条件，适合测试物体移动与状态更新	推荐使用 GR1 post-train，而非仅 latent-action 预训练版本	https://huggingface.co/nvidia/DreamDojo/tree/main/2B_GR1_post-train	
Ctrl-World	DROID	机器人操作	L0 / L1 / L2	输入历史视觉信息与机器人动作，适合交互式 Memory 测试	已公开代码、配置和模型	https://github.com/Robert-gyj/Ctrl-World	
Genie Envisioner	GE-Base-slow v0.1（基础模型）；GE-Sim 单列动作控制对照	机器人世界模型	L0 / L1 / L2	GE-Base 用历史视觉与指令预测未来，需单独验证	GE-Base 正在下载；GE-Sim 官方样例已跑通，不能代替 GE-Base 的复现	https://github.com/AgibotTech/Genie-Envisioner-V1	
iVideoGPT	VP2 RoboDesk	Action-conditioned Video WM	L0 / L1 / L2	离散自回归架构，可作为 Diffusion World Model 的架构对照	原生时间窗口较短，长时实验需要 autoregressive rollout	https://github.com/thuml/iVideoGPT	
WorldMem	Minecraft	显式 Memory World Model	L1 / L2	专门研究长期世界状态记忆及交互后的状态更新，是 Memory Benchmark 的核心基线	已公开代码和模型，与本 Benchmark 高度相关	https://xizaoqu.github.io/worldmem/	
Oasis	500M	Minecraft World Model	L0 / L1 / L2	与 WorldMem 处于相近游戏环境，可比较普通 WM 与显式 Memory WM	权重存在访问条件	https://github.com/etched-ai/open-oasis	
HyDRA	Official Checkpoint	Camera-conditioned World Model	L1 / L2*	支持历史视频和相机轨迹，适合场景回访及长期视觉状态保持	L2 更适合 Observed Update，不适合主动手物操作	https://github.com/H-EmbodVis/HyDRA	
Matrix-Game 3.0	5B Distilled	Interactive Game WM	L0 / L1 / L2	支持交互式长视频生成和相机控制，适合测试长时间状态一致性	推荐使用已公开的 5B Distilled 版本	https://github.com/SkyworkAI/Matrix-Game	
