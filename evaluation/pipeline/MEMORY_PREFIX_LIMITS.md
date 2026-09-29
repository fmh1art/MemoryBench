# 当前 ACWM 的视频前缀与 Memory 限制

核查日期：2026-09-11。仓库：`/mnt/vdb_1t/project/acwm/MemWorld`。

用户指定的权重实际路径：`/mnt/vdc_5t/model/acwm/memworld-step4529/dit_step4529.safetensors`。读取了 safetensors 的键与形状，包含 `context_compressor`、`context_keypoint_tokenizer`、`target_keypoint_tokenizer`、`context_time_embedding`；未进行新的 GPU 推理。权重目录中的 README 是模板，没有训练前缀时长 / 最大记忆跨度说明，safetensors metadata 只有 `format: pt`。因此不能由 step4529 推断训练时看过多长视频。

## 前缀长度与单次 forward 长度是两件事

几分钟的历史视频可以作为**建立记忆候选池的来源**，但当前默认 forward 不把完整前缀逐帧送进去，而是读取固定数量的历史 latent 槽位：

| 历史部分 | 默认槽位数 | 当前默认来源 |
| --- | ---: | --- |
| long context | 16 | 每隔 32 个源帧取一个较早历史端点，通常从当前 chunk 起点前 64 帧到前 544 帧 |
| short context | 8 | 最近约 32 个源帧，间隔 4 帧 |
| dynamic context | 8 | 从已构建的、更大历史候选集合检索，与 long / short 允许重复 |
| 合计 | 32 | 不足时补空槽；这是记忆槽，不等于连续 32 帧前缀 |

另外有 1 个 anchor latent 和 8 个待预测 latent，对应每个 section **32 张新视频帧**（解码片段包含起始帧共 33 帧），不是再增加 9 个历史检索槽。历史 RGB 与历史手骨架有对应的条件分支，未来手骨架单独进入 target 分支；保留历史及未来 head pose，符合此前确认的实验条件。

出处：[`utils/memory_context.py`](/mnt/vdb_1t/project/acwm/MemWorld/utils/memory_context.py) 的常量、`short_context_frame_indices_for_clip` / `long_context_frame_indices_for_clip`；[`preprocess/sample_pt_v2/sample_v2.py`](/mnt/vdb_1t/project/acwm/MemWorld/preprocess/sample_pt_v2/sample_v2.py) 的 `build_*_context_layout`；[`script/infer/infer_sample_pt_eval.py`](/mnt/vdb_1t/project/acwm/MemWorld/script/infer/infer_sample_pt_eval.py) 的 `prepare_sample` / 默认 `expected_*_context_length`；[`diffsynth/pipelines/wan_video_memory.py`](/mnt/vdb_1t/project/acwm/MemWorld/diffsynth/pipelines/wan_video_memory.py) 的 `forward`。

按严格 30 FPS 换算，long / short 的固定近期覆盖大约 18.1 秒；**这不是整个前缀的上限**。memory-office 的采集帧间隔并非统一 1/30 秒，真实覆盖秒数必须用 sensor 时间计算。

默认 `expected_*_context_length` 会对缺失槽位补齐，超过对应长度时报错。代码里虽有长度参数，也不能将它视作任意增大上下文就可靠的承诺；权重训练分布和位置编码兼容性都需要另行验证。当前默认位置编码模式是 `legacy_76`，不能仅因要长记忆就把旧权重换成另一个 RoPE 模式。

## Dynamic memory 能否回看很久之前？

**可以选到较久之前的历史，但前提是它已经进入候选池且仍可读取。当前没有“无限长视频、无限保留且恒定资源”的实现保证。**

当前 sample_pt_v2 预处理的 `build_historical_short_endpoint_candidates` 会从历史 sample 的 short endpoints 构建候选，过滤到当前 chunk 之前、帧索引为 4 的倍数、排除 anchor 的帧，并与 static candidates 合并。没有单独的「只允许回看最近 N 秒」限制。长视频需要完整前缀的历史 sample / latent cache 已经被建立，只有最后一个 sample 不能自动得到其余历史。

sample_pt 推理的动态选择默认分数为 `0.9 × FOV overlap + 0.1 × freshness`（freshness 默认 ratio），对 8 个目标 latent 的未来 head pose 检索，尽量使用不同候选。它主要看相机视野相似度与新鲜度，**不是按“物体曾被放进哪个抽屉”进行语义检索**。

出处：[`preprocess/sample_pt_v2/stage_3_enrich_sample_v2.py`](/mnt/vdb_1t/project/acwm/MemWorld/preprocess/sample_pt_v2/stage_3_enrich_sample_v2.py) 的 `build_frame_latent_cache`、`build_historical_short_endpoint_candidates`、`dynamic_score_rows`；[`script/infer/infer_sample_pt_eval.py`](/mnt/vdb_1t/project/acwm/MemWorld/script/infer/infer_sample_pt_eval.py) 的 `build_history_dynamic_candidates`、`dynamic_score_rows`。

当前 sample_pt 推理窗口从后半段开始时，`seed_offline_history_cache` 会遍历索引中窗口之前的 sample（并使用窗口首个 sample 中仍在窗口前的 short endpoints），用 `< window_start_frame` 过滤来预热历史 cache；首个窗口 chunk 的 dynamic context 沿用离线 sample，此后才重建 rollout 检索。它不是仅凭一个最终操作 sample 读取整个真实前缀。输入索引不完整、static 来源超出许可前缀，都会破坏所声称的记忆实验条件。

另一条 `WanVideoMemoryPipeline.__call__` 直接推理路径的 `_select_dynamic_context_records` 则使用传入的 memory latents 与历史生成帧，按 FOV overlap 检索，没有同样的 freshness 公式。区分入口，不能把不同实现的参数混为一谈。

候选越多，历史编码 / 存储、CPU 内存和检索工作越大；即使每次 forward 的 32 个槽固定，前面的检索与缓存也不是恒定成本。抽屉“放过物体”和“没有放物体”时的 head pose 可能很接近，检索仍可能漏掉真正包含物体证据的历史帧，所以应记录检索来源帧做归因。

## `memory_max_size=500` 是不是前缀上限？

**不是当前主路径中已生效的前缀上限。**

`infer_with_memory.py` / `infer_trajectory_with_memory.py` 保留了 `--memory_max_size=500`、`--memory_update_stride=4`，但目前调用的 `WanVideoMemoryPipeline.__call__` 中，这些参数被接收并写入可视化元数据，没有据此执行候选池截断 / stride 更新。该实现保留历史生成帧，检索函数遍历可用候选。

仓库还有 [`utils/memory_pool.py`](/mnt/vdb_1t/project/acwm/MemWorld/utils/memory_pool.py) 的 `DynamicMemoryPool(max_size=500)`，但当前主推理路径没有实例化它。即使单独使用这个类，也只是带淘汰机制的候选池；其初始载入与低重叠情况下的淘汰并不保证严格 500 条，更不能据此宣称能无损记住无限时长。

## 对此次 drawer memory 评估的含义

标注允许选择整个几分钟 case 内任意有效前缀。本批 75 路视频的传感器时长约 117–824 秒；标注工具已经检查其帧映射，但这不等于这些完整长度都已在 step4529 上跑过推理。

后续适配必须让 memory 的 RGB 来源严格位于 `[prefix_start, t)`，使用 t−1 为末次真实 RGB anchor；t 之后可以输入真实手部动作和 head pose，但不能将未来 RGB / GT 物体标签送进条件。更后面的预测使用模型自己的历史生成帧滚动，不能使用 `--teacher_forced_rgb` 冒充完整未来 rollout。

区间级标注不能直接当 `sample_pt` 索引。本次已加入 `build_evaluation_cases.py` / `pack_evaluation_cases.py` / `infer_evaluation_cases.py` 完成适配：严格按所选前缀建 cache、对齐 sensor 动作、显式记录 32 帧分块补帧、将生成结果裁回原始评估区间。原推理 CLI 新增 `--episode_start_clip`，保留此前的索引用于 memory 预热，同时从指定 clip 开始生成。产物说明见 [evaluation_cases/README.md](../evaluation_cases/README.md)；已用 step4529 在一个样本上完成两段低步数连续推理，验证数据与 rollout 兼容性。

为验证 memory，建议输出每个预测 chunk 检索到的原始帧号，并与人工证据段比较。主分数应使用模型自己的检索；额外提供「人工保证检索到证据」的结果时，应明确标为 oracle 对照，不能混入主结果。
