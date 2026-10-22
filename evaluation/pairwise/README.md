# MemoryBench 人工 A/B/Tie 评估

每个样本只作一个综合选择：**A / B / Tie**。标注选项和计分参考用户提供的 *Training Object Permanence in World Models* §4.4 与 Table 2。保留 MemoryBench 原来的 History Video、GT Future、Text Prompt、两个匿名 prediction，以及同步播放、断点续评和多人分配。

## 与论文对齐的判断规则

论文 §4.4 只有 A、B、about the same 三个选项，将三条标准**联合考虑**，不分别评分：文本描述一致性、自然运动与物理合理性、物体恒存性。当前 demo 同样只收集一个直接选择，不要求 Gate 或子维度判断，不自动推导胜负，也没有备注或“无法判断”选项。

页面提示：

> 综合考虑文本一致性、运动与物理合理性，以及遮挡前后物体状态是否正确。两者整体表现相近时选 Tie。

GT 用于参考目标状态，允许合理的视角和动作轨迹差异。History 可独立回看；GT/A/B 可同步播放、拖动、慢放和全屏。素材无法播放时可展开故障报告，报告不计为 A/B/Tie 投票，也不视为 Tie。

按用户要求，**输入保持不变**：仍展示 GT，使用相同的原始视频、文本与时间偏移。本次不改变分辨率/FPS，不删除 GT，也不把它声明为对论文全部数据和众包流程的复现。

## 本机访问

人工服务监听服务器的 `127.0.0.1:8920`。在 VS Code 转发 **8920**，或从自己电脑执行：

```bash
ssh -N -L 8920:127.0.0.1:8920 <用户名>@<服务器>
```

然后打开 **http://127.0.0.1:8920**。更新表单后请刷新页面。

服务使用：

- 全量研究清单：`evaluation/pairwise/runs/human-all-15-20261022/study.json`
- 投票数据库：`/var/lib/memorybench-human/votes.sqlite3`
- 15 个非 tail 模型、649 个 case、57,577 个比较组合；同 case 全配对，允许模型覆盖不齐。随机有放回出题，不设个人次数或单题人数上限。
- 原 demo 的 9 道比较及其历史标注保留。GE-SIM、DreamDojo 分别显示对应仿真参考。研究扩充后需刷新整个网页；未提交草稿会继续原题，提交后从全量题库随机抽取。

服务以 root 独立后台进程运行，会话退出后继续运行，机器重启后需重新启动。用 `ps -eo user,pid,args | grep '[e]valuation.pairwise.server'` 查看实际 PID 和参数。重启使用下述备份迁移工具，并传入当前 `--study` 路径；已有服务运行时不要启动同端口实例。实际部署状态以 `http://127.0.0.1:8920/api/config` 和 `/var/lib/fmh/runtime/memorybench-human-status.json` 为准。

如果本地端口 8920 被旧转发占用，在 VS Code 将该行的本地端口改为 18920，再打开 `http://127.0.0.1:18920/`。服务器端口仍是 8920。页面显示题库规模；全量部署应为 15 个模型、649 个 case、57,577 个组合。

## 安装与启动

Python >=3.11；准备清单需要 `ffprobe`。无需 GPU、VLM/API、前端构建或外部 CDN。

```bash
python -m pip install -e '.[pairwise]'
# 或已有项目环境：
python -m pip install -r evaluation/pairwise/requirements.txt

python -m evaluation.pairwise.prepare \
  --model Model-1=/path/to/model1.cases.json \
  --model Model-2=/path/to/model2.cases.json \
  --output evaluation/pairwise/runs/study.json

python -m evaluation.pairwise.server \
  --study evaluation/pairwise/runs/study.json \
  --db evaluation/pairwise/runs/votes.sqlite3 \
  --host 127.0.0.1 --port 8920
```

本机可用 `/mnt/vdb_1t/project/acwm/MemWorld/.venv/bin/python` 运行。也可使用 `PAIRWISE_PYTHON=/path/to/python bash evaluation/pairwise/run.sh --study ... --db ...`。

登记评估者代号后立即开始；首次登记显示私密续评码。同一浏览器自动续评，换设备可输入续评码。选择 A/B/Tie 后点击提交，草稿自动保存。原始路径、模型名和其他人的选择不向评估者提供。

多人共享一个服务实例。若通过受邀内网直接访问，可显式用 `--host 0.0.0.0`；HTTPS 反代加 `--secure-cookie`，由可信反代正确传递 scheme/host。代号与续评码用于受邀评估和续评，不作为实名鉴权或防止重复注册的身份系统。

组内多人都需要参与时，启动参数增加 `--open-assignment`。该模式允许每位评估者独立完成所有比较一次，其他人的任务占用或已达标的单题人数不会阻止领取；`raters_per_pair` 保留为研究的目标人数。原有投票、草稿、匿名 A/B 映射和续评码继续使用，不改变 study 签名。同一评估者完成全部比较后会明确显示个人完成状态，不会重复投票。

本机 root 启动的 8920 人工 demo 使用 `/var/lib/memorybench-human/votes.sqlite3`。在 **root 终端**从项目根目录执行以下命令，可切换开放领取并沿用原数据库（将 PID 换成实际运行该 demo 的进程号）：

```bash
/var/lib/fmh/venvs/memorybench-pairwise/bin/python \
  -m evaluation.pairwise.restart_demo --pid <现有demo的PID>
```

工具会核对 PID、study 和数据库，在线备份后停止旧服务，再做一次包含最后提交的备份，使用同一数据库重启并验证旧投票、评估者身份和任务映射仍在。备份保存在数据库旁的 `backups/`，只含计数的重启结果写入 `/var/lib/fmh/runtime/memorybench-human-status.json`。重启后每位同学刷新页面即可继续，无需重新登记。数据库目录仍保持 root 私有。

全量服务再次重启时增加 `--study evaluation/pairwise/runs/human-all-15-20261022/study.json --random-assignment`，确保沿用当前完整题库和随机模式。只查看某位评估者的提交次数及不同 case/比较数，可运行 `python -m evaluation.pairwise.restart_demo --stats-rater '评估者代号'`；该操作只读，不重启，也不输出续评码或投票选择。

### 导出人工记录并对照 VLM Elo

正式 SQLite 保存姓名和续评凭据，不能直接提交到数据仓库。在 root 终端用 `python -m evaluation.pairwise.restart_demo --export-human <新的快照目录>` 导出：同一只读事务取得全部已提交记录，按实际 A/B 映射还原模型，用稳定 HMAC 代号替代评估者身份，排除姓名、续评码、会话密钥、原始评估者 ID、逐票精确时间及草稿。保留每次重复判断的独立记录和耗时；不改动在线数据库。

```bash
python -m evaluation.pairwise.human_report \
  --vlm-run /path/to/MemoryBenchPredictionsV1/.metadata/evaluation_runs/pairwise_20261021_terra_merged \
  --human-snapshot /path/to/human-snapshot \
  --output /path/to/new-comparison
```

输出 `comparison.csv/json`、`vlm_only/`、`vlm_plus_human/` 和说明。沿用相同 Bradley–Terry 实现并核对 VLM 点估计与既有榜单一致，逐票等权，保留人工重复，不加权、不多数投票；本工具只计算点估计，不生成置信区间。VLM 使用无 History 的抽帧拼图，人工使用带 History 的完整视频；合并分数是混合观看协议的阶段性统计。

人工导出是注明北京时间的累计快照，不会随每次在线提交自动更新。后续快照应使用新目录；只选择一个人工快照及一个最新合并 VLM run，不能将重叠的历史 run 再次相加。逐视频 VLM Gate/维度评分和播放故障不属于两两偏好，不能并入 Elo。

## 输入清单与抽样

### 全量非 tail 人工题库与随机出题

`--random-assignment` 从完整 case/model-pair 集合中等概率、有放回抽取，不设个人次数或单题人数限制，自动包含开放领取。A/B 每次重新随机，重复比较保存为不同记录。刷新、重连或并发标签页继续同一份未提交草稿；提交请求重试不重复计票。原数据库的 `(rater,task)` 唯一约束在事务内移除，原有行全部保留。

准备全量题库使用 `python -m evaluation.pairwise.human_study`，参数为 `--predictions`、`--data`、`--jobs`、`--source`（旧 study）、`--sim-data`（MemoryDataV1-Sim 本机目录）和 `--output`（新的 study 路径）。只排除 `_tail`；不限制 case 数，不按自动评估结果筛选；同一 case 下有效预测全部两两配对，允许模型覆盖不齐及短预测。逐个用 ffprobe 校验视频，缓存 path/size/mtime 对应时长，`study.audit.json` 记录数量及不可用素材。

GE-SIM 和 DreamDojo 使用各自仿真 GT，通过匿名 A/B 对应的独立播放器展示。缺失的仿真 GT 写入审计并阻止部署，不用真实 GT 替代。原九题的预测、提示、历史、GT 和评估范围保持不变。

包含仿真参考时，同步播放覆盖 GT/A/B 中最长的实际视频段，短视频停在末帧，避免较短的真实 GT 提前截断仿真预测；独立参考播放器可单独回看。不改变原分辨率、帧率或时间速度。纯真实视频配对继续沿用 GT future 时间轴。

准备成功后，在 root 终端执行：

```bash
python -m evaluation.pairwise.restart_demo --pid <现有demo的PID> \
  --new-study /path/to/new/study.json --random-assignment
```

工具核验旧任务含义未变，在线备份、停止服务后再次备份，更新 study 元数据，沿用原数据库、浏览器会话、续评码及草稿，再启动随机模式。健康检查失败时恢复旧研究元数据并启动旧命令。以线上 `/api/config` 的 `random_assignment`、`models`、`cases`、`total_tasks` 及重启回执确认实际部署状态。

下文的单人不重复规则适用于普通配额及 `--open-assignment` 模式；随机模式下，所有重复判断独立保留并计分。

全量验收工具 `python -m evaluation.pairwise.tests.full_smoke --study ... --predictions ... --output <新测试目录> [--browser-python <装有Playwright的Python>]` 会对照完整预测索引检查 case/model 联集、全部配对，并逐模型对验证领题、Range 视频读取、草稿、提交及重试。浏览器检查覆盖所有模型、真实/仿真参考组合、刷新续评和移动端布局。所有脚本投票仅写入新建的 `SCRIPTED-VALIDATION-ONLY.sqlite3`，不能作为人工偏好或合并进正式数据库。

输入格式兼容现有 `evaluation/cases.example.json`，可额外提供 `history_video`、`text_prompt`。仅访问明确传入的清单及直接引用的素材，不递归扫描项目或磁盘。

- 取所有模型共同覆盖的 case；GT、时间偏移、历史及评估范围必须一致，真实/仿真参考分开建 study。
- 默认 History 为 GT 相邻 `../input/rgb_prefix.mp4`；其他布局显式提供 `history_video`。GT 应为完整 `ground_truth/future.mp4`。
- 预测应预先去掉 anchor、历史重播和 padding；`gen_start_seconds` 可排除前置段。保持原始比例与时间速度，不独立拉伸时长。
- `--limit 30 --seed 42` 在交集中按 IID/OOD 比例抽样；两类均存在且额度 >=2 时各保留至少一个。重复 `--case-id CASE` 可指定精确范围。
- IID 仅依据 case_id 包含字面字符串 `mine`，其余为 OOD。
- 默认全模型两两比较；`--baseline Model-1` 只与共同基线比较。默认 `--raters-per-pair 3`。
- 多次生成的选择规则应预先固定，工具不按评分挑选预测。

文本优先级为 `--prompts` 显式覆盖 > 清单 `text_prompt` > 可选 `--adapter-prompts` > 未记录。覆盖格式 `{"模型": {"case_id": "实际文本"}}`；显式 null 表示已核实无文本条件。`--adapter-prompts` 只从对应 `prediction/other_files/adapter.json` 读取 prompt，要求 model/case_id 均完全匹配。不会从 GT 注释或数据集通用说明猜测模型实际文本。不同模型的文本不同时按匿名 A/B 分开展示。

手工清单见 [study.example.json](study.example.json)。`duration_seconds` 是原文件结束时间，`start_seconds` 是共同实验 t=0 的文件内偏移。GT/A/B 的统一时间轴覆盖完整 GT future，标记评估段；短预测到实际结束处停留并标注时长差异。使用浏览器可解码的 MP4，推荐 H.264/yuv420p。

可重建本机 demo（使用新的输出文件名）：

```bash
python -m evaluation.pairwise.prepare \
  --model DWM=/mnt/vdc_5t/data/world-model-benchmark/evaluation_qwen38_20261017/DWM.cases.json \
  --model Hand2World=/mnt/vdc_5t/data/world-model-benchmark/evaluation_qwen38_20261017/Hand2World.cases.json \
  --model RynnWorld-Teleop=/mnt/vdc_5t/data/world-model-benchmark/evaluation_qwen38_20261017/RynnWorld-Teleop.cases.json \
  --adapter-prompts --limit 3 --seed 42 \
  --output evaluation/pairwise/runs/demo/new-study.json
```

## 计分：论文 §4.4 / Table 2

[论文 §4.4](https://arxiv.org/html/2609.28654v1#S4.SS4) 使用 blind pairwise comparison；Table 2 给出虚拟平局正则化说明。本实现独立实现其中的统计方法。

1. 每次实际判断等权：胜 1、平 0.5、负 0。使用原始投票，不先多数投票，不按 case 重新加权，不由 Gate 派生结果。
2. 拟合 Bradley–Terry 极大似然模型：`P(i beats j)=sigmoid(s_i-s_j)`。
3. 每个模型增加**总共一场虚拟平局**，均匀分给对手。在包含 M 个模型的连通比较图中，每个无序模型对增加 `1/(M-1)` 场平局，各方获得半胜，保证每个模型总共一场虚拟比赛。
4. 转为 Elo：`1500 + 400/ln(10) × (s_i-mean(s))`，连通模型集合的全局均值为 1500。
5. 进行 **1000 次按评估者分组 bootstrap**：有放回抽样评估者，每次保留被抽中者整组判断，再拟合 BT。报告 95% 百分位置信区间及第一名概率；并列最高者平分该次第一名概率。
6. 同时报告原始 **Score rate = (wins + 0.5 × ties) / actual games** 和投票数；虚拟平局不计入原始得分率或实际投票数。

比较图不连通时，只报告各分量内部的名次与第一名概率，各自居中到 1500，不输出跨分量排名。无有效比较的模型保持 unrated。少于两名有效评估者时不输出 bootstrap CI/第一名概率；区间条件于所选 case 和观测比较图。默认按 Overall/IID/OOD 分别出同一种综合得分，不再有 Inventory/Appearance/Spatial 排行榜。播放故障不是投票；同一评估者不会重复比较同一 case/model-pair，不安排自身模型对比。

```bash
python -m evaluation.pairwise.report \
  --study evaluation/pairwise/runs/demo/study-with-prompts.json \
  --db evaluation/pairwise/runs/demo/votes.sqlite3 \
  --output evaluation/pairwise/runs/abtie-report \
  --bootstrap 1000 --seed 42
```

输出 `votes.json/csv`、`issues.csv`、`leaderboard.csv`、`pairs.csv`、`summary.json` 和离线可看的 `report.html`。原始票保留评估者、case_id、匿名位置到模型的映射、唯一 choice 和耗时。未达到全研究配额时明确标记阶段性结果。没有 Gate 或分维度导出，也没有其他计分模式。

## 保存与协议隔离

SQLite WAL 与事务保证并发配额；同一评估者不能重复投票。A/B 顺序对不同评估者随机分配，刷新后固定。30 分钟租约、每分钟续租，关闭页面超时后可由其他人补充；revision 防止不同标签页覆盖，网络重试不重复计票。视频通过私密会话和不透明 URL 提供，支持 HTTP Range。

研究清单和视频 path/size/mtime 指纹绑定数据库，修改素材或 cohort 时新建研究。输入清单版本保持不变，单独保存标注协议 `memorybench.pairwise.wrop_ab_tie.v1`。旧版数据库若没有已提交投票，可保留参与者和 A/B 映射升级，旧多维草稿清空并增加 revision；若已有旧版投票，则拒绝自动合并，要求新建数据库。旧导出目录也不能被新协议覆盖。

不要公开数据库、原始 manifest 或管理员导出目录。`runs/` 与本地数据库已忽略，不把身份、续评码、视频和结果提交进 Git。

## GPT5.6 抽帧评估

模型评估沿用同一个 `Store.save` 和 A/B/Tie 计分协议，使用独立研究清单和数据库，
评估者的 **id 和 name 均为 `gpt5.6`**。研究 provenance 标记其为模型判断，
不会描述为真人投票。每次请求只发送 GT、A、B 三张 2×4 拼图；每张包含 8 帧、
实际时间戳和视频时长，不发送 History。History 文件只为兼容研究清单保留引用。

每个视频在自己的可用 future 内均匀采样，最长截取到 GT 结束时刻；保留纵横比。
短视频不延长到 GT 时长，不能把各拼图相同格位置当成相同时间。若片段不足 8 个
不同帧，会重复最近的真实帧并记录 `unique_frames`，不生成虚构画面。
文本条件来自生成记录；缺失时明确标记“未记录”，不由 GT 注释猜测。

```bash
python -m pip install -e '.[pairwise,pairwise-model]'

# 从已同步的评估 jobs 创建固定的 1000 条比较清单。
python -m evaluation.pairwise.model_study \
  --jobs /path/to/MemoryBenchPredictionsV1/.metadata/evaluation_runs/qwen38_20261018/jobs.json \
  --predictions /path/to/MemoryBenchPredictionsV1 \
  --data /path/to/MemoryBenchDataV1 \
  --output evaluation/pairwise/runs/gpt5.6-20261020 \
  --count 1000 --seed 42

# 仅生成首条的三个拼图，检查 inputs/*.json 中的路径，不调用 API。
python -m evaluation.pairwise.model_judge \
  --run evaluation/pairwise/runs/gpt5.6-20261020 \
  --api-config /path/to/gpt5_6_sol_hydra.yaml --preview

# 首条真实调用；验证后去掉 --limit 1，完成其余任务。相同命令也用于续跑。
python -m evaluation.pairwise.model_judge \
  --run evaluation/pairwise/runs/gpt5.6-20261020 \
  --api-config /path/to/gpt5_6_sol_hydra.yaml --limit 1
```

配置从本机 YAML 读取 `api_type: chat`、`openai_base_url`、`api_version`、`key`、
`llm_name`、`max_parrel_requests`；密钥不写入结果。模型须与研究声明一致，
本次为 `gpt-5.6-sol`。可选 `price_yuan_per_million_token` 用于按返回 usage 估算费用。

准备脚本取真实视频模型的共同案例，默认排除 GT 不相同的 GE-SIM、DreamDojo。
先校验选中视频的 SHA256，再按固定种子均衡抽取模型对，按共同案例的 IID/OOD
比例分配，尽量均衡案例出现次数。`provenance.selected_task_ids` 限定本次任务，
报告也以这份清单计算完成度；不设置该字段时仍按原来的完整模型组合工作。
A/B 位置使用数据库密钥随机确定并持久化，续跑保持不变。

输出包含 `study.json`、`plan.json`、`verified_sources.json`、`prompt_sources.json`、
`inputs/`（逐题完整 prompt 和三图信息）、`sheets/`（拼图与实际帧时间）、
`responses/`（逐次响应、usage、校验结果）、`votes.sqlite3` 和 `report/`。
`rater_session.local.json` 含该评估者的私有续评码，权限为 0600。
有效响应先落盘再入库，已入库的题目跳过；格式错误、超时和 API 失败不会作为 tie。
一个有效评估者不产生按评估者 bootstrap 的置信区间。

### 真实 GT 与仿真 GT、定额追加比较

`model_study` 默认去除所有以 `_tail` 结尾的模型；`model_judge` 在发送请求前
拒绝包含 `_tail` 的待评任务。历史批次和原始投票保留，新榜单使用去除这些模型后的研究。

`extend_study` 在新目录导入原批次不含 `_tail` 的有效投票，保留用户 ID、A/B 映射
与选择，并为 DreamDojo、GE-SIM 添加真实 GT 和对应仿真 GT。一方为仿真模型时
有 4 张图，两方均为仿真模型时有 5 张图；每张仍为 8 帧，不发送 History。
各预测按自己的参考时长抽帧，并使用相应任务描述，避免用真实 GT 的结束时间截断
较长的仿真任务。渲染风格差异本身不计为预测错误，预测相对于自身 GT 的状态偏差仍计入。

```bash
python -m evaluation.pairwise.extend_study \
  --source evaluation/pairwise/runs/gpt5.6-20261020 \
  --output evaluation/pairwise/runs/gpt5.6-20261021-dual-gt \
  --predictions /path/to/MemoryBenchPredictionsV1 \
  --sim-data /path/to/MemoryDataV1-Sim \
  --sim-per-pair 8 --seedance-per-pair 200 --seed 20261021

python -m evaluation.pairwise.model_judge \
  --run evaluation/pairwise/runs/gpt5.6-20261021-dual-gt \
  --api-config /path/to/gpt5_6_sol_hydra.yaml
```

清单在 API 调用前固定，并写入 `preregistration.json`。Seedance 三个版本在同一批
200 个案例上各比较一次，总计 600 条；按原研究共同案例中的 IID/OOD 比例抽样，
避开已有的 Seedance 比较案例。仿真模型对按固定配额均衡抽样，缺少可校验 GT 的
案例记录在排除清单中。抽样只读取已有任务身份，不依据选择结果挑选案例、改 prompt、
追加配额或提前停止。假设不会发送给 judge；排序按实际完整结果计算。

新的 `provenance.model_references` 为 `case_id → model → {clip, action_description,
memory_target, success_criterion}`。参考文件也计入研究指纹；`reference_models` 中
的模型缺少自己的 GT 时拒绝运行。`allow_partial_models: true` 允许同一研究中
部分模型仅覆盖部分案例，任务只能引用该案例实际可用的预测；旧研究默认仍要求完整共同案例。
有定额加样时，整体排行继续按每条实际投票等权拟合，并应另报新增匹配案例的直接胜平负。

## 验证

```bash
python -m pip install pytest
python -m pytest evaluation/pairwise/tests/test_pairwise.py -q

# Playwright 仅用于可选的真实浏览器流程验证。
python -m pip install playwright
python -m playwright install chromium
python evaluation/pairwise/tests/browser_smoke.py \
  --study evaluation/pairwise/runs/demo/study-with-prompts.json \
  --output evaluation/pairwise/runs/new-abtie-browser-check
```

浏览器验证自行启动临时服务与全新测试数据库，不接受现有正式服务 URL。测试检查每题只有三个选项、实际视频解码与同步、选择草稿和刷新恢复、移动端、27 次完整配额判断及 1000 次 bootstrap 导出。脚本生成的选择仅测试管道，不是人工偏好。

可用 `--server-python /path/to/python` 与 `--chromium /path/to/chrome` 指定已有环境。旧 `runs/browser-check-*` 是修改前协议的测试留档；当前三选一协议使用 `runs/abtie-browser-check-01/` 的 receipt、截图和报告。
