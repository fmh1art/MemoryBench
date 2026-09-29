# ACWM 关键帧标注 demo

只需标注 5 个关键帧：前缀开始、t／动作开始、动作结束、评估开始、评估结束。无需填写名称、物体描述、gate 条件、证据段、参考框或备注。

**一个 case 视频可以标注多个独立的 evaluation 样本。** 每个样本有自己的 ID、5 个边界和保存版本，导出时每个样本单独一行，可切换已有样本继续修改。支持多个摄像头视角。

程序目录：`/mnt/vdb_1t/project/acwm/acwm_data_rsi_system/mem_bench/ground_truth/annotation`。入口为 `server.py`，也可执行 `./run.sh`，默认复用 MemWorld 的 Python 环境；用 `ACWM_ANNOTATOR_PYTHON` 指定其他兼容环境。

默认数据：`/mnt/vdc_5t/data/acwm/memory-office`（已检查 68 个 case、75 路 RGB 视频）。

默认输出：`/mnt/vdd_15t/exp_output/acwm/memory_annotations`。原始数据保持只读；程序不占用 GPU。

## 前缀 / 生成片段简化模式

启动时加 `--annotation-mode prefix_generation`，分别标注前缀开始、前缀结束、生成开始、生成结束，默认结束／开始联动，可取消。
不同样本可复用区间；每条的两段可不连续或重叠，构建 case 时顺序拼接为连续时间轴。保存为 `acwm.prefix_generation_episode.v2`，不会伪造动作／评分边界。
旧五边界模式仍是默认值。eval case builder 已支持按源时间排序且不重叠的独立区间；完整生成段作为评估区间，未单独标注的动作结束保持 null。
本批 39 条录制的页面、保存位置与使用方法见
[trans / drawer 标注说明](/mnt/vdc_5t/data/acwm/260921_ego_request/annotations/README.md)。

## 打开 / 启动

当前服务器服务名：`acwm-memory-annotator.service`，端口 **8890**，监听 `127.0.0.1`。服务由 systemd 管理，关闭 Codex session 不会停止它；重启服务器后需要重新启动。

远程访问可以在 VS Code 的「端口」面板转发 **8890**，或在自己的电脑执行：

```bash
ssh -N -L 8890:127.0.0.1:8890 <登录用户名>@<服务器地址>
```

然后在自己电脑浏览器打开 **http://127.0.0.1:8890**。SSH 转发只影响访问通道，断开后已保存标注仍在服务器。

手动启动一个实例：

```bash
cd /mnt/vdb_1t/project/acwm/acwm_data_rsi_system/mem_bench/ground_truth/annotation
/mnt/vdb_1t/project/acwm/MemWorld/.venv/bin/python server.py \
  --data-root /mnt/vdc_5t/data/acwm/memory-office \
  --output-root /mnt/vdd_15t/exp_output/acwm/memory_annotations \
  --host 127.0.0.1 --port 8890
```

运行中的 systemd 实例已经占用 8890 时，不要重复运行同端口实例。查看 / 停止：

```bash
systemctl status acwm-memory-annotator.service
systemctl stop acwm-memory-annotator.service
```

依赖为`/mnt/vdb_1t/project/acwm/MemWorld/.venv` 已有的 FastAPI / Starlette（FileResponse 支持 Range）、Uvicorn、Pydantic 2、PyAV、Pandas、PyArrow、NumPy、Pillow。前端是静态 HTML / CSS / JS，无需 npm 构建，也不依赖外部 CDN。推荐支持 H.264 与 `requestVideoFrameCallback` 的现代 Chrome / Edge。

## 一次标注的操作

1. 左侧选择 case，右上选择视角（有 left 时默认 left）。
2. 定位视频并标记 5 个关键帧。前缀开始默认第 0 帧；t 同时是前缀结束与最终动作开始边界。
3. 「动作结束」与「评估开始」之间可以保留过渡段；评估段应选择物体可判读的稳定片段。
4. 点击「完成标注」。只检查这 5 个边界是否完整、顺序正确且在视频范围内，不要求其他内容。未标全可以「保存草稿」。
5. 继续标注同一个视频时点击「新建样本」；用「评估样本」下拉框切换和编辑已保存样本。新样本不会覆盖已有样本，各样本的前缀或评估区间允许重叠。
6. 点击「导出推理清单」或「导出关键帧标注」；每个已完成样本独立输出一条 JSONL 记录，草稿不会被导出。

倍速可选 0.25×、0.5×、1×、2×、4×、8×、16×、32×。浏览器不支持原生高倍速时使用跳帧快览（例如 Chrome 的 32×）；暂停后恢复精确帧显示，适合先快速找事件、再逐帧标边界。

空格播放 / 暂停；左右方向键逐帧；Shift + 左右跳 1 秒；数字键 1–5 标记五个边界；Ctrl / Cmd + S 保存草稿。输入框内打字不会触发标记快捷键。

切换 case / 视角 / 样本前会保存已修改内容为草稿。浏览器本地也暂存修改，意外刷新后恢复。保存采用原子替换，保留旧版本；不同页面覆盖同一标注会收到版本冲突提示。

## 时间与区间约定

所有帧号从 **0** 开始，区间为 **左闭右开** `[start_frame, end_frame_exclusive)`。结束位置是第一个不包含的帧，而不是最后一个包含的帧。

```text
prefix_start                  t          action_end    eval_start       eval_end
     |------ 历史 RGB --------|------ 最终操作 ------|--- 过渡 ---|--- 稳定评估 ---|
     |------ 历史动作 --------|-------------- 未来动作 + head pose ------------|
                             |------------------- 生成视频 ------------------|
                                                            |-- 物体评分 --|
```

其中 `prefix.end = action.start = generation.start = t`。RGB 最后可见帧 / 首次预测的 anchor 是 **t−1**；首个要预测的帧是 **t**。不能把 t 帧的 GT RGB 当作 anchor，否则会多看到一帧未来。图示的操作段可比文字宽度短或长，时间边界以保存的帧号为准。

本工具不要求人工把边界凑成 32 的倍数。后续推理适配器应按 32 帧一段滚动预测，必要时延长未来动作条件再裁切生成结果；不能用未来 GT RGB 填充 anchor / memory。

WMData 的 `meta/extents.parquet` 同时提供 MP4 `video_frame_pts` 和 RGB `video_timestamp_ns`。两者通过**帧序号**一一对应。视频控制使用真实 MP4 PTS；动作 / head pose 对齐使用传感器时间。

例如首个 case 有 20,030 帧，MP4 播放约 667.667 秒，传感器跨度约 713.662 秒。因此绝不能用「播放时间 × 30」去查手部传感器行。部分后续 case 还是可变帧率视频。暂停时的图像通过 PyAV 从关键帧解码到**完全相同的 PTS**，解码失败时不拿相邻帧冒充。

视频结束边界允许等于帧数 N。N 不是一张真实帧，其 sensor 时间没有精确采样值，因此 JSON 保存 `sensor_timestamp_ns: null` 和单独的 `sensor_timestamp_ns_estimate`（最后一次采样加中位帧间隔），不会把估计值当真实传感器时间。

## 文件与接口

```text
memory_annotations/
  episodes/<event-id>.json           # 每个评估样本的最新草稿 / 完成标注
  history/<event-id>/<revision>.json # 修改前的历史版本
  exports/inference.jsonl            # 点击导出时更新；只有完成的样本
  exports/evaluation.jsonl           # 点击导出时更新；只有完成的样本
  server.log                        # 部署服务的运行日志
```

`GET /api/cases` 列出 case；`GET /api/annotations?case_id=...` 读标注；`POST /api/annotations` 保存；`GET /api/export/inference` 与 `/api/export/evaluation` 导出。接口字段见 `/api/docs`。

关键帧 JSON 保存原始视频身份（相机、stream / file ID、checksum）、样本 ID、五个边界和保存状态。通过校验的记录另保存每个边界的真实 PTS / sensor 时间与派生区间。新记录标记 `annotation_scope: "keyframes"`，不写入空的描述性标注字段。

已有旧版标注仍可打开。历史物体描述、gate、证据段和参考框保留在数据里，不再显示或要求编辑，也不会因为这些旧内容与新时间边界不一致而阻止完成关键帧标注。若修改了含旧辅助内容的样本边界，记录 `auxiliary_annotations_need_review: true`，供后续评估者复核。旧证据段不再处于新前缀内时，不输出错误的派生记忆时长。
推理 JSONL 使用字段白名单，只保留来源、历史 RGB 区间、历史 / 未来动作和 head pose 区间、生成范围、anchor 和空文本条件。物体答案、gate 文字、参考框、证据段不会进入模型条件。来源仍指向整个 WMData case，执行推理时必须按清单中的区间读 RGB。

**标注导出的 JSONL 是区间级推理请求格式，不是 `sample_pt` 的 `index.json`。** 已根据确认的 5 个样本完成适配脚本，产物和推理入口见 [样本构建说明](../evaluation/case_definition/README.md)。`../../case_definition/build_evaluation_cases.py` 提取原始区间及独立手骨架，`../../case_definition/pack_evaluation_cases.py` 用真实 Wan VAE / 文本编码器生成可供当前推理使用的 `sample_pt_v2`，`../../pipeline/infer_evaluation_cases.py` 负责从 t 开始滚动预测并裁出评估片段。保留历史及未来 head pose。

## 与后续评估的关系

此页面只负责定位前缀、动作段和稳定评估段，完成标注表示时间边界就绪。后续评估仍遵循 `memory_i = gate_i × object_match_i`：先判断生成动作是否实现、目标区域是否被揭示；gate 不通过时 memory 为 0，通过后在稳定评估段判断物体匹配。gate / object score 默认都是 `null`，由评估程序另行判定，不需要在此页面填写。

前缀长度和动态记忆的代码核查见 [MEMORY_PREFIX_LIMITS.md](../evaluation/pipeline/MEMORY_PREFIX_LIMITS.md)。
