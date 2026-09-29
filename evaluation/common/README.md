# 公共工具

- `config.py`：API、抽帧和评分权重的配置校验；本机配置为 `../configs/local.toml`，可分享模板为 `../configs/example.toml`。
- `video.py`：真实 PTS 视频读取、均匀抽帧和图像保存。
- `vlm.py`：VLM 请求、最多 8 图、重试和原始记录；只负责调用，不决定 benchmark 指标。
- `schema.py`：JSON 的通用严格字段约束。

不在这里加入任务或评分逻辑。
