"""Render the existing v2 scores without changing their definition."""

import csv
from ..metric.schemas import METRICS


def write_report(root, status):
    lines = [
        "# Memory Evaluation v2",
        "",
        f"完成 {status['completed_count']}/{status['case_count']}，错误 {status['error_count']}。",
        "",
        "人工任务标注；6 帧生成动作 Gate；GT 与生成稳定片段各均匀抽 8 帧，每对两张图直接 VLM judge；三维按配置权重汇总。",
        "",
        "| Case | Action | Gate | 物体清单 | 外观 | 空间关系 | Memory |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def num(x):
        return "N/A" if x is None else f"{x:.4f}"

    for r in status["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"[{r['evaluation_case_id']}]({r['evaluation_case_id']}/comparison.mp4)",
                    r["annotation"]["action_description"],
                    num(r.get("gate")),
                    *(num(r.get(k)) for k in METRICS),
                    num(r.get("memory_score")),
                ]
            )
            + " |"
        )
    for r in status["cases"]:
        if r.get("review_warning"):
            lines += [
                "",
                f"复核提示 · {r['evaluation_case_id']}：{r['review_warning']}",
            ]
    lines += [
        "",
        f"平均 Memory：{num(status['mean_memory_score'])}，平均值分母 {status['aggregate_denominator']}。",
        "",
        "Gate 失败时 Memory=0，内容维度未评估并显示 N/A；GT 没有适用事实的维度也显示 N/A，剩余正权重归一化。",
        "对象身份不成立时不得获得物体外观、状态或关系信用，普通 inside-target 关系不计空间分。",
        "旧版 v1 的类别匹配分数与 v2 含义不同，不能将数值变化直接解释为模型能力变化。",
        "程序验证只检查输入、协议和计算一致性，VLM 的具体物体识别仍可在逐对图片与原始响应中复核。",
        "",
        "[协议说明](PROTOCOL.md) · [配置](run_config.json) · [程序验证](validation.json) · [指标 CSV](metrics.csv)",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    columns = [
        "evaluation_case_id",
        "status",
        "gate",
        *METRICS,
        "memory_score",
        "evaluation_seconds",
        "error",
    ]
    with (root / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(status["cases"])


def write_generation_report(root, status):
    """Progress report while ACWM generation and evaluation run in parallel."""
    rows = status["cases"]
    finished = [r for r in rows if r["status"] == "completed"]
    errors = [r for r in rows if r["status"] == "error"]
    columns = [
        "evaluation_case_id",
        "status",
        "gpu",
        "gate",
        *METRICS,
        "memory_score",
        "inference_seconds",
        "evaluation_seconds",
        "error",
    ]
    temporary = root / "metrics.csv.tmp"
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "metrics.csv")
    lines = [
        "# ACWM Memory benchmark",
        "",
        f"完成 {len(finished)}/{len(rows)}；错误 {len(errors)}。",
        "",
        "| Case | 状态 | Gate | 清单 | 外观 | 空间关系 | Memory |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def number(value):
        return "—" if value is None else f"{value:.4f}"

    for r in rows:
        link = f"[{r['evaluation_case_id']}]({r['evaluation_case_id']}/comparison.mp4)"
        lines.append(
            "| "
            + " | ".join(
                [
                    link,
                    r["status"],
                    number(r.get("gate")),
                    *(number(r.get(k)) for k in METRICS),
                    number(r.get("memory_score")),
                ]
            )
            + " |"
        )
    lines += [
        "",
        f"已完成样本平均 Memory：{number(status['mean_memory_score'])}，分母 {len(finished)}。",
        "",
        "读取人工动作与目标标注；Gate 使用生成动作段的 6 帧，内容使用双方稳定段各 8 帧组成图片对直接评分。",
        "完整 GT 与生成视频、双方标注稳定段均保留供复核。Gate 不通过时 Memory=0；接口/参考无效错误保留为空，不计成模型 0 分。",
        "评估协议见 [PROTOCOL.md](PROTOCOL.md)。",
        "逐对图像判分、Gate 和原始请求可在各 case 的 evaluation/pairs、gate.json 与 vlm_calls 中审计。",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
