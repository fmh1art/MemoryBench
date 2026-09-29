#!/usr/bin/env python3
"""Add source-frame mappings and auditable Gate consistency notes to a finished run."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.reporting"
import argparse
import csv
from datetime import datetime
import json

import numpy as np

from ..common.vlm import write_json


def finalize_v2(root, status):
    from ..reporting.report import write_report

    adjustments = []
    for row in status["cases"]:
        output = root / row["evaluation_case_id"]
        case = json.loads((output / "case.json").read_text())
        result = json.loads((output / "evaluation/result.json").read_text())
        with np.load(output / "frame_mapping.npz") as mapping:
            times = {
                int(f): int(t)
                for f, t in zip(mapping["source_frame"], mapping["sensor_timestamp_ns"])
            }

        def mapped(frame, start, generated=False):
            original = start + frame["index"]
            return frame | {
                (
                    "conditioning_source_frame" if generated else "source_frame"
                ): original,
                "sensor_timestamp_ns": times[original],
            }

        selection = result["selected_frames"]
        write_json(
            output / "selected_frames_with_source.json",
            {
                "note": "Generated indices refer to the conditioning timeline, not guaranteed visual alignment.",
                "stable_source_interval": [
                    case["marks"]["eval_start"],
                    case["marks"]["eval_end"],
                ],
                "action_source_interval": [
                    case["marks"]["t"],
                    case["marks"]["action_end"],
                ],
                "pairs": [
                    {
                        "progress": p["progress"],
                        "gt": mapped(p["gt"], case["marks"]["eval_start"]),
                        "generated": mapped(
                            p["generated"], case["marks"]["eval_start"], True
                        ),
                    }
                    for p in selection["pairs"]
                ],
                "gen_action_images": [
                    mapped(f, case["marks"]["t"], True)
                    for f in selection["gen_action_images"]
                ],
            },
        )
        for i, pair in enumerate(result["frame_results"], 1):
            if pair.get("arithmetic_adjustments"):
                adjustments.append(
                    {
                        "evaluation_case_id": row["evaluation_case_id"],
                        "pair": i,
                        "adjustments": pair["arithmetic_adjustments"],
                    }
                )
    write_json(root / "arithmetic_audit.json", adjustments)
    write_report(root, status)
    with (root / "REPORT.md").open("a") as report:
        report.write(
            "\n逐对评分的原视频帧号与传感器时间见各样本的 selected_frames_with_source.json。\n"
        )
        report.write(
            "\n程序根据同次 VLM 返回的物体判断复算清单 F1、外观均值并执行身份约束；原始分数与差异见 [arithmetic_audit.json](arithmetic_audit.json)，原始响应保持不变。\n"
        )
        report.write(
            "\n视频、抽帧、请求和评分复算检查见 [validation.json](validation.json)。\n"
        )
    print(
        json.dumps(
            {
                "completed": len(status["cases"]),
                "pairs_with_arithmetic_adjustments": len(adjustments),
            }
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "evaluation_results",
    )
    args = p.parse_args()
    root = args.root.resolve()
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "completed"
    if str(status.get("framework_version", "")).startswith("2."):
        finalize_v2(root, status)
        return
    reviews = []
    evidence_lines = []
    lines = [
        "# Gate 复核与原始帧映射",
        "",
        "以下保留 VLM 自动评分，没有人工改写分数。Gate 定位和物体提取阶段的可读性矛盾会单独标记。",
        "",
        "| Case | 自动 Gate | 可见探测帧 / 全部探测帧 | 物体提取可读帧 / 5 | Memory | 复核提示 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in status["cases"]:
        cid = row["evaluation_case_id"]
        output = root / cid
        case = json.loads((output / "case.json").read_text())
        result = json.loads((output / "evaluation/result.json").read_text())
        loc = json.loads((output / "evaluation/localization_gen.json").read_text())
        warning = ""
        if result["gate"] and not result["generated_readable_pairs"]:
            warning = "Gate 定位判为通过，但物体提取没有可读生成帧；需要复核 Gate。"
        elif result["gate"] and result["generated_readable_pairs"] < 5:
            warning = (
                "可读生成帧少于 5 帧；缺失评分位置按框架规则计零，请复核短暂可见区间。"
            )
        row.update(
            review_warning=warning,
            gate_review_required=bool(warning),
            readable_probe_frames=loc["readable_sample_count"],
            probe_frames=loc["sampled_frame_count"],
            generated_readable_pairs=result["generated_readable_pairs"],
        )
        positive = [
            o
            for o in loc["observations"]
            if o["action_completed"] and o["target_visible"]
        ]
        review = {
            "evaluation_case_id": cid,
            "review_warning": warning,
            "gate_review_required": bool(warning),
            "automatic_gate": result["gate"],
            "automatic_memory_score": result["memory_score"],
            "readable_probe_frames": loc["readable_sample_count"],
            "probe_frames": loc["sampled_frame_count"],
            "generated_readable_pairs": result["generated_readable_pairs"],
            "positive_gate_evidence": [
                o | {"image": f'evaluation/frames/gen/{o["frame_id"]}.jpg'}
                for o in positive
            ],
        }
        write_json(output / "gate_review.json", review)
        reviews.append(review)
        mapping = np.load(output / "frame_mapping.npz")
        times = {
            int(f): int(t)
            for f, t in zip(mapping["source_frame"], mapping["sensor_timestamp_ns"])
        }
        selected = []
        for pair in result["selected_frames"]:
            gtf = case["marks"]["eval_start"] + pair["gt"]["index"]
            assert case["marks"]["eval_start"] <= gtf < case["marks"]["eval_end"]
            gen = pair["generated"]
            gf = case["marks"]["t"] + gen["index"] if gen else None
            selected.append(
                {
                    "progress": pair["progress"],
                    "gt": pair["gt"]
                    | {"source_frame": gtf, "sensor_timestamp_ns": times[gtf]},
                    "generated": (
                        gen
                        | {
                            "conditioning_source_frame": gf,
                            "conditioning_timestamp_ns": times[gf],
                            "inside_annotated_stable_interval": gf
                            >= case["marks"]["eval_start"],
                        }
                        if gen
                        else None
                    ),
                }
            )
        write_json(
            output / "selected_frames_with_source.json",
            {
                "ground_truth_input": "annotated stable evaluation clip",
                "generated_source_indices_mean": "input condition timeline; not a claim of visual correspondence",
                "pairs": selected,
            },
        )
        lines.append(
            f'| [{cid}]({cid}/comparison.mp4) | {result["gate"]} | {loc["readable_sample_count"]}/{loc["sampled_frame_count"]} | '
            f'{result["generated_readable_pairs"]}/5 | {result["memory_score"]:.4f} | {warning or "—"} |'
        )
        if warning:
            evidence_lines += [
                "",
                f"**{cid} 的 Gate 正例证据**："
                + "、".join(
                    f'[{o["frame_id"]}]({cid}/evaluation/frames/gen/{o["frame_id"]}.jpg)'
                    for o in positive
                ),
                "",
            ]
    status["gate_review_case_ids"] = [
        r["evaluation_case_id"] for r in reviews if r["gate_review_required"]
    ]
    write_json(root / "status.json", status)
    write_json(root / "gate_review_summary.json", reviews)
    lines += evidence_lines
    lines += [
        "",
        "每个 case 的 selected_frames_with_source.json 将实际评分帧映射回原始视频帧号与传感器时间；GT 全部位于人工标注的稳定段。",
        "生成帧的映射表示它所使用的动作/head pose 时间位置；评分仍按双方各自的读取进程对齐。",
    ]
    (root / "GATE_REVIEW.md").write_text("\n".join(lines) + "\n")
    columns = [
        "evaluation_case_id",
        "status",
        "gpu",
        "gate",
        "inventory",
        "appearance",
        "state",
        "spatial",
        "memory_score",
        "inference_seconds",
        "evaluation_seconds",
        "readable_probe_frames",
        "probe_frames",
        "generated_readable_pairs",
        "gate_review_required",
        "review_warning",
    ]
    with (root / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(status["cases"])
    earliest = min(
        datetime.fromisoformat(r["inference_started_at"]) for r in status["cases"]
    )
    video_done = max(
        datetime.fromisoformat(r["video_ready_at"]) for r in status["cases"]
    )
    eval_start = min(
        datetime.fromisoformat(r["evaluation_started_at"]) for r in status["cases"]
    )
    eval_done = max(datetime.fromisoformat(r["completed_at"]) for r in status["cases"])
    timings = {
        "inference_wall_seconds": (video_done - earliest).total_seconds(),
        "final_evaluation_wall_seconds": (eval_done - eval_start).total_seconds(),
        "inference_to_final_results_wall_seconds_including_diagnostics": (
            eval_done - earliest
        ).total_seconds(),
    }
    write_json(root / "timings.json", timings)
    report = (root / "REPORT.md").read_text()
    report += f'\n5 个样本推理总墙钟时间：{timings["inference_wall_seconds"]/60:.2f} 分钟；最终口径的 VLM 评估：{timings["final_evaluation_wall_seconds"]/60:.2f} 分钟。\n'
    report += "\nGate 与物体提取的可读性矛盾及图片证据见 [GATE_REVIEW.md](GATE_REVIEW.md)。自动 Gate 通过率应结合这些证据复核。\n"
    report += "\n视频物理帧数、复制校验、推理设置、GT/生成帧输入隔离和评分复算见 [validation.json](validation.json)。\n"
    (root / "REPORT.md").write_text(report)
    print(
        json.dumps(
            {
                "completed": len(reviews),
                "gate_review_case_ids": status["gate_review_case_ids"],
                **timings,
            }
        )
    )


if __name__ == "__main__":
    main()
