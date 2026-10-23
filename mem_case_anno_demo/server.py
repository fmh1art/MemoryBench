"""Frame/PTS based, read-only WMData video viewer with local episode annotations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import threading
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictInt

DEFAULT_DATA = Path(os.environ.get("WMDATA_DATA_ROOT", "data/wmdata"))
DEFAULT_OUTPUT = Path(os.environ.get("MEMORYBENCH_ANNOTATIONS", "annotations"))
HERE = Path(__file__).resolve().parent
MARKS = ("prefix_start", "t", "action_end", "eval_start", "eval_end")
LEGACY_FIELDS = (
    "title",
    "evidence",
    "target_description",
    "expected_state",
    "gate_criterion",
    "references",
    "notes",
)


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    frame: StrictInt
    # Normalized xyxy in the selected, uncropped source camera image.
    box: list[float] = Field(min_length=4, max_length=4)
    label: str = Field(default="", max_length=500)


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    case_id: str
    view_id: str
    revision: int = Field(default=0, ge=0)
    status: str = Field(default="draft", pattern=r"^(draft|complete)$")
    annotation_scope: str = Field(default="keyframes", pattern=r"^(keyframes|prefix_generation)$")
    boundaries_linked: bool | None = None
    title: str = Field(default="", max_length=500)
    marks: dict[str, StrictInt | None]
    evidence: list[StrictInt | None] = Field(
        default_factory=lambda: [None, None], min_length=2, max_length=2
    )
    target_description: str = Field(default="", max_length=4000)
    expected_state: str = Field(default="", max_length=4000)
    gate_criterion: str = Field(default="", max_length=4000)
    references: list[Reference] = Field(default_factory=list, max_length=100)
    notes: str = Field(default="", max_length=8000)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Catalog:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.cases = {}
        self.views = {}
        for case in sorted(self.root.iterdir()):
            if not case.is_dir() or not (case / "meta/files.parquet").exists():
                continue
            files = pd.read_parquet(case / "meta/files.parquet")
            views = []
            for row in files.to_dict("records"):
                if (
                    row.get("storage_type") != "video"
                    or row.get("content_type") != "rgb_video"
                ):
                    continue
                path = (case / row["relative_path"]).resolve()
                if not path.is_relative_to(case.resolve()) or not path.is_file():
                    continue
                view_id = hashlib.sha256(str(row["file_id"]).encode()).hexdigest()[:12]
                record = dict(row, path=path, case_dir=case, view_id=view_id)
                self.views[(case.name, view_id)] = record
                views.append(
                    {
                        "id": view_id,
                        "name": path.stem,
                        "width": int(row["video_width"]),
                        "height": int(row["video_height"]),
                    }
                )
            views.sort(key=lambda x: (x["name"] not in ("left", "camera"), x["name"]))
            if views:
                self.cases[case.name] = {"id": case.name, "views": views}

    def view(self, case_id, view_id):
        result = self.views.get((case_id, view_id))
        if result is None:
            raise HTTPException(404, "找不到这个 case / 视角")
        return result

    @lru_cache(maxsize=144)
    def metadata(self, case_id, view_id):
        row = self.view(case_id, view_id)
        extents = pd.read_parquet(row["case_dir"] / "meta/extents.parquet")
        extents = extents[extents.file_id == row["file_id"]].sort_values(
            "logical_sample_start"
        )
        if len(extents) != 1 or int(extents.iloc[0].logical_sample_start) != 0:
            raise HTTPException(
                422, "此 demo 需要一个从第 0 帧开始的 RGB extent；不猜测多文件帧映射"
            )
        extent = extents.iloc[0]
        pts = np.asarray(extent.video_frame_pts, dtype=np.int64)
        sensor = np.asarray(extent.video_timestamp_ns, dtype=np.int64)
        if (
            len(pts) < 2
            or len(pts) != len(sensor)
            or np.any(np.diff(pts) <= 0)
            or np.any(np.diff(sensor) <= 0)
        ):
            raise HTTPException(422, "RGB PTS / sensor 时间戳缺失、不递增或长度不一致")
        tb = Fraction(int(row["video_time_base_num"]), int(row["video_time_base_den"]))
        media = pts.astype(float) * float(tb)
        sensor_step = int(np.median(np.diff(sensor)))
        with av.open(str(row["path"])) as container:
            stream = container.streams.video[0]
            end = (
                float(
                    (stream.start_time or 0) * stream.time_base
                    + stream.duration * stream.time_base
                )
                if stream.duration
                else float(media[-1] + np.median(np.diff(media)))
            )
            first = next(container.decode(stream))
            if Fraction(first.pts) * stream.time_base != int(pts[0]) * tb:
                raise HTTPException(422, "MP4 与 WMData 首帧 PTS 不一致，请先修复映射")
        if end <= media[-1]:
            end = float(media[-1] + np.median(np.diff(media)))
        streams = pd.read_parquet(row["case_dir"] / "meta/streams.parquet")
        rgb_stream = streams[streams.stream_id == extent.stream_id].iloc[0]
        return {
            "case_id": case_id,
            "view_id": view_id,
            "file_id": row["file_id"],
            "stream_id": extent.stream_id,
            "camera_frame_id": rgb_stream.camera_frame_id,
            "pixel_domain_id": rgb_stream.pixel_domain_id,
            "relative_path": row["relative_path"],
            "video_path": str(row["path"]),
            "case_path": str(row["case_dir"]),
            "checksum": row.get("checksum"),
            "width": int(row["video_width"]),
            "height": int(row["video_height"]),
            "frame_count": len(pts),
            "time_base": [tb.numerator, tb.denominator],
            "media_pts": pts.tolist(),
            "media_seconds": media.tolist(),
            "sensor_timestamp_ns": sensor.tolist(),
            "media_end_seconds": end,
            "media_duration_seconds": end - float(media[0]),
            "sensor_duration_seconds": (int(sensor[-1]) - int(sensor[0]) + sensor_step)
            / 1e9,
            "sensor_end_timestamp_ns_estimate": int(sensor[-1]) + sensor_step,
            "interval_convention": "[start_frame, end_frame_exclusive)",
        }

    @lru_cache(maxsize=128)
    def frame(self, case_id, view_id, index):
        meta = self.metadata(case_id, view_id)
        if not 0 <= index < meta["frame_count"]:
            raise HTTPException(422, "帧号超出视频范围")
        target = meta["media_pts"][index] * Fraction(*meta["time_base"])
        with av.open(meta["video_path"]) as container:
            stream = container.streams.video[0]
            # Seek to the preceding keyframe, then decode to the exact WMData PTS.
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                actual = frame.pts * stream.time_base
                if actual == target:
                    data = io.BytesIO()
                    frame.to_image().save(data, format="JPEG", quality=90)
                    return data.getvalue()
                if actual > target:
                    break
        raise HTTPException(422, "没有解码到该帧的精确 PTS；拒绝返回相邻帧")


def validate_annotation(a: Annotation, meta):
    errors = []
    n = meta["frame_count"]
    if a.annotation_scope == "prefix_generation":
        keys = ("prefix_start", "prefix_end", "t", "eval_end")
        if set(a.marks) == {"prefix_start", "t", "eval_end"}:
            a.marks["prefix_end"] = a.marks["t"]
        if set(a.marks) != set(keys):
            raise HTTPException(422, "请提供前缀开始、前缀结束、生成开始、生成结束")
        values = [a.marks[k] for k in keys]
        if any(v is not None and not 0 <= v <= n for v in values):
            raise HTTPException(422, f"帧号超出范围 0…{n}")
        if any(v is None for v in values):
            return ["请标全 4 个边界"]
        return [] if 0 <= values[0] < values[1] <= n and 0 <= values[2] < values[3] <= n else [
            "每段需满足：开始 < 结束；两段可以不连续或重叠"
        ]
    if set(a.marks) != set(MARKS):
        raise HTTPException(
            422, "marks 必须包含 prefix_start、t、action_end、eval_start、eval_end"
        )
    for name, frame in a.marks.items():
        if frame is not None and not 0 <= frame <= n:
            raise HTTPException(422, f"{name} 超出帧范围 0…{n}")
    v = [a.marks[k] for k in MARKS]
    if any(x is None for x in v):
        errors.append("请标全 5 个时间边界")
    elif not (0 <= v[0] < v[1] < v[2] <= v[3] < v[4] <= n):
        errors.append("顺序应为：前缀开始 < t < 动作结束 ≤ 评估开始 < 评估结束")
    return list(dict.fromkeys(errors))


def boundary(meta, frame):
    eof = frame == meta["frame_count"]
    return {
        "frame": frame,
        "media_pts": None if eof else meta["media_pts"][frame],
        "media_seconds": (
            meta["media_end_seconds"] if eof else meta["media_seconds"][frame]
        ),
        "sensor_timestamp_ns": None if eof else meta["sensor_timestamp_ns"][frame],
        "sensor_timestamp_ns_estimate": (
            meta["sensor_end_timestamp_ns_estimate"] if eof else None
        ),
        "is_eof": eof,
    }


def interval(meta, start, end):
    return {
        "start": boundary(meta, start),
        "end_exclusive": boundary(meta, end),
        "frame_count": end - start,
    }


def canonical(a: Annotation, meta, errors):
    data = a.model_dump()
    if a.annotation_scope == "prefix_generation":
        data["title"] = data["title"].strip()
    # New records only need keyframes. Keep nonempty legacy annotations intact.
    for key in LEGACY_FIELDS:
        if not data[key] or (key == "evidence" and data[key] == [None, None]):
            del data[key]
    data["annotation_scope"] = a.annotation_scope
    data.update(
        schema_version="acwm.memory_episode.v1",
        updated_at=datetime.now(timezone.utc).isoformat(),
        validation_errors=errors,
        source={
            k: meta[k]
            for k in (
                "case_path",
                "video_path",
                "relative_path",
                "file_id",
                "stream_id",
                "camera_frame_id",
                "pixel_domain_id",
                "checksum",
                "width",
                "height",
                "frame_count",
                "time_base",
            )
        },
        interval_convention=meta["interval_convention"],
    )
    if "lightwheel_source" in meta:
        data["source"]["lightwheel"] = meta["lightwheel_source"]
    if a.annotation_scope == "prefix_generation":
        data["schema_version"] = "acwm.prefix_generation_episode.v2"
        data["boundaries_linked"] = (
            a.marks.get("prefix_end") == a.marks.get("t")
            and a.boundaries_linked is not False
        )
        data["hand_source"] = "source"
        if not errors:
            p, pe, t, end = [a.marks[k] for k in ("prefix_start", "prefix_end", "t", "eval_end")]
            data["segments"] = {
                "prefix": interval(meta, p, pe),
                "generation": interval(meta, t, end),
            }
            data["anchor"] = boundary(meta, pe - 1)
            indices = list(range(p, pe)) + list(range(t, end))
            count = pe - p
            data["assembly"] = {
                "policy": "concatenate_prefix_then_generation",
                "fps": 30,
                "prefix_frame_range": [0, count],
                "generation_frame_range": [count, len(indices)],
                "anchor_frame": count - 1,
                "source_frame_indices": indices,
                "source_sensor_timestamp_ns": [meta["sensor_timestamp_ns"][i] for i in indices],
                "case_timestamp_ns": [i * 1_000_000_000 // 30 for i in range(len(indices))],
                "source_intervals_overlap": max(p, t) < min(pe, end),
                "alignment": "Align hand/pose using source timestamps first; then concatenate rows in source_frame_indices order. No gap fill or interpolation across the splice.",
            }
        return data
    if not errors:
        p, t, aend, es, ee = [a.marks[k] for k in MARKS]
        data["segments"] = {
            name: interval(meta, start, end)
            for name, start, end in (
                ("prefix", p, t),
                ("action", t, aend),
                ("evaluation", es, ee),
                ("generation", t, ee),
            )
        }
        data["anchor"] = boundary(meta, t - 1)
        evidence_start, evidence_end = a.evidence
        if (
            evidence_start is not None
            and evidence_end is not None
            and p <= evidence_start < evidence_end <= t
        ):
            data["segments"]["evidence"] = interval(meta, *a.evidence)
            data["memory_delay_sensor_seconds"] = (
                meta["sensor_timestamp_ns"][t]
                - meta["sensor_timestamp_ns"][a.evidence[1]]
            ) / 1e9
        data["evaluation_policy"] = {
            "score": "gate_pass ? object_match_score : 0",
            "gate_assessed_on": "generated action + evaluation; criteria supplied by evaluator",
            "gate_pass": None,
            "object_match_score": None,
            "memory_score": None,
            "object_matching_window": "evaluation only",
            "labels_are_evaluation_only": True,
        }
    return data


def inference_manifest(row):
    """Deliberate allowlist: evaluator labels and future GT RGB are not model conditions."""
    result = {
        "schema_version": "acwm.memory_inference_request.v1",
        "annotation_id": row["id"],
        "case_id": row["case_id"],
        "view_id": row["view_id"],
        "source": row["source"],
        "interval_convention": row["interval_convention"],
        "rgb_prefix": row["segments"]["prefix"],
        "rgb_anchor": row["anchor"],
        "past_actions": row["segments"]["prefix"],
        "future_actions": row["segments"]["generation"],
        "past_head_pose": row["segments"]["prefix"],
        "future_head_pose": row["segments"]["generation"],
        "generate": row["segments"]["generation"],
        "text_condition": "",
        "contract": {
            "rgb_future_ground_truth_allowed": False,
            "action_source": "WMData hand landmarks21, render separate black-background hand stream",
            "action_alignment": "source RGB sensor_timestamp_ns, not MP4 seconds * FPS",
            "anchor_frame": "t - 1 (last observed frame); generate first unseen frame t",
            "rollout": "32 new frames per section, pad conditions if needed then trim; no future GT RGB padding",
        },
    }
    if row.get("annotation_scope") == "prefix_generation":
        result["case_name"] = row.get("title", "")
    if "assembly" in row:
        result["schema_version"] = "acwm.memory_inference_request.v2"
        result["assembly"] = row["assembly"]
        result["hand_source"] = "source"
        result["contract"]["anchor_frame"] = "prefix_end - 1 in source; prefix_frame_count - 1 in assembled case"
        result["contract"]["rgb_future_ground_truth_allowed"] = False
        result["contract"]["source_interval_overlap"] = row["assembly"]["source_intervals_overlap"]
    if "lightwheel" in row["source"]:
        result["status"] = "requires_lightwheel_condition_extraction"
        result["contract"]["action_source"] = "Original Lightwheel MCAP hand/pose topics; extraction and alignment pending"
    return result


def create_app(data_root=DEFAULT_DATA, output_root=DEFAULT_OUTPUT, annotation_mode="keyframes", *, catalog=None):
    catalog = catalog if catalog is not None else Catalog(Path(data_root))
    output_root = Path(output_root).resolve()
    records_dir = output_root / "episodes"
    records_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.RLock()
    app = FastAPI(title="ACWM Memory Annotation", docs_url="/api/docs")
    app.state.catalog = catalog
    app.state.output_root = output_root

    def records():
        with lock:
            return [
                json.loads(path.read_text()) for path in sorted(records_dir.glob("*.json"))
            ]

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "cases": len(catalog.cases),
            "output_root": str(output_root),
            "annotation_mode": annotation_mode,
            "source_kind": getattr(catalog, "source_kind", "wmdata"),
        }

    @app.get("/api/cases")
    def cases():
        saved = records()
        return [
            dict(
                case,
                completed=sum(
                    r["case_id"] == case["id"] and r["status"] == "complete"
                    for r in saved
                ),
                drafts=sum(
                    r["case_id"] == case["id"] and r["status"] == "draft" for r in saved
                ),
            )
            for case in catalog.cases.values()
        ]

    @app.get("/api/cases/{case_id}/views/{view_id}")
    def metadata(case_id: str, view_id: str):
        return catalog.metadata(case_id, view_id)

    @app.get("/api/cases/{case_id}/views/{view_id}/video")
    def video(case_id: str, view_id: str):
        return FileResponse(
            catalog.view(case_id, view_id)["path"], media_type="video/mp4"
        )

    @app.get("/api/cases/{case_id}/views/{view_id}/frames/{frame}.jpg")
    def frame_image(case_id: str, view_id: str, frame: int):
        data = catalog.frame(case_id, view_id, frame)
        meta = catalog.metadata(case_id, view_id)
        return Response(
            data,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "private, max-age=86400",
                "X-Frame-Index": str(frame),
                "X-Media-PTS": str(meta["media_pts"][frame]),
                "X-Sensor-Timestamp-Ns": str(meta["sensor_timestamp_ns"][frame]),
            },
        )

    @app.get("/api/annotations")
    def annotations(case_id: str | None = None):
        return [r for r in records() if case_id is None or r["case_id"] == case_id]

    @app.post("/api/annotations")
    def save(a: Annotation):
        if a.annotation_scope != annotation_mode:
            raise HTTPException(422, "标注模式与当前工作台不一致，请刷新页面")
        meta = catalog.metadata(a.case_id, a.view_id)
        errors = validate_annotation(a, meta)
        if a.status == "complete" and errors:
            raise HTTPException(422, errors)
        path = records_dir / f"{a.id}.json"
        with lock:
            if (output_root / "deleted" / f"{a.id}.json").exists():
                raise HTTPException(409, "此样本已删除，不能从旧页面重新保存；请新建样本")
            previous = json.loads(path.read_text()) if path.exists() else None
            if previous and (previous["case_id"], previous["view_id"]) != (
                a.case_id,
                a.view_id,
            ):
                raise HTTPException(409, "标注 ID 已属于另一个视频")
            if a.revision != (previous["revision"] if previous else 0):
                raise HTTPException(
                    409, "此标注已在其他页面更新；请保留当前文字并重新加载，避免覆盖"
                )
            if previous:
                # A keyframe-only client must not erase older descriptive labels.
                inherited = {
                    key: previous[key]
                    for key in LEGACY_FIELDS
                    if key in previous and key not in a.model_fields_set
                }
                a = Annotation.model_validate(a.model_dump() | inherited)
            a.revision += 1
            data = canonical(a, meta, errors)
            if previous:
                has_auxiliary = any(
                    previous.get(key)
                    for key in LEGACY_FIELDS
                    if key not in ("title", "evidence")
                ) or any(frame is not None for frame in previous.get("evidence", []))
                if previous.get("auxiliary_annotations_need_review") or (
                    has_auxiliary and previous["marks"] != a.marks
                ):
                    data["auxiliary_annotations_need_review"] = True
                atomic_json(
                    output_root / "history" / a.id / f"{previous['revision']:06d}.json",
                    previous,
                )
            atomic_json(path, data)
        return data

    @app.get("/api/deleted-annotations")
    def deleted_annotations():
        with lock:
            return [p.stem for p in sorted((output_root / "deleted").glob("*.json"))]

    @app.delete("/api/annotations/{annotation_id}")
    def delete_annotation(annotation_id: str, revision: int):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", annotation_id):
            raise HTTPException(422, "无效样本 ID")
        with lock:
            path = records_dir / f"{annotation_id}.json"
            if not path.exists():
                raise HTTPException(404, "样本不存在或已删除")
            previous = json.loads(path.read_text())
            if previous["revision"] != revision:
                raise HTTPException(409, "样本已更新，请重新加载后再删除")
            trash = output_root / "deleted"
            trash.mkdir(exist_ok=True)
            os.replace(path, trash / path.name)
            # Refresh cached lists too, so deleted samples cannot enter later builds.
            export("inference")
            export("evaluation")
        return {"deleted": annotation_id, "recoverable": True}

    @app.get("/api/export/{kind}")
    def export(kind: str):
        if kind not in ("evaluation", "inference"):
            raise HTTPException(404, "导出类型不存在")
        with lock:
            rows = [row for row in records() if row["status"] == "complete"]
            if kind == "inference":
                rows = [inference_manifest(row) for row in rows]
            text = "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                for row in rows
            )
            exports = output_root / "exports"
            exports.mkdir(exist_ok=True)
            path = exports / f"{kind}.jsonl"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        return Response(
            text,
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f'attachment; filename="memory_{kind}.jsonl"',
                "X-Episode-Count": str(len(rows)),
            },
        )

    @app.get("/")
    def index():
        return FileResponse(HERE / "static" / (
            "ranges.html" if annotation_mode == "prefix_generation" else "index.html"
        ))

    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--annotation-mode", choices=("keyframes", "prefix_generation"), default="keyframes")
    args = parser.parse_args()
    import uvicorn

    # Idle MP4 range requests must not hold a service restart open indefinitely.
    uvicorn.run(
        create_app(args.data_root, args.output_root, args.annotation_mode),
        host=args.host,
        port=args.port,
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
