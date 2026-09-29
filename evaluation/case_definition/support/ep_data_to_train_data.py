#!/usr/bin/env python3
"""Convert one WMData package into a complete MemWorld sample_pt dataset."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from fractions import Fraction
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

PROJECT_ROOT = Path(os.environ.get(
    "MEMWORLD_ROOT", str(Path(__file__).resolve().parent.parent / "MemWorld")
)).expanduser().resolve()
if not (PROJECT_ROOT / "dataset/sample_dataset.py").is_file():
    raise ImportError(f"MemWorld checkout not found at {PROJECT_ROOT}; set MEMWORLD_ROOT")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
import torch
from dataset.poses import convert_c2w_opencv_to_memworld
from dataset.sample_dataset import SamplePTDataset, atomic_torch_save
from utils.memory_context import (
    anchor_layout_for_clip, target_frame_indices_for_clip,
    short_context_frame_indices_for_clip, long_context_frame_indices_for_clip,
    relative_poses_for_frame_indices, NEG_INF_FRAME, is_padding_frame,
)
from utils.overlap_utils import calculate_overlap_matrix_from_c2w

def load_standard(root: Path):
    core = root / "src"
    review = root / "integrations/reviewers/hand_comparison/src"
    if not (core / "wmdata_standard").is_dir() or not (review / "wmdata_hand_review").is_dir():
        raise ValueError(f"Expected evophys-wmdata checkout: {root}")
    for path in (core, review):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from wmdata_standard.validator import validate_dataset
    from wmdata_hand_review import geometry
    return validate_dataset, geometry


class CaseReader:
    def __init__(self, root: Path, standard_root: Path):
        import json
        self.root = root.resolve()
        validator, self.geometry = load_standard(standard_root)
        self.validation = validator(self.root, deep=False).to_dict()
        if not self.validation["ok"]:
            raise ValueError(f"WMData metadata validation failed: {self.validation['errors']}")
        self.dataset = json.loads((self.root / "dataset.json").read_text())
        self.tables = {name: pq.read_table(self.root / "meta" / f"{name}.parquet").to_pylist()
                       for name in ("trajectories", "streams", "files", "extents", "provenance")}
        self.files = {r["file_id"]: r for r in self.tables["files"]}
        self.provenance = {r["provenance_id"]: r for r in self.tables["provenance"]}
 
    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Path escapes case: {relative}")
        return path

    def extents(self, stream):
        return sorted((r for r in self.tables["extents"] if r["stream_id"] == stream["stream_id"]),
                      key=lambda r: r["logical_sample_start"])

    def rows(self, stream):
        rows = []
        for extent in self.extents(stream):
            f = self.files[extent["file_id"]]
            if f["storage_type"] != "parquet" or extent["row_group_id"] is None:
                raise ValueError("Expected aligned Parquet row group")
            pf = pq.ParquetFile(self.path(f["relative_path"]))
            piece = pf.read_row_group(int(extent["row_group_id"])).to_pylist()
            if len(piece) != extent["logical_sample_count"]:
                raise ValueError("Extent does not own the complete row group")
            if extent["logical_sample_start"] != len(rows):
                raise ValueError("Non-contiguous logical coverage")
            rows.extend(piece)
        timestamps = [int(r["timestamp_ns"]) for r in rows]
        if len(rows) != stream["sample_count"] or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError(f"Invalid count/timeline for {stream['stream_id']}")
        return rows

    def rgb_rows(self, stream):
        rows = []
        for e in self.extents(stream):
            f = self.files[e["file_id"]]
            if f["content_type"] != "rgb_video":
                raise ValueError("RGB must use rgb_video storage")
            if e["logical_sample_start"] != len(rows):
                raise ValueError("Non-contiguous RGB extents")
            if len(e["video_frame_pts"]) != e["logical_sample_count"] or len(e["video_timestamp_ns"]) != e["logical_sample_count"]:
                raise ValueError("Incomplete RGB PTS/capture mapping")
            for pts, t in zip(e["video_frame_pts"], e["video_timestamp_ns"]):
                rows.append({"frame_index": len(rows), "timestamp_ns": int(t), "pts": int(pts), "file_id": e["file_id"]})
        if len(rows) != stream["sample_count"] or any(b["timestamp_ns"] <= a["timestamp_ns"] for a,b in zip(rows,rows[1:])):
            raise ValueError("Invalid RGB timeline")
        return rows

    def frames(self, stream, stop_index=None):
        """Sequential, bounded-memory decoding; translate decoder PTS into declared time base."""
        for extent in self.extents(stream):
            start = int(extent["logical_sample_start"])
            if stop_index is not None and start >= stop_index:
                return
            f = self.files[extent["file_id"]]
            declared_base = Fraction(int(f["video_time_base_num"]), int(f["video_time_base_den"]))
            pts_values = extent["video_frame_pts"]
            wanted_count = len(pts_values) if stop_index is None else min(len(pts_values), stop_index-start)
            cursor = 0
            with av.open(str(self.path(f["relative_path"]))) as container:
                videos = list(container.streams.video)
                if len(videos) != 1:
                    raise ValueError("Expected exactly one video track per file")
                video = videos[0]
                video.thread_type = "AUTO"
                # Seek to a keyframe at or before the first mapped frame.
                container.seek(int(Fraction(pts_values[0]) * declared_base / video.time_base), stream=video, backward=True)
                for frame in container.decode(video):
                    if frame.pts is None:
                        raise ValueError("Decoded frame has no PTS")
                    actual = Fraction(frame.pts) * frame.time_base / declared_base
                    wanted = pts_values[cursor]
                    if actual < wanted:
                        continue
                    if actual > wanted:
                        raise ValueError(f"Missing registered RGB PTS {wanted}; got {actual}")
                    if (frame.width, frame.height) != (f["video_width"], f["video_height"]):
                        raise ValueError("Decoded size does not match files metadata")
                    yield start+cursor, frame.to_image()
                    cursor += 1
                    if cursor == wanted_count:
                        break
            if cursor != wanted_count:
                raise ValueError("Video ended before its registered samples")

    @staticmethod
    def unique(candidates, description):
        if len(candidates) != 1:
            raise ValueError(f"Expected one {description}; candidates={[r['stream_id'] for r in candidates]}")
        return candidates[0]

    def bundles(self, rgb_stream_id=None, hand_source="auto", left_stream_id=None, right_stream_id=None):
        rgbs = [s for s in self.tables["streams"] if s["schema_id"] == "rgb/frames@v1"]
        if rgb_stream_id:
            rgbs = [s for s in rgbs if s["stream_id"] == rgb_stream_id]
        if not rgbs:
            raise ValueError("No selected RGB stream")
        for rgb in rgbs:
            streams = [s for s in self.tables["streams"] if s["trajectory_index"] == rgb["trajectory_index"]]
            pose = self.unique([s for s in streams if s["schema_id"] == "camera_pose/pose9@v1" and s["camera_frame_id"] == rgb["camera_frame_id"]], "bound camera pose")
            calibration = self.unique([s for s in streams if s["modality"] == "camera_calibration" and s["camera_frame_id"] == rgb["camera_frame_id"] and s["pixel_domain_id"] == rgb["pixel_domain_id"]], "bound calibration")
            if calibration["schema_id"] != "camera_calibration/pinhole@v1":
                raise ValueError("This adapter supports pinhole calibration only; do not silently reinterpret distortion models")
            hands = {}
            if hand_source != "none":
                candidates = [s for s in streams if s["schema_id"] == "hand/landmarks21@v1"]
                source = [s for s in candidates if self.provenance[s["provenance_id"]]["origin"] in ("native", "exact_conversion")]
                methods = {"hand/fast-garland@v1", "evophys/fastgarland_wilor_temporal_landmarks21@v1"}
                derived = [s for s in candidates if self.provenance[s["provenance_id"]].get("method_id") in methods]
                # This preference is a documented adapter policy, never inferred from paths.
                chosen = source if hand_source == "source" or (hand_source == "auto" and source) else derived
                for side, explicit in (("left", left_stream_id), ("right", right_stream_id)):
                    matches = [s for s in (candidates if explicit else chosen) if s["side"] == side and (not explicit or s["stream_id"] == explicit)]
                    if matches:
                        hands[side] = self.unique(matches, f"{side} hand; pass --{side}-stream-id to disambiguate")
                    elif explicit or hand_source in ("source", "fast-garland"):
                        raise ValueError(f"No selected {side} hand")
                if not hands and hand_source == "auto" and candidates:
                    raise ValueError("Unknown hand provenance; choose stream IDs explicitly or hand_source=none")
            yield {"rgb": rgb, "pose": pose, "calibration": calibration, "hands": hands}


class HandTimeline:
    def __init__(self, rows, max_gap_ns, max_nearest_ns):
        self.rows = rows
        self.times = [int(r["timestamp_ns"]) for r in rows]
        self.max_gap_ns = max_gap_ns
        self.max_nearest_ns = max_nearest_ns

    @staticmethod
    def value(row):
        if row.get("status") != "valid" or row.get("positions_ref") is None:
            return None
        p = np.asarray(row["positions_ref"], dtype=np.float64)
        if p.shape != (21,3) or not np.isfinite(p).all():
            raise ValueError("Malformed valid hand observation")
        return p

    def at(self, t):
        i = bisect_left(self.times, t)
        if i < len(self.times) and self.times[i] == t:
            return self.value(self.rows[i]), "exact"
        if 0 < i < len(self.times):
            a,b = self.value(self.rows[i-1]), self.value(self.rows[i])
            gap = self.times[i]-self.times[i-1]
            if a is not None and b is not None and gap <= self.max_gap_ns:
                return a+(b-a)*((t-self.times[i-1])/gap), "interpolated"
            return None, "unmatched"  # never bridge a failed observation
        j = 0 if i == 0 else len(self.times)-1
        if self.times and abs(self.times[j]-t) <= self.max_nearest_ns:
            return self.value(self.rows[j]), "boundary_nearest"
        return None, "unmatched"


def valid_runs(valid, times, max_gap_ns):
    start = None
    for i, ok in enumerate(valid):
        if start is not None and (not ok or (i and times[i]-times[i-1] > max_gap_ns)):
            yield start, i
            start = None
        if ok and start is None:
            start = i
    if start is not None:
        yield start, len(valid)


def resize_image(image, height, width):
    # Match MemWorld aspect-fill + keep_bottom geometry, shared by RGB and skeleton.
    w,h = image.size
    scale = max(width/w,height/h)
    resized = image.resize((max(width,round(w*scale)),max(height,round(h*scale))), Image.Resampling.BICUBIC)
    x = (resized.width-width)//2
    return resized.crop((x,resized.height-height,x+width,resized.height))


def render_hands(size, camera_pose, calibration, points_by_side, geometry):
    image = Image.new("RGB",size,(0,0,0))
    draw = ImageDraw.Draw(image)
    colors = [(0,255,255),(0,80,255),(255,255,0),(255,0,255),(80,255,80)]
    w2c = np.linalg.inv(camera_pose)
    for side,points in points_by_side.items():
        if points is None:
            continue
        camera = points @ w2c[:3,:3].T + w2c[:3,3]
        pixels,depth = geometry.project_camera_points(camera,calibration)
        valid = np.isfinite(pixels).all(axis=1) & (depth>0) & (pixels[:,0]>=0) & (pixels[:,0]<size[0]) & (pixels[:,1]>=0) & (pixels[:,1]<size[1])
        for finger in range(5):
            chain = [0]+list(range(1+4*finger,5+4*finger))
            for a,b in zip(chain,chain[1:]):
                if valid[a] and valid[b]:
                    coords = tuple(pixels[a])+tuple(pixels[b])
                    draw.line(coords,fill=(0,0,0),width=7)
                    draw.line(coords,fill=colors[finger],width=3)
        color = (0,255,80) if side == "left" else (0,170,255)
        for xy,ok in zip(pixels,valid):
            if ok:
                x,y = xy
                draw.ellipse((x-5,y-5,x+5,y+5),fill=(0,0,0))
                draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
    return image


DEFAULT_MODEL = "/aifs/xuyijie/memworld/models/Wan2.2-TI2V-5B"


@dataclass(frozen=True)
class ConversionOptions:
    model_dir: str = os.environ.get("MEMWORLD_MODEL_DIR_DEFAULT", DEFAULT_MODEL)
    standard_root: str = str(PROJECT_ROOT.parent / "evophys-wmdata")
    device: str = "cuda:0"
    height: int = 352
    width: int = 640
    prompt: str = "A first-person view of a person interacting with the environment."
    rgb_stream_id: str | None = None  # None exports all views as independent scenes
    hand_source: str = "auto"  # source first, known fast-garland second; absence allowed
    left_stream_id: str | None = None
    right_stream_id: str | None = None
    max_pose_delta_ms: float = 100.0
    max_hand_gap_ms: float = 100.0
    max_hand_nearest_ms: float = 20.0
    max_rgb_gap_ms: float = 200.0
    max_clips: int | None = None  # global limit; None converts the complete package
    overlap_samples: int = 1024
    overlap_radius: float = 50.0
    seed: int = 42
    single_batch_size: int = 4
    tiled: bool = True

    def validate(self):
        if self.height <= 0 or self.width <= 0 or self.height % 32 or self.width % 32:
            raise ValueError("height/width must be positive multiples of 32")
        if self.hand_source not in ("auto","source","fast-garland","none"):
            raise ValueError("Invalid hand_source")
        if self.hand_source == "none" and (self.left_stream_id or self.right_stream_id):
            raise ValueError("hand_source=none conflicts with explicit hand stream IDs")
        if self.max_clips is not None and self.max_clips <= 0:
            raise ValueError("max_clips must be positive")
        if min(self.overlap_samples,self.single_batch_size) <= 0 or self.overlap_radius <= 0:
            raise ValueError("Batch size, overlap samples/radius must be positive")
        if any(not np.isfinite(v) or v <= 0 for v in (self.max_pose_delta_ms,self.max_hand_gap_ms,self.max_hand_nearest_ms,self.max_rgb_gap_ms)):
            raise ValueError("Alignment limits must be finite and positive")


class WanEncoder:
    """Load actual VAE and text encoder, never the DiT; GPU residency is sequential."""
    def __init__(self, options):
        from utils.model_loading import build_memory_pipeline
        self.options = options
        model_dir = Path(options.model_dir).resolve()
        for name in ("Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl"):
            if not (model_dir/name).exists():
                raise FileNotFoundError(f"Missing pretrained asset: {model_dir/name}; supply --model-dir")
        if options.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; select a working CUDA environment or --device cpu")
        self.pipe = build_memory_pipeline(model_dir=model_dir, include_dit=False,
            include_text_encoder=True, include_vae=True, device="cpu", torch_dtype=torch.bfloat16)
        self.pipe.device = options.device
        self.pipe.text_encoder.to(options.device).eval()
        with torch.inference_mode():
            self.prompt_emb = {k:v.detach().cpu().contiguous() for k,v in self.pipe.encode_prompt(options.prompt).items()}
        self.pipe.text_encoder.to("cpu")
        if options.device.startswith("cuda"):
            torch.cuda.empty_cache()
        self.pipe.vae.to(options.device).eval()

    def encode(self, videos):
        # videos: iterable of 1- or 33-frame sequences of identically resized PIL RGB images
        tensors = []
        for frames in videos:
            array = np.stack([np.array(im,dtype=np.uint8,copy=True) for im in frames])
            tensor = torch.from_numpy(array).permute(3,0,1,2).float().div_(127.5).sub_(1)
            tensors.append(tensor)
        batch = torch.stack(tensors).to(device=self.options.device,dtype=torch.bfloat16)
        with torch.inference_mode():
            result = self.pipe.encode_video(batch, tiled=self.options.tiled).detach().cpu().contiguous()
        expected = (len(videos),48,(len(videos[0])-1)//4+1,self.options.height//16,self.options.width//16)
        if tuple(result.shape) != expected or not torch.isfinite(result).all():
            raise ValueError(f"Unexpected VAE result {tuple(result.shape)}; expected {expected}")
        return result


def write_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,path)


def fingerprint(reader, options):
    digest = hashlib.sha256(json.dumps(asdict(options),sort_keys=True).encode())
    files = [reader.root/"dataset.json"] + sorted((reader.root/"meta").glob("*.parquet"))
    # Metadata contains declared payload checksums. Conversion is not a deep checksum audit.
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda:handle.read(1024*1024),b""):
                digest.update(block)
    return digest.hexdigest()


def prepare_view(reader,bundle,options):
    rgb_rows = reader.rgb_rows(bundle["rgb"])
    pose_rows = reader.rows(bundle["pose"])
    for row in pose_rows:
        if row["status"] not in ("valid","fail"):
            raise ValueError("Invalid pose status")
        if row["status"] == "valid":
            p = np.asarray(row["pose9"],dtype=float)
            if p.shape != (9,) or not np.isfinite(p).all():
                raise ValueError("Malformed valid pose")
            a,b = p[3:6],p[6:9]
            if max(abs(np.linalg.norm(a)-1),abs(np.linalg.norm(b)-1),abs(a@b)) > 1e-4:
                raise ValueError("Invalid Rotation6D; refusing to normalize a malformed pose")
        elif row.get("pose9") is not None:
            raise ValueError("Failed pose must have null payload")
    geo = reader.geometry
    pose_index = geo.PoseIndex.from_rows(pose_rows)
    calibrations = geo.parse_calibrations(bundle["calibration"]["schema_id"],reader.rows(bundle["calibration"]))
    cal_times = [c.timestamp_ns for c in calibrations]
    optical,memworld,cals,valid = [],[],[],[]
    for rgb in rgb_rows:
        t = rgb["timestamp_ns"]
        pose = geo.resolve_pose(pose_index,t,max_delta_ns=round(options.max_pose_delta_ms*1e6))
        j = bisect_right(cal_times,t)-1
        cal = calibrations[j] if j>=0 else None
        ok = pose is not None and cal is not None and cal.status == "valid"
        if ok:
            f = reader.files[rgb["file_id"]]
            ok = (cal.width,cal.height)==(f["video_width"],f["video_height"])
            ok = ok and all(k in cal.parameters and np.isfinite(cal.parameters[k]) for k in ("fx","fy","cx","cy"))
            ok = ok and cal.parameters["fx"]>0 and cal.parameters["fy"]>0
        optical.append(pose)
        memworld.append(convert_c2w_opencv_to_memworld(pose) if pose is not None else np.eye(4))
        cals.append(cal)
        valid.append(bool(ok))
    hands = {side:HandTimeline(reader.rows(stream),round(options.max_hand_gap_ms*1e6),round(options.max_hand_nearest_ms*1e6)) for side,stream in bundle["hands"].items()}
    times = [r["timestamp_ns"] for r in rgb_rows]
    runs = list(valid_runs(valid,times,round(options.max_rgb_gap_ms*1e6)))
    return {"rgb_rows":rgb_rows,"optical":optical,"c2ws":np.stack(memworld),"calibrations":cals,"hands":hands,"runs":runs,"invalid_frames":len(valid)-sum(valid)}


def choose_dynamic(clip_idx,c2ws,cache,options):
    start = clip_idx*32
    anchor,_ = anchor_layout_for_clip(clip_idx)
    candidates = sorted(i for i in cache if i < anchor)
    if not candidates:
        return [NEG_INF_FRAME]*8
    queries = target_frame_indices_for_clip(clip_idx)[1:]
    scores = calculate_overlap_matrix_from_c2w(c2ws[queries],c2ws[candidates],
        num_samples=options.overlap_samples,radius=options.overlap_radius,
        device="cpu",target_frame_indices=queries,overlap_seed=options.seed)
    selected,used = [],set()
    for row in scores:
        ranked = np.argsort(-row,kind="stable")
        candidate = next((candidates[j] for j in ranked if row[j]>0 and candidates[j] not in used),None)
        if candidate is None:
            selected.append(NEG_INF_FRAME)
        else:
            selected.append(candidate);used.add(candidate)
    return selected


def build_sample(scene,clip_idx,c2ws,rgb_latent,kp_latent,previous,cache,encoder,options):
    anchor,anchor_frames = anchor_layout_for_clip(clip_idx)
    target_frames = target_frame_indices_for_clip(clip_idx)
    if previous is None:
        target = rgb_latent
        keypoint = kp_latent
    else:
        target = torch.cat([previous[0][:,-1:],rgb_latent[:,1:]],dim=1)
        keypoint = None if kp_latent is None else torch.cat([previous[1][:,-1:],kp_latent[:,1:]],dim=1)
    sample = {"sample_id":f"{scene}_clip{clip_idx:04d}","video_name":scene,"clip_idx":clip_idx,
        "first_frame_indices":[clip_idx*32],"anchor_frame_indices":anchor_frames,
        "target_frame_indices":target_frames,"target_latents":target,
        "target_poses":relative_poses_for_frame_indices(c2ws,anchor,target_frames),
        "target_keypoint_latents":keypoint,"prompt":options.prompt,"prompt_emb":encoder.prompt_emb}
    pools = {"long":long_context_frame_indices_for_clip(clip_idx),
             "short":short_context_frame_indices_for_clip(clip_idx),
             "dynamic":choose_dynamic(clip_idx,c2ws,cache,options)}
    for pool,indices in pools.items():
        zero = torch.zeros_like(target[:,:1])
        sample[f"{pool}_context_frame_indices"] = indices
        sample[f"{pool}_context_latents"] = torch.cat([zero if is_padding_frame(i) else cache[int(i)][0] for i in indices],dim=1)
        sample[f"{pool}_context_poses"] = relative_poses_for_frame_indices(c2ws,anchor,indices)
        sample[f"{pool}_keypoint_latents"] = None if keypoint is None else torch.cat([zero if is_padding_frame(i) else cache[int(i)][1] for i in indices],dim=1)
    return sample


def convert_case(case_dir, output_dir, *, options: ConversionOptions | None = None, _encoder=None):
    """Convert a whole case. Return output Path; only publish index.json after validation.

    _encoder is an internal test seam, not a CLI option. Normal runs always use Wan weights.
    Existing nonempty outputs are rejected to avoid mixing datasets/configurations.
    """
    options = options or ConversionOptions()
    options.validate()
    case = Path(case_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output == case or output.is_relative_to(case) or case.is_relative_to(output):
        raise ValueError("Input and output directories must be disjoint")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    reader = CaseReader(case,Path(options.standard_root))
    bundles = list(reader.bundles(options.rgb_stream_id,options.hand_source,options.left_stream_id,options.right_stream_id))
    prepared = [(b,prepare_view(reader,b,options)) for b in bundles]
    possible = sum(max(0,(end-start-1)//32) for _,v in prepared for start,end in v["runs"])
    if not possible:
        raise ValueError("No continuous valid RGB/pose/calibration interval with at least 33 frames")
    output.mkdir(parents=True,exist_ok=True)
    report = {"status":"running","input":str(case),"dataset":reader.dataset,
        "input_and_config_digest":fingerprint(reader,options),"options":asdict(options),
        "metadata_validation":reader.validation,"possible_samples":possible,"views":[],
        "conversion_contract":"wmdata_to_memworld_native_rgb_v1",
        "memory_policy":"long16_short8_dynamic8; dynamic from last 17 clips of every-4-frame singles; fixed FOV 45/30 deg, seeded MemWorld overlap",
        "time_policy":"preserve RGB capture order/timestamps; no CFR resampling; pose interpolation bounded; split invalid geometry/large RGB gaps",
        "hand_failure_policy":"black skeleton for missing/failed hand; masks in provenance only; not a loss mask",
        "quality_policy":"quality streams not consumed; no dataset-preparation filtering",
        "test_encoder":_encoder is not None}
    write_json(output/"conversion_report.json",report)
    entries = []
    began = time.monotonic()
    try:
        encoder = _encoder or WanEncoder(options)
        for view_id,(bundle,view) in enumerate(prepared):
            if options.max_clips is not None and len(entries)>=options.max_clips:
                break
            view_report = {"rgb_stream_id":bundle["rgb"]["stream_id"],"pose_stream_id":bundle["pose"]["stream_id"],
                "calibration_stream_id":bundle["calibration"]["stream_id"],"hand_stream_ids":{k:v["stream_id"] for k,v in bundle["hands"].items()},
                "invalid_geometry_frames":view["invalid_frames"],"runs":[],"hand_alignment_counts":{},"emitted_samples":0}
            report["views"].append(view_report)
            runs = [(a,b) for a,b in view["runs"] if b-a>=33]
            if not runs:
                continue
            stop = runs[-1][0]+((runs[-1][1]-runs[-1][0]-1)//32)*32+1
            if options.max_clips is not None:
                remaining = options.max_clips-len(entries)
                for a,b in runs:
                    count = (b-a-1)//32
                    if remaining<=count:
                        stop = a+remaining*32+1;break
                    remaining-=count
            frame_iter = iter(reader.frames(bundle["rgb"],stop_index=stop))
            cursor_frame = None
            align_counts = Counter()
            for run_id,(start,end) in enumerate(runs):
                if options.max_clips is not None and len(entries)>=options.max_clips:
                    break
                count = (end-start-1)//32
                if options.max_clips is not None:
                    count = min(count,options.max_clips-len(entries))
                scene = re.sub(r"[^A-Za-z0-9_.-]","_",case.name)+f"_view{view_id:02d}_segment{run_id:03d}"
                c2ws = view["c2ws"][start:end]
                sample_dir = output/"samples"/scene
                sample_dir.mkdir(parents=True,exist_ok=True)
                (output/"provenance").mkdir(exist_ok=True)
                write_json(output/"provenance"/f"{scene}.json",{
                    "rgb_stream_id":bundle["rgb"]["stream_id"],"source_frame_indices":list(range(start,start+count*32+1)),
                    "timestamp_ns":[r["timestamp_ns"] for r in view["rgb_rows"][start:start+count*32+1]],
                    "c2ws":c2ws[:count*32+1].tolist()})
                run_report = {"scene":scene,"source_start":start,"source_end_exclusive":end,
                    "emitted_frames":count*32+1,"unused_tail_frames":end-start-(count*32+1)}
                view_report["runs"].append(run_report)
                cache,previous = {},None
                rgb_buffer,kp_buffer,mask_buffer = [],[],[]
                for local in range(count*32+1):
                    absolute = start+local
                    while cursor_frame is None or cursor_frame[0]<absolute:
                        cursor_frame = next(frame_iter)
                    if cursor_frame[0]!=absolute:
                        raise ValueError("Decoder skipped a requested canonical frame")
                    image = cursor_frame[1]
                    rgb_buffer.append(resize_image(image,options.height,options.width))
                    t = view["rgb_rows"][absolute]["timestamp_ns"]
                    points,mask = {},{}
                    for side,timeline in view["hands"].items():
                        value,method = timeline.at(t)
                        points[side]=value;mask[side]=value is not None
                        align_counts[f"{side}:{method}:{'valid' if value is not None else 'missing'}"]+=1
                    mask_buffer.append(mask)
                    if view["hands"]:
                        skeleton = render_hands(image.size,view["optical"][absolute],view["calibrations"][absolute],points,reader.geometry)
                        kp_buffer.append(resize_image(skeleton,options.height,options.width))
                    if len(rgb_buffer)<33:
                        continue
                    clip_idx=(local-32)//32
                    # Encode target windows sequentially to bound peak GPU memory.
                    rgb_latent=encoder.encode([rgb_buffer])[0]
                    kp_latent=encoder.encode([kp_buffer])[0] if kp_buffer else None
                    sample=build_sample(scene,clip_idx,c2ws,rgb_latent,kp_latent,previous,cache,encoder,options)
                    sample["wmdata_source"]={"dataset_id":reader.dataset["dataset_id"],"dataset_version":reader.dataset["dataset_version"],
                        "rgb_stream_id":bundle["rgb"]["stream_id"],"source_frame_start":start+clip_idx*32,
                        "frame_timestamp_ns":[r["timestamp_ns"] for r in view["rgb_rows"][start+clip_idx*32:absolute+1]],
                        "hand_valid_by_rgb_frame":mask_buffer}
                    path=sample_dir/f"{sample['sample_id']}.pt"
                    atomic_torch_save(sample,str(path))
                    entries.append({"sample_id":sample["sample_id"],"video_name":scene,"clip_idx":clip_idx,"path":path.relative_to(output).as_posix()})
                    view_report["emitted_samples"]+=1
                    print(f"Converted {len(entries)}/{min(possible,options.max_clips or possible)}: {sample['sample_id']}",flush=True)
                    # Single-frame latents needed by future memory. Keep only the bounded history.
                    if clip_idx+1<count:
                        slots=list(range(0,32,4))
                        for k in range(0,len(slots),options.single_batch_size):
                            batch_slots=slots[k:k+options.single_batch_size]
                            rgbs=encoder.encode([[rgb_buffer[i]] for i in batch_slots])
                            kps=encoder.encode([[kp_buffer[i]] for i in batch_slots]) if kp_buffer else [None]*len(batch_slots)
                            for i,rgb_one,kp_one in zip(batch_slots,rgbs,kps):
                                cache[clip_idx*32+i]=(rgb_one,kp_one)
                        earliest=max(0,(clip_idx+1-17)*32)
                        cache={i:v for i,v in cache.items() if i>=earliest}
                    previous=(rgb_latent,kp_latent)
                    rgb_buffer=rgb_buffer[-1:];kp_buffer=kp_buffer[-1:];mask_buffer=mask_buffer[-1:]
                view_report["hand_alignment_counts"]=dict(align_counts)
            frame_iter.close()
        if not entries:
            raise ValueError("Conversion produced no samples")
        # Validate through the actual Dataset before making the directory discoverable by training.
        index={"format":"sample_pt_v1","sample_count":len(entries),"samples":entries}
        staging=output/".validation"
        staging.mkdir()
        (staging/"samples").symlink_to(output/"samples",target_is_directory=True)
        write_json(staging/"index.json",index)
        dataset=SamplePTDataset(str(staging),drop_context_prob=0)
        for i in range(len(dataset)):
            item=dataset[i]
            context = item["prompt_emb"]["context"]
            if tuple(context.shape) != (1,512,4096) or not torch.isfinite(context).all():
                raise ValueError("Invalid Wan text embedding")
            if tuple(item["target_pose"].shape)!=(9,12) or tuple(item["context_pose"].shape)!=(32,12):
                raise ValueError("Dataset returned incompatible pose shapes")
            if item["context_latents"].shape[1]!=32 or item["target_latents"].shape[1]!=9:
                raise ValueError("Dataset returned incompatible temporal shapes")
            for key in ("target_latents","context_latents","target_pose","context_pose","target_keypoint_latents","context_keypoint_latents"):
                if not torch.isfinite(item[key]).all():
                    raise ValueError(f"Nonfinite training field: {key}")
        (staging/"samples").unlink();(staging/"index.json").unlink();staging.rmdir()
        report.update(status="complete",sample_count=len(entries),elapsed_seconds=time.monotonic()-began,
            limited=options.max_clips is not None and len(entries)<possible,
            validation="all exported samples loaded using SamplePTDataset(drop_context_prob=0)")
        write_json(output/"conversion_report.json",report)
        write_json(output/"index.json",index)  # commit marker, published last
    except Exception as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}",sample_count=len(entries))
        write_json(output/"conversion_report.json",report)
        raise
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    defaults=ConversionOptions()
    for name in ("model_dir","standard_root","device","prompt","rgb_stream_id","left_stream_id","right_stream_id"):
        parser.add_argument("--"+name.replace("_","-"),default=getattr(defaults,name))
    parser.add_argument("--hand-source",choices=["auto","source","fast-garland","none"],default="auto")
    for name in ("height","width","max_clips","overlap_samples","seed","single_batch_size"):
        parser.add_argument("--"+name.replace("_","-"),type=int,default=getattr(defaults,name))
    for name in ("max_pose_delta_ms","max_hand_gap_ms","max_hand_nearest_ms","max_rgb_gap_ms","overlap_radius"):
        parser.add_argument("--"+name.replace("_","-"),type=float,default=getattr(defaults,name))
    parser.add_argument("--tiled",action=argparse.BooleanOptionalAction,default=True)
    args=vars(parser.parse_args())
    case=args.pop("case_dir");output=args.pop("output_dir")
    result=convert_case(case,output,options=ConversionOptions(**args))
    print(f"Training-ready dataset: {result}")


if __name__ == "__main__":
    main()
