"""Eight real frames with timestamps, cached without model identities."""

import fcntl
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .manifest import digest

SHEET_VERSION = "eight-frames-v1"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def duration(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            value = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            value = container.duration / av.time_base
        else:
            raise ValueError(f"Missing video duration: {path}")
        if value <= 0:
            raise ValueError(f"Invalid video duration: {path}")
        return value


def sample_frames(clip, timeline):
    """Uniformly sample each available future, using actual presentation times.

    Short predictions keep their true times. Frames beyond the GT horizon are
    excluded; no invented/padded frames are used to extend short videos.
    """
    start = clip.start_seconds
    end = min(clip.duration_seconds, start + timeline)
    with av.open(clip.path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate or stream.guessed_rate or 25)
        targets = np.linspace(start, max(start, end - 1 / fps), 8).tolist()
        origin = float((stream.start_time or 0) * stream.time_base)
        previous = None
        chosen = []
        target_index = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Frames require presentation timestamps")
            stamp = float(frame.pts * stream.time_base) - origin
            if stamp < start - 1e-6:
                continue
            if stamp >= end - 1e-6:
                break
            while target_index < len(targets) and stamp >= targets[target_index]:
                candidate = (stamp, frame)
                if previous and abs(previous[0] - targets[target_index]) <= abs(stamp - targets[target_index]):
                    candidate = previous
                chosen.append((candidate[0] - start, candidate[1].to_image()))
                target_index += 1
            previous = (stamp, frame)
            if target_index == 8:
                break
        if previous is None:
            raise ValueError("No frames in the requested future interval")
        while len(chosen) < 8:
            chosen.append((previous[0] - start, previous[1].to_image()))
    return chosen, [t - start for t in targets], end - start


def make_sheet(clip, timeline, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    source = Path(clip.path)
    stat = source.stat()
    key = digest([SHEET_VERSION, str(source.resolve()), stat.st_size, stat.st_mtime_ns,
                  clip.start_seconds, clip.duration_seconds, timeline])
    path = cache / f"{key}.jpg"
    metadata = cache / f"{key}.json"
    with (cache / f"{key}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and metadata.exists():
            result = json.loads(metadata.read_text())
            if sha256(path) != result["image_sha256"]:
                raise ValueError("Cached contact sheet changed")
            return result
        frames, targets, sampled_duration = sample_frames(clip, timeline)
        # 4 columns x 2 rows; aspect ratios preserved, timestamps outside frames.
        width, height, label, header = 480, 360, 36, 48
        canvas = Image.new("RGB", (4 * width, header + 2 * (height + label)), "#151a23")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=24)
        draw.text((14, 10), f"8 frames | video: {clip.duration_seconds - clip.start_seconds:.3f}s | "
                  f"shown: 0-{sampled_duration:.3f}s | GT horizon: {timeline:.3f}s",
                  fill="white", font=font)
        timestamps = []
        for i, (stamp, image) in enumerate(frames):
            x, y = (i % 4) * width, header + (i // 4) * (height + label)
            thumb = ImageOps.contain(image, (width - 4, height - 4), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x + (width - thumb.width) // 2, y + (height - thumb.height) // 2))
            draw.text((x + 12, y + height + 4), f"#{i + 1}   t = {stamp:.3f} s", fill="white", font=font)
            timestamps.append(stamp)
        temporary = path.with_suffix(".tmp.jpg")
        canvas.save(temporary, quality=93, subsampling=0)
        temporary.replace(path)
        result = {
            "version": SHEET_VERSION, "image": str(path.resolve()),
            "image_sha256": sha256(path), "source": str(source.resolve()),
            "source_sha256": sha256(source), "start_seconds": clip.start_seconds,
            "source_duration_seconds": clip.duration_seconds,
            "sampled_duration_seconds": sampled_duration, "gt_horizon_seconds": timeline,
            "target_seconds": targets, "actual_seconds": timestamps,
            "unique_frames": len(set(timestamps)), "layout": [2, 4],
            "image_size": list(canvas.size),
        }
        write_json(metadata, result)
        return result
