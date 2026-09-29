from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Frame:
    role: str
    index: int
    pts: int
    time: float  # seconds from this video's first decoded frame

    @property
    def id(self):
        return f"{self.role}_f{self.index:06d}"

    def as_dict(self):
        return {
            "frame_id": self.id,
            "index": self.index,
            "pts": self.pts,
            "time_seconds": self.time,
        }


def uniform_frames(frames: list[Frame], count: int, pad=False):
    """Uniform in video time, not frame number; never reuse a frame to inflate coverage."""
    if not frames:
        return [None] * count if pad else []
    if len(frames) < count and not pad:
        return frames[:]
    times = np.array([f.time for f in frames])
    targets = np.linspace(times[0], times[-1], count)
    chosen, used = [], set()
    for target in targets:
        available = [i for i in range(len(frames)) if i not in used]
        if not available:
            chosen.append(None)
            continue
        index = min(available, key=lambda i: (abs(times[i] - target), i))
        chosen.append(frames[index])
        used.add(index)
    # Sorting preserves chronological order even with highly irregular timestamps.
    present = sorted((f for f in chosen if f is not None), key=lambda f: f.index)
    if pad and len(present) < count:
        # Spread sparse available frames across normalized progress; absent slots count zero.
        slots = [None] * count
        for slot, frame in zip(
            np.linspace(0, count - 1, len(present)).round().astype(int), present
        ):
            slots[int(slot)] = frame
        return slots
    return present


class Video:
    def __init__(self, path: Path, role: str, output: Path, image_max_side=768):
        self.path = path.resolve()
        self.role = role
        self.output = output / "frames" / role
        self.output.mkdir(parents=True, exist_ok=True)
        self.image_max_side = image_max_side
        self.frames = []
        if not self.path.is_file():
            raise ValueError(f"Video does not exist: {self.path}")
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            self.time_base = stream.time_base
            start = None
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is None:
                    raise ValueError(
                        f"{role}: video frame has no presentation timestamp"
                    )
                time = float(frame.pts * frame.time_base)
                start = time if start is None else start
                if self.frames and time - start <= self.frames[-1].time:
                    raise ValueError(f"{role}: non-increasing video PTS")
                self.frames.append(Frame(role, index, frame.pts, time - start))
                self.width, self.height = frame.width, frame.height
        if not self.frames:
            raise ValueError(f"{role}: no decodable video frames")
        digest = sha256()
        with self.path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        self.checksum = digest.hexdigest()

    def metadata(self):
        return {
            "path": str(self.path),
            "sha256": self.checksum,
            "frame_count": len(self.frames),
            "width": self.width,
            "height": self.height,
            "first_to_last_frame_seconds": self.frames[-1].time,
            "time_base": [self.time_base.numerator, self.time_base.denominator],
        }

    def image(self, frame: Frame):
        path = self.output / f"{frame.id}.jpg"
        if path.exists():
            return path
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            target = frame.pts * self.time_base
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            for decoded in container.decode(stream):
                time = decoded.pts * decoded.time_base
                if time == target:
                    img = decoded.to_image()
                    img.thumbnail((self.image_max_side, self.image_max_side))
                    img.save(path, quality=92)
                    return path
                if time > target:
                    break
        raise ValueError(f"Could not decode exact frame {frame.id}")

    def contact_sheet(self, frames):
        # Only used to identify the physical target, never to grade object details.
        cols, cell_w, cell_h = 4, 320, 264
        sheet = Image.new(
            "RGB",
            (cols * cell_w, ((len(frames) + cols - 1) // cols) * cell_h),
            "#18232f",
        )
        draw = ImageDraw.Draw(sheet)
        for pos, frame in enumerate(frames):
            with Image.open(self.image(frame)) as source:
                img = source.copy()
            img.thumbnail((cell_w - 8, cell_h - 30))
            x, y = (pos % cols) * cell_w, (pos // cols) * cell_h
            sheet.paste(img, (x + (cell_w - img.width) // 2, y + 25))
            draw.text((x + 5, y + 5), f"{frame.id}  {frame.time:.3f}s", fill="white")
        path = self.output.parent / f"{self.role}_overview.jpg"
        sheet.save(path, quality=90)
        return path
