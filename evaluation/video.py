"""Aligned decoded-PTS sampling: Gate contact sheets and standalone Memory frame pairs."""

from bisect import bisect_left
from math import isfinite
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont, ImageOps

MEMORY_FRACTIONS = tuple(round(0.1 + 0.8 * index / 7, 12) for index in range(8))


def _timestamps(path: str) -> list[float]:
    times = []
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream: {path}")
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError(f"Frame without presentation timestamp: {path}")
            timestamp = float(frame.pts * frame.time_base)
            if times and timestamp <= times[-1]:
                raise ValueError(f"Non-increasing video timestamps: {path}")
            times.append(timestamp)
    if not times:
        raise ValueError(f"Empty video: {path}")
    return [time - times[0] for time in times]


def _nearest(times: list[float], target: float) -> int:
    right = bisect_left(times, target)
    candidates = {max(0, right - 1), min(len(times) - 1, right)}
    return min(candidates, key=lambda index: (abs(times[index] - target), index))


def _sheet(
    path, role, times, indices, targets, offset, output, tile_width, tile_height
):
    header, label_height = 44, 32
    sheet = Image.new(
        "RGB", (6 * tile_width, header + 2 * (tile_height + label_height)), "#171b24"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    draw.text(
        (12, 12),
        f"{role} | row-major frames 1-12 | shared experiment time",
        fill="white",
        font=font,
    )
    slots = {}
    for slot, index in enumerate(indices):
        slots.setdefault(index, []).append(slot)
    with av.open(path) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in slots:
                # contain preserves aspect ratio; canvas provides letterbox padding.
                image = ImageOps.contain(
                    frame.to_image().convert("RGB"), (tile_width, tile_height)
                )
                for slot in slots[index]:
                    x = (slot % 6) * tile_width
                    y = header + (slot // 6) * (tile_height + label_height)
                    draw.text(
                        (x + 5, y + 5),
                        f"#{slot + 1} t={targets[slot]:.3f}s frame={times[index] - offset:.3f}s",
                        fill="white",
                        font=font,
                    )
                    sheet.paste(
                        image,
                        (
                            x + (tile_width - image.width) // 2,
                            y + label_height + (tile_height - image.height) // 2,
                        ),
                    )
            if index >= max(indices):
                break
    destination = output / f"I_{role.lower()}.jpg"
    sheet.save(destination, quality=95)
    return destination


def make_sheets(
    case, output: Path, tile_width=480, tile_height=320, max_skew_seconds=0.1
):
    if tile_width < 420 or tile_height < 64:
        raise ValueError(
            "Tiles must be at least 420px wide and 64px high for readable timestamps"
        )
    if not isfinite(max_skew_seconds) or max_skew_seconds < 0:
        raise ValueError("max_skew_seconds must be nonnegative")
    gt_times, gen_times = _timestamps(case.gt_video), _timestamps(case.gen_video)
    end = min(
        gt_times[-1] - case.gt_start_seconds, gen_times[-1] - case.gen_start_seconds
    )
    if end <= 0:
        raise ValueError("Videos have no positive shared duration after start offsets")
    read_end = end if case.read_end_seconds is None else case.read_end_seconds
    if not 0 <= case.read_start_seconds <= read_end <= end:
        raise ValueError(
            f"Read interval must lie within shared timeline [0, {end:.6f}] seconds"
        )
    targets = [end * index / 11 for index in range(12)]
    # Preserve exact interval endpoints; floating arithmetic can overshoot end.
    targets[0], targets[-1] = 0.0, end
    read_indices = [
        i + 1 for i, t in enumerate(targets) if case.read_start_seconds <= t <= read_end
    ]
    if not read_indices:
        raise ValueError(
            "Uniform sampling misses the Read interval; provide appropriately trimmed video clips"
        )
    gt_indices = [_nearest(gt_times, t + case.gt_start_seconds) for t in targets]
    gen_indices = [_nearest(gen_times, t + case.gen_start_seconds) for t in targets]
    frames = []
    for slot, (t, gt_index, gen_index) in enumerate(
        zip(targets, gt_indices, gen_indices), 1
    ):
        gt_actual = gt_times[gt_index] - case.gt_start_seconds
        gen_actual = gen_times[gen_index] - case.gen_start_seconds
        skew = max(abs(gt_actual - t), abs(gen_actual - t), abs(gt_actual - gen_actual))
        if skew > max_skew_seconds + 1e-9:
            raise ValueError(
                f"Frame #{slot} cannot align within {max_skew_seconds}s (skew={skew:.6f}s)"
            )
        frames.append(
            {
                "frame_index": slot,
                "time_seconds": t,
                "gt_actual_seconds": gt_actual,
                "gen_actual_seconds": gen_actual,
                "gt_frame_index": gt_index,
                "gen_frame_index": gen_index,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    gt_path = _sheet(
        case.gt_video,
        "GT",
        gt_times,
        gt_indices,
        targets,
        case.gt_start_seconds,
        output,
        tile_width,
        tile_height,
    )
    gen_path = _sheet(
        case.gen_video,
        "GEN",
        gen_times,
        gen_indices,
        targets,
        case.gen_start_seconds,
        output,
        tile_width,
        tile_height,
    )
    metadata = {
        "shared_end_seconds": end,
        "read_end_seconds": read_end,
        "read_frame_indices": read_indices,
        "frames": frames,
        "gt_unique_frames": len(set(gt_indices)),
        "gen_unique_frames": len(set(gen_indices)),
        "gt_uncovered_tail_seconds": gt_times[-1] - case.gt_start_seconds - end,
        "gen_uncovered_tail_seconds": gen_times[-1] - case.gen_start_seconds - end,
        "max_skew_seconds": max_skew_seconds,
        "tile_size": [tile_width, tile_height],
    }
    return gt_path, gen_path, metadata


def make_memory_pairs(case, output, gate_sampling, max_side=1024):
    """Eight aligned frame pairs spanning 10–90% of the shared video timeline."""
    end = gate_sampling["shared_end_seconds"]
    targets = [end * fraction for fraction in MEMORY_FRACTIONS]
    times = [_timestamps(case.gt_video), _timestamps(case.gen_video)]
    offsets = [case.gt_start_seconds, case.gen_start_seconds]
    indices = [
        [_nearest(clock, t + offset) for t in targets]
        for clock, offset in zip(times, offsets)
    ]
    pairs = []
    directory = output / "memory_frames"
    directory.mkdir(exist_ok=True)
    for slot, t in enumerate(targets):
        actual = [
            clock[index[slot]] - offset
            for clock, index, offset in zip(times, indices, offsets)
        ]
        skew = max(abs(actual[0] - t), abs(actual[1] - t), abs(actual[0] - actual[1]))
        if skew > gate_sampling["max_skew_seconds"] + 1e-9:
            raise ValueError(f"Memory frame #{slot + 1} cannot align: skew={skew:.6f}s")
        pairs.append(
            {
                "frame_index": slot + 1,
                "time_seconds": t,
                "gt_actual_seconds": actual[0],
                "gen_actual_seconds": actual[1],
                "gt_frame_index": indices[0][slot],
                "gen_frame_index": indices[1][slot],
                "is_read": case.read_start_seconds
                <= t
                <= gate_sampling["read_end_seconds"],
            }
        )
    for role, path, wanted in zip(
        ("gt", "gen"), (case.gt_video, case.gen_video), indices
    ):
        with av.open(path) as container:
            for index, frame in enumerate(container.decode(video=0)):
                for slot in [
                    slot for slot, desired in enumerate(wanted) if desired == index
                ]:
                    image = frame.to_image().convert("RGB")
                    # No collage, timestamps or stretching in the Judge's image input.
                    if max(image.size) > max_side:
                        image = ImageOps.contain(image, (max_side, max_side))
                    image.save(directory / f"{slot + 1}_{role}.jpg", quality=95)
                if index >= max(wanted):
                    break
    return {
        "fractions": list(MEMORY_FRACTIONS),
        "interval": [MEMORY_FRACTIONS[0] * end, MEMORY_FRACTIONS[-1] * end],
        "frames": pairs,
        "max_side": max_side,
        "gt_unique_frames": len(set(indices[0])),
        "gen_unique_frames": len(set(indices[1])),
    }
