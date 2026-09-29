"""Build frame-synchronous GT / generated video review assets."""

import av
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def comparison_video(gt, generated, output, expected):
    """Frame-synchronous review artifact; VLM receives the separate original videos."""
    count = 0
    selected = set(np.linspace(0, expected - 1, 5).round().astype(int).tolist())
    panels = []
    with av.open(str(gt)) as gc, av.open(str(generated)) as pc:
        with imageio.get_writer(
            str(output / "comparison.mp4"),
            fps=30,
            codec="libx264",
            macro_block_size=1,
            quality=None,
            ffmpeg_log_level="error",
            ffmpeg_params=["-crf", "16", "-threads", "2"],
        ) as writer:
            for gf, pf in zip(gc.decode(video=0), pc.decode(video=0), strict=True):
                left, right = gf.to_image(), pf.to_image()
                if left.size != right.size:
                    raise ValueError("GT and generated dimensions differ")
                w, h = left.size
                panel = Image.new("RGB", (2 * w, h + 36), "#101827")
                panel.paste(left, (0, 36))
                panel.paste(right, (w, 36))
                draw = ImageDraw.Draw(panel)
                draw.text((12, 10), f"GROUND TRUTH | frame {count}", fill="white")
                draw.text((w + 12, 10), f"ACWM GENERATED | frame {count}", fill="white")
                writer.append_data(np.asarray(panel))
                if count in selected:
                    panels.append(
                        panel.resize((800, round(panel.height * 800 / panel.width)))
                    )
                count += 1
    if count != expected:
        raise ValueError(f"Comparison frame count {count} != {expected}")
    sheet = Image.new("RGB", (800, sum(p.height for p in panels)))
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(output / "comparison.jpg", quality=92)
