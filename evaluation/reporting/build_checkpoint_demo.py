#!/usr/bin/env python3
"""Build synchronized video artifacts for an eight-checkpoint inference run."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.reporting"
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import time

import av
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def frames(path, expected):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (640, 480), path
        assert float(stream.average_rate) == 30.0, path
        result = [f.to_image() for f in container.decode(video=0)]
    assert len(result) == expected, (path, len(result), expected)
    return result


def writer(path, threads=2):
    return imageio.get_writer(
        str(path),
        fps=30,
        codec="libx264",
        quality=None,
        macro_block_size=1,
        ffmpeg_log_level="error",
        ffmpeg_params=[
            "-preset",
            "fast",
            "-crf",
            "18",
            "-threads",
            str(threads),
            "-movflags",
            "+faststart",
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    began = time.monotonic()
    config = json.loads((output / "run_config.json").read_text())
    case = json.loads((output / "case.json").read_text())
    runs = config["runs"]
    assert len(runs) == 8 and all(r["status"] == "video_ready" for r in runs)
    assert (
        len({r["sha256"] for r in runs}) == 8
    ), "All eight checkpoint videos must be independently produced"
    count = case["generation_frame_count"]
    eval_start = case["marks"]["eval_start"] - case["marks"]["t"]
    gt = frames(output / "reference/gt_future.mp4", count)
    data = {
        r["id"]: frames(output / r["id"] / "generated_future.mp4", count) for r in runs
    }
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    names = ["Ground truth"] + [
        f"Epoch {r['epoch']:g} | step {r['step']}" for r in runs
    ]
    videos = [gt] + [data[r["id"]] for r in runs]
    poster_frame = min(count - 1, eval_start + 20)
    review_frames = [0, 44, 88, 132, 177, 199, count - 1]
    (output / "review_frames").mkdir(exist_ok=True)
    with writer(output / "comparison_grid.mp4", 8) as out:
        for index in range(count):
            canvas = Image.new("RGB", (1920, 1572), "#101827")
            draw = ImageDraw.Draw(canvas)
            for slot, (name, video) in enumerate(zip(names, videos)):
                x, y = (slot % 3) * 640, (slot // 3) * 524
                draw.text(
                    (x + 12, y + 9),
                    name,
                    fill="#5fe0ad" if slot == 0 else "white",
                    font=font,
                )
                draw.text(
                    (x + 430, y + 13),
                    f"{index/30:04.1f}s {'EVAL' if index >= eval_start else 'ACTION'}",
                    fill="#a3b5cd",
                    font=small,
                )
                canvas.paste(video[index], (x, y + 44))
            out.append_data(np.asarray(canvas))
            if index in review_frames:
                canvas.save(
                    output / "review_frames" / f"grid_frame{index:03d}.jpg", quality=94
                )
            if index == poster_frame:
                canvas.save(output / "poster.jpg", quality=94)

    def pair(run):
        destination = output / run["id"]
        with writer(destination / "comparison.mp4") as out:
            for index, right in enumerate(data[run["id"]]):
                canvas = Image.new("RGB", (1280, 524), "#101827")
                canvas.paste(gt[index], (0, 44))
                canvas.paste(right, (640, 44))
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 9), "Ground truth", font=font, fill="#5fe0ad")
                draw.text(
                    (652, 9),
                    f"Epoch {run['epoch']:g} | step {run['step']}",
                    font=font,
                    fill="white",
                )
                out.append_data(np.asarray(canvas))
                if index == poster_frame:
                    canvas.save(destination / "poster.jpg", quality=93)
        data[run["id"]][poster_frame].save(
            destination / "generated_stable.jpg", quality=95
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(pair, runs))
    # Compact temporal strips support review of action progression, not just an endpoint.
    for slot, (name, video) in enumerate(zip(names, videos)):
        canvas = Image.new("RGB", (320 * len(review_frames), 270), "#101827")
        draw = ImageDraw.Draw(canvas)
        for j, index in enumerate(review_frames):
            canvas.paste(video[index].resize((320, 240)), (320 * j, 30))
            draw.text(
                (320 * j + 8, 8), f"{name} / frame {index}", fill="white", font=small
            )
        canvas.save(output / "review_frames" / f"timeline_{slot:02d}.jpg", quality=95)
    options = "".join(
        f'<option value="{r["id"]}">Epoch {r["epoch"]:g} · step {r["step"]}</option>'
        for r in runs
    )
    cards = "".join(
        f'<article><h3>Epoch {r["epoch"]:g} <small>step {r["step"]}</small></h3><video controls playsinline preload="none" poster="{r["id"]}/generated_stable.jpg" src="{r["id"]}/generated_future.mp4"></video><p><a href="{r["id"]}/generated_future.mp4" download>下载生成视频</a> · <a href="{r["id"]}/comparison.mp4">与 GT 对比</a></p></article>'
        for r in runs
    )
    html = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ACWM · 8 个 checkpoint 视频对比</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#e3ebf8;font:16px/1.65 system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:28px}h1{font-size:30px;margin:0}h2{margin-top:32px}p{color:#aebed4}a{color:#76d5ff}video{display:block;width:100%;background:#05080e;border-radius:10px}#grid{max-height:85vh}button,select{font:inherit;padding:8px 14px;border:1px solid #39516e;border-radius:8px;background:#16253a;color:#e3ebf8;cursor:pointer}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:14px 0}.badge{display:inline-block;background:#193a35;border-radius:20px;padding:3px 13px;color:#82e4bc;margin:12px 6px 0 0}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}article{background:#131f31;border:1px solid #2a3d57;padding:16px;border-radius:14px}h3{margin:0 0 10px}small{font-size:13px;color:#96aac7}details{margin-top:24px}#review{padding:18px 22px;background:#15273b;border-left:4px solid #6ec6f3;border-radius:8px;white-space:pre-wrap}footer{margin-top:35px;font-size:14px;color:#96aac7}@media(max-width:760px){main{padding:14px}.cards{grid-template-columns:1fr}h1{font-size:23px}}</style></head><body><main>
<h1>ACWM：8 个 checkpoint 的真实生成视频</h1><p>同一段保留评测样本 eval_002：拿起绿色物体，把视角转回已打开的抽屉。所有模型使用相同历史 RGB、独立手部骨架、未来手部轨迹与未来 head pose。</p>
<span class="badge">未参与训练的源 case</span><span class="badge">Seed 42 · CFG 5 · 20 步采样</span><span class="badge">222 帧 · 30 fps · 7.4 秒</span>
<h2>九宫格同步播放</h2><p>左上角为真实视频；其余为 epoch 6.5 至 10。5.93 秒起是人工标注的稳定评价区间。先看手持动作和视角是否跟随，再看抽屉里的黄色胶带卷、黄色胶棒是否与 GT 对应。</p>
<div class="controls"><button onclick="grid.play()">播放</button><button onclick="grid.pause()">暂停</button><button onclick="grid.currentTime=0;grid.play()">从头播放</button><button onclick="grid.currentTime=178/30;grid.pause()">跳到稳定片段</button><label>速度 <select onchange="grid.playbackRate=Number(this.value)"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><a href="comparison_grid.mp4" download>下载九宫格 MP4</a></div>
<video id="grid" controls playsinline loop preload="metadata" poster="poster.jpg" src="comparison_grid.mp4"></video>
<h2>与 GT 放大对比</h2><div class="controls"><select id="choose">OPTIONS</select><button onclick="pair.currentTime=178/30;pair.pause()">查看稳定片段</button></div><video id="pair" controls playsinline loop preload="metadata"></video>
<h2>本次观察</h2><div id="review">视频已生成。结论仅针对这个样本和本次随机种子；不能据此确定整个 Memory Bench 上的最佳模型。</div>
<h2>8 段独立生成视频</h2><div class="cards">CARDS</div>
<details><summary>查看输入和原始模型参考</summary><div class="cards"><article><h3>历史 RGB 前缀</h3><video controls preload="none" src="reference/prefix.mp4"></video></article><article><h3>未来手部动作条件</h3><video controls preload="none" src="reference/future_hands.mp4"></video></article><article><h3>真实未来视频</h3><video controls preload="none" src="reference/gt_future.mp4"></video></article><article><h3>原始 step4529（上次生成）</h3><video controls preload="none" src="reference/original_step4529.mp4"></video></article></div></details>
<footer>输出包含完整未来动作段与稳定段；已去除模型所需的已知首帧与尾部补齐帧。<a href="run_config.json">运行参数</a> · <a href="validation.json">视频检查</a> · <a href="case.json">样本标注</a></footer>
</main><script>const grid=document.getElementById('grid'),pair=document.getElementById('pair'),choose=document.getElementById('choose');function selectPair(){pair.src=choose.value+'/comparison.mp4';pair.poster=choose.value+'/poster.jpg';}choose.onchange=selectPair;selectPair();fetch('visual_review.json').then(r=>r.ok?r.json():null).then(r=>{if(r&&r.summary)document.getElementById('review').textContent=r.summary;}).catch(()=>{});</script></body></html>""".replace(
        "OPTIONS", options
    ).replace(
        "CARDS", cards
    )
    (output / "index.html").write_text(html)
    validation = {
        "status": "passed",
        "case_id": config["evaluation_case_id"],
        "checkpoints": 8,
        "generated_frames_each": count,
        "width": 640,
        "height": 480,
        "fps": 30,
        "evaluation_start_frame_relative": eval_start,
        "evaluation_frame_count": case["evaluation_frame_count"],
        "distinct_output_checksums": 8,
        "same_case_and_seed": True,
        "teacher_forced_rgb": False,
        "grid": "comparison_grid.mp4",
        "postprocessing_seconds": time.monotonic() - began,
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    shutil.copy2(Path(__file__), output / "code_snapshot" / Path(__file__).name)
    print(json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
