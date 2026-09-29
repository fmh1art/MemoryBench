"""Use MemWorld's actual VAE/text encoders and sample_pt_v2 builders."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.case_definition"
import argparse
import json
import os
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("MEMWORLD_ROOT", "external/MemWorld")).resolve()
MODEL = Path(os.environ.get("WAN_MODEL_DIR", "models/Wan2.2-TI2V-5B"))


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def pack_one(root):
    began = time.monotonic()
    meta = json.loads((root / "case.json").read_text())
    if meta["status"] == "ready" and (root / "validation.json").exists():
        return
    scene = meta["evaluation_case_id"]
    packed = root / "packed"
    command = [
        sys.executable,
        "-u",
        "-m",
        "preprocess.sample_pt_v2.build_sample_dataset_v2",
        "--source_root",
        str(root / "context"),
        "--output_root",
        str(packed),
        "--model_dir",
        str(MODEL),
        "--width",
        str(meta["width"]),
        "--height",
        str(meta["height"]),
        "--enable_keypoint_condition",
        "--clip_batch_size",
        "1",
        "--context_encode_batch_size",
        "8",
        "--skip_existing",
        "--scene_names",
        scene,
    ]
    subprocess.run(command, cwd=REPO, check=True)
    context_link = packed / "context"
    if not context_link.exists():
        context_link.symlink_to("../context", target_is_directory=True)
    sys.path.insert(0, str(REPO))
    import torch
    import numpy as np
    from argparse import Namespace
    from preprocess.sample_pt_v2.stage_3_enrich_sample_v2 import (
        discover_sample_groups,
        load_entries,
        build_long_context,
        build_dynamic_context,
    )
    from preprocess.sample_pt_v2.sample_v2 import atomic_torch_save, validate_sample_v2
    from script.infer.infer_sample_pt_eval import (
        prepare_sample,
        seed_offline_history_cache,
        limit_episode_clip_entries,
    )
    from dataset.poses import load_c2ws_from_json

    entries = load_entries(discover_sample_groups(packed, [scene])[scene])
    c2ws = load_c2ws_from_json(root / "context/jsons" / f"{scene}.json")
    args = Namespace(
        dynamic_iou_weight=0.9,
        dynamic_freshness_weight=0.1,
        dynamic_freshness_mode="ratio",
        dynamic_freshness_tau=256.0,
        dynamic_overlap_num_samples=5000,
        dynamic_overlap_radius=50.0,
        dynamic_overlap_fov_half_h=45.0,
        dynamic_overlap_fov_half_v=30.0,
        dynamic_overlap_device="cuda",
        dynamic_overlap_chunk_size=256,
        dynamic_overlap_seed=0,
        write_dynamic_debug=True,
    )
    reports = []
    for entry in entries:
        predict = entry.clip_idx >= meta["inference_start_clip"]
        if predict:
            build_long_context(entry, entries, c2ws)
            build_dynamic_context(entry, entries, c2ws, args)
            atomic_torch_save(entry.sample, entry.path)
        errors = validate_sample_v2(entry.sample, require_enriched=predict)
        if errors:
            raise ValueError(f"{entry.path}: {errors}")
        if predict:
            prepared = prepare_sample(
                entry.sample, use_long_context=True, sample_path=entry.path
            )
            assert list(prepared["context_latents"].shape)[1] == 32
            assert list(prepared["target_latents"].shape)[1] == 9
            assert (
                prepared["context_keypoint_latents"].shape
                == prepared["context_latents"].shape
            )
            assert (
                prepared["target_keypoint_latents"].shape
                == prepared["target_latents"].shape
            )
            reports.append(
                {
                    "clip_idx": entry.clip_idx,
                    "context_latents": list(prepared["context_latents"].shape),
                    "target_latents": list(prepared["target_latents"].shape),
                    "target_keypoint_latents": list(
                        prepared["target_keypoint_latents"].shape
                    ),
                    "dynamic_sources": prepared["dynamic_context_source_frames"],
                }
            )
            print(f"enriched prediction clip {entry.clip_idx}", flush=True)
    assert len(entries) == meta["total_clips"]
    assert len(reports) == meta["generation_clips"]
    index = json.loads((packed / "index.json").read_text())
    tuples = [
        (item["clip_idx"], item, packed / item["path"]) for item in index["samples"]
    ]
    selected, window = limit_episode_clip_entries(
        tuples, None, meta["generation_clips"], start_clip=meta["inference_start_clip"]
    )
    assert (
        len(selected) == meta["generation_clips"]
        and window["start_entry_index"] == meta["inference_start_clip"]
    )
    cache, hand_cache = {}, {}
    seeded = seed_offline_history_cache(
        tuples[: window["start_entry_index"] + 1],
        meta["model_anchor_frame"],
        cache,
        hand_cache,
    )
    mapping = np.load(root / "frame_mapping.npz")
    prefix_end = meta["marks"].get("prefix_end", meta["marks"]["t"])
    candidate_sources = [int(mapping["source_frame"][i]) for i in cache]
    assert candidate_sources and min(candidate_sources) >= meta["marks"]["prefix_start"]
    assert max(candidate_sources) < prefix_end
    first = entries[meta["inference_start_clip"]].sample
    for field in [
        "long_context_source_frames",
        "short_context_source_frames",
        "dynamic_context_source_frames",
        "static_context_frame_indices",
    ]:
        assert all(
            i is None
            or meta["marks"]["prefix_start"]
            <= mapping["source_frame"][int(i)]
            < prefix_end
            for i in (first.get(field) or [])
        ), field
    validation = {
        "status": "passed",
        "total_samples": len(entries),
        "prediction_samples": len(reports),
        "history_seed": seeded,
        "history_source_min_frame": min(candidate_sources),
        "history_source_max_frame": max(candidate_sources),
        "prediction_tensor_shapes": reports,
        "checks": [
            "sample_pt_v2 structural validation",
            "actual inference prepare_sample",
            "RGB / hand shapes",
            "exact inference start window",
            "prefix cache availability",
            "first prediction memory RGB strictly inside annotated prefix",
        ],
        "packing_seconds": time.monotonic() - began,
    }
    write_json(root / "validation.json", validation)
    meta.update(status="ready", packing_seconds=validation["packing_seconds"])
    write_json(root / "case.json", meta)
    print(
        json.dumps(
            {"case": scene, "status": "ready", "seconds": validation["packing_seconds"]}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE.parents[1] / "data/evaluation_cases")
    parser.add_argument("--case", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3,4")
    args = parser.parse_args()
    if args.case:
        pack_one(args.case.resolve())
        return
    annotations = json.loads((args.root / "annotations_snapshot.json").read_text())
    folders = [
        args.root / f"eval_{i+1:03d}_{r['id'][:8]}" for i, r in enumerate(annotations)
    ]
    gpus = args.gpus.split(",")
    waiting = list(enumerate(folders))
    active = []
    failures = []
    while waiting or active:
        for item in list(waiting):
            index, folder = item
            gpu = gpus[index % len(gpus)]
            if not (folder / "case.json").exists() or any(
                row[0] == gpu for row in active
            ):
                continue
            log = (folder / "pack.log").open("a")
            process = subprocess.Popen(
                [sys.executable, "-u", __file__, "--case", str(folder.resolve())],
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "OMP_NUM_THREADS": "4",
                    "TOKENIZERS_PARALLELISM": "false",
                },
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active.append((gpu, process, log, folder))
            waiting.remove(item)
            print(f"packing {folder.name} on GPU {gpu}", flush=True)
        for item in list(active):
            gpu, process, log, folder = item
            if process.poll() is None:
                continue
            log.close()
            active.remove(item)
            if process.returncode:
                failures.append(folder.name)
            print(f"{folder.name} exit={process.returncode}", flush=True)
        if waiting or active:
            time.sleep(3)
    if failures:
        raise RuntimeError(
            f"Packing failures: {failures}; inspect individual pack.log files"
        )
    cases = [json.loads((p / "case.json").read_text()) for p in folders]
    write_json(
        args.root / "manifest.json",
        {
            "format": "acwm_memory_evaluation_cases_v1",
            "case_count": len(cases),
            "source_case_count": len({r["source_case_id"] for r in cases}),
            "status": "ready",
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
