#!/usr/bin/env python3
"""Score one annotated pair of stable videos and its separate generated action clip."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.pipeline"
import argparse
from ..common.config import load_config
from ..pipeline.pipeline import evaluate
from ..metric.scoring import metric_table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gt", type=Path, required=True, help="GT annotated stable evaluation clip"
    )
    p.add_argument(
        "--generated",
        type=Path,
        required=True,
        help="Generated annotated stable evaluation clip",
    )
    p.add_argument(
        "--action",
        type=Path,
        required=True,
        help="Generated action clip, t to action_end",
    )
    p.add_argument(
        "--annotation", type=Path, required=True, help="Explicit memory_annotation.json"
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/local.toml",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True, help="New/empty result directory"
    )
    args = p.parse_args()
    try:
        result = evaluate(
            args.gt,
            args.generated,
            load_config(args.config),
            args.output_dir,
            action_path=args.action,
            annotation_path=args.annotation,
            progress=lambda s: print(s, file=sys.stderr, flush=True),
        )
        print(metric_table(result))
        print(f"Memory: {result['memory_score']:.6f}")
        return 0
    except Exception as exc:
        print(f"Evaluation failed (no score): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
