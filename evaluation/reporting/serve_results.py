#!/usr/bin/env python3
"""Serve the review bundle with byte-range video support."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "evaluation.reporting"
import argparse
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "evaluation_results",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8891)
    args = p.parse_args()
    app = FastAPI()
    app.mount("/", StaticFiles(directory=args.root, html=True), name="results")
    uvicorn.run(app, host=args.host, port=args.port, timeout_graceful_shutdown=5)
