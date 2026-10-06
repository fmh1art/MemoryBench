"""Serve saved scores, contact sheets and paired videos with Python's standard library."""

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

STATIC = Path(__file__).parent / "static"


class Catalog:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.results = {}
        self.directories = {}
        self.cases = []
        for path in sorted(self.root.glob("*/result.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            case_id = result["case_id"]
            if case_id != path.parent.name or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id
            ):
                raise ValueError(f"Invalid case identity in {path}")
            category = re.sub(r"_\d+_\d+$", "", case_id)
            self.results[case_id] = result
            self.directories[case_id] = path.parent
            self.cases.append(
                {
                    "case_id": case_id,
                    "category": category,
                    "status": result["status"],
                    "memory_score": result.get("memory_score"),
                    "memory_target": result["case"]["memory_target"],
                    "action_description": result["case"]["action_description"],
                    "dimensions": result.get("scores", {}).get("dimensions", {}),
                }
            )
        if not self.cases:
            raise ValueError("No */result.json files found in the result directory")
        summary_path = self.root / "summary.json"
        self.summary = (
            json.loads(summary_path.read_text()) if summary_path.exists() else {}
        )
        run_path = self.root / "run.json"
        run = json.loads(run_path.read_text()) if run_path.exists() else {}
        self.model = run.get("model", "VLM")
        self.protocol = run.get("protocol", "memory_evaluation.contact_sheets.v1")

    def media(self, case_id, kind):
        result = self.results.get(case_id)
        if result is None:
            return None
        directory = self.directories[case_id]
        paths = {
            "gt_sheet": directory / "I_gt.jpg",
            "gen_sheet": directory / "I_gen.jpg",
            "gt_video": Path(result["case"]["gt_video"]),
            "gen_video": Path(result["case"]["gen_video"]),
        }
        match = re.fullmatch(r"(gt|gen)_frame_([1-4])", kind)
        if match and result.get("memory_sampling"):
            paths[kind] = directory / "memory_frames" / f"{match[2]}_{match[1]}.jpg"
        path = paths.get(kind)
        return path if path is not None and path.is_file() else None


def byte_range(header, size):
    """Return an inclusive single HTTP byte range; browsers use it to seek MP4s."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or size == 0 or not any(match.groups()):
        raise ValueError("Invalid range")
    first, last = match.groups()
    if first:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
    else:
        suffix = int(last)
        if suffix == 0:
            raise ValueError("Empty suffix range")
        start, end = max(0, size - suffix), size - 1
    if start >= size or start > end:
        raise ValueError("Unsatisfiable range")
    return start, end


def handler_for(catalog):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.route()

        def do_HEAD(self):
            self.route(head=True)

        def log_message(self, *_args):
            pass

        def json(self, value, head):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def file(self, path, head):
            size = path.stat().st_size
            start, end = 0, size - 1
            ranged = self.headers.get("Range")
            if ranged:
                try:
                    start, end = byte_range(ranged, size)
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(206 if ranged else 200)
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            )
            self.send_header("Content-Length", str(max(0, end - start + 1)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-cache")
            if ranged:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not head:
                with path.open("rb") as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        block = stream.read(min(64 * 1024, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)

        def route(self, head=False):
            path = unquote(urlsplit(self.path).path)
            try:
                if path in ("/", "/app.js", "/style.css"):
                    self.file(
                        STATIC / ("index.html" if path == "/" else path[1:]), head
                    )
                    return
                if path == "/api/catalog":
                    self.json(
                        {
                            "model": catalog.model,
                            "protocol": catalog.protocol,
                            "summary": catalog.summary,
                            "cases": catalog.cases,
                        },
                        head,
                    )
                    return
                if path.startswith("/api/case/"):
                    result = catalog.results.get(path.removeprefix("/api/case/"))
                    if result is not None:
                        self.json(result, head)
                        return
                parts = path.strip("/").split("/")
                if len(parts) == 3 and parts[0] == "media":
                    media = catalog.media(parts[1], parts[2])
                    if media is not None:
                        self.file(media, head)
                        return
                self.send_error(404, "Not found")
            except (BrokenPipeError, ConnectionResetError):
                # Switching cases aborts old image/video downloads normally.
                return

    return Handler


def create_server(results, host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), handler_for(Catalog(results)))
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = create_server(args.results, args.host, args.port)
    print(f"MemoryBench viewer: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
