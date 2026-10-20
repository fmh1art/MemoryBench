"""Verify the recorded production baseline without importing evaluation runtimes."""

import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "B300_SNAPSHOT.json").read_text())
    if manifest["schema"] != "memorybench_b300_code_snapshot_v1":
        raise ValueError("Unsupported snapshot manifest schema")
    failures = []
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest path: {name}")
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Snapshot path escapes regular source files: {name}")
        if not path.is_file():
            failures.append(f"missing: {name}")
            continue
        content = path.read_bytes()
        if (len(content) != expected["size"]
                or hashlib.sha256(content).hexdigest() != expected["sha256"]):
            failures.append(f"changed: {name}")
    if failures:
        raise ValueError("Not the captured B300 baseline:\n" + "\n".join(failures))
    return len(manifest["files"])


if __name__ == "__main__":
    count = verify(Path(__file__).resolve().parents[1])
    print(f"Verified {count} source/protocol files against B300_SNAPSHOT.json")
