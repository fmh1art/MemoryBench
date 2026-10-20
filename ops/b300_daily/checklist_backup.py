"""Small immutable CSV snapshots, isolated by publication stage and content."""
import hashlib
import os
from pathlib import Path


def save(work, stage, content):
    if stage not in ('archive', 'inference', 'evaluation') or not isinstance(content, bytes):
        raise ValueError('Explicit supported checklist stage and bytes are required')
    folder = Path(work)/'checklist-backups'/stage
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(hashlib.sha256(content).hexdigest()+'.csv')
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise ValueError('Existing immutable checklist backup differs')
        return path
    # The caller owns the common publisher lock. Never replace a backup.
    with path.open('xb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    return path
