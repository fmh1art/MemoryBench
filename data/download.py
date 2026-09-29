"""Download and checksum-verify MemoryBenchData using a direct connection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('data'))
    p.add_argument('--repo', default='fmh1art/MemoryBenchData')
    p.add_argument('--download-only', action='store_true')
    args = p.parse_args()
    for key in list(os.environ):
        if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
            del os.environ[key]
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '*'
    from modelscope.hub.snapshot_download import dataset_snapshot_download
    from modelscope.hub.api import HubApi
    token = os.environ.get('MODELSCOPE_API_TOKEN')
    if token:
        HubApi().login(token)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = Path(dataset_snapshot_download(args.repo, local_dir=str(destination / 'MemoryBenchData')))
    records = json.loads((snapshot / 'archives.json').read_text())
    for record in records:
        archive = (snapshot / record['path']).resolve()
        if not archive.is_relative_to(snapshot.resolve()):
            raise ValueError('Unsafe archive path')
        if archive.stat().st_size != record['bytes']:
            raise ValueError(f'Size mismatch: {archive.name}')
        digest = hashlib.sha256()
        with archive.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != record['sha256']:
            raise ValueError(f'Checksum mismatch: {archive.name}')
        print(f'Verified {archive.name}', flush=True)
    if not args.download_only:
        if not hasattr(tarfile, 'data_filter'):
            raise RuntimeError('Upgrade Python to a security-patched 3.11+ with tarfile.data_filter')
        for record in records:
            with tarfile.open(snapshot / record['path'], 'r:gz') as archive:
                archive.extractall(destination, filter='data')
        print(f'Cases extracted to {destination / "evaluation_cases"}')


if __name__ == '__main__':
    main()
