"""Reviewed checkout amendment; the frozen 38k scoring/input contract is unchanged.

No automatic revision adoption. Unknown commits, manifests or semantic changes fail.
The 2026-10-17 audit compared all 649 case JSONs and evaluation_cases.jsonl:
only input_contract.future_action_text was added. No existing media changed.
That new text is deliberately NOT consumed by this pipeline.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess

CANONICAL_COMMIT = '35cb2c1623275b519c55893656263b44cb66f055'
CANONICAL_MANIFEST = '3df7d6d9f29dca2968c14f44c71b485f6bfb9f748b365f337bdafaa85728239b'
CHECKOUT_COMMIT = '5bc2176a8a4d437769b2a0182c08d1d9e5dd958f'
CHECKOUT_MANIFEST = '258d97e68cbb0b643fe6ba5641af2bd604fe1d04ecd1d8575d1136edfe18f4ae'
POLICY = 'reviewed_additive_future_action_metadata_20261017'


def validate_addition(original, current):
    normalized = copy.deepcopy(current)
    if len(original['cases']) != 649 or len(normalized['cases']) != 649:
        raise ValueError('Benchmark amendment requires the same 649 cases')
    ids = [c['evaluation_case_id'] for c in original['cases']]
    if len(set(ids)) != 649:
        raise ValueError('Canonical case IDs are not unique')
    for old, new in zip(original['cases'], normalized['cases']):
        if 'future_action_text' in old['input_contract']:
            raise ValueError('Canonical contract unexpectedly includes future action text')
        if new['input_contract'].pop('future_action_text', None) != 'input/future_action.txt':
            raise ValueError('Unreviewed benchmark metadata addition')
    if normalized != original:
        raise ValueError('Benchmark semantics changed beyond the reviewed ignored text field')


def validate_checkout(dataset, reference, recipe):
    data = Path(dataset)
    if (recipe['benchmark_commit'] != CANONICAL_COMMIT
            or reference['manifest_sha256'] != CANONICAL_MANIFEST):
        raise ValueError('Canonical benchmark reference changed')
    commit = subprocess.check_output(['git', '-C', str(data), 'rev-parse', 'HEAD'],
                                     text=True, timeout=30).strip()
    raw = (data/'manifest.json').read_bytes()
    if commit != CHECKOUT_COMMIT or hashlib.sha256(raw).hexdigest() != CHECKOUT_MANIFEST:
        raise ValueError('Unreviewed benchmark checkout/manifest; operator review required')
    # The original manifest is an immutable LFS object already present on B300.
    old = (data/'.git/lfs/objects'/CANONICAL_MANIFEST[:2]/CANONICAL_MANIFEST[2:4]/CANONICAL_MANIFEST).read_bytes()
    if hashlib.sha256(old).hexdigest() != CANONICAL_MANIFEST:
        raise ValueError('Canonical cached manifest changed')
    validate_addition(json.loads(old), json.loads(raw))
    return dict(policy=POLICY, canonical_commit=CANONICAL_COMMIT,
                canonical_manifest_sha256=CANONICAL_MANIFEST,
                observed_checkout_commit=commit, observed_manifest_sha256=CHECKOUT_MANIFEST,
                added_future_action_text_used=False, cases=649,
                canonical_case_source='frozen reference_run.json; only gen_video prefix remapped')
