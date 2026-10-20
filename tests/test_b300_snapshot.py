"""CPU-only baseline and deployment decoder tests; no API/GPU/model requests."""

import asyncio
import copy
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from evaluation.inventory.schema import Judgment, ObjectGroup
from ops.b300_daily.inventory_decoder import (
    InventorySchemaTransport,
    constrained_inventory_schema,
)
from scripts.verify_b300_snapshot import verify


def test_captured_baseline_hashes():
    assert verify(Path(__file__).resolve().parents[1]) == 51


def test_inventory_decoder_preserves_native_schema():
    native = Judgment.model_json_schema()
    saved = copy.deepcopy(native)
    constrained = constrained_inventory_schema(native)
    assert native == saved
    restored = copy.deepcopy(constrained)
    restored["$defs"]["ObjectGroup"] = native["$defs"]["ObjectGroup"]
    assert restored == native
    assert len(constrained["$defs"]["ObjectGroup"]["anyOf"]) == 4
    changed = copy.deepcopy(native)
    changed["$defs"]["ObjectGroup"]["properties"]["gt_count"] = {"type": "integer"}
    with pytest.raises(ValueError, match="Unrecognized native count schema"):
        constrained_inventory_schema(changed)


@pytest.mark.parametrize("gt", [None, 0, 1, 2])
@pytest.mark.parametrize("gen", [None, 0, 1, 2])
def test_native_count_pairs(gt, gen):
    payload = dict(object_id="item", description="test item", gt_count=gt,
                   gen_count=gen, evidence=[dict(frame_indices=[1], observation="test")])
    if gt == 0 and gen == 0:
        with pytest.raises(ValidationError, match="absent in both videos"):
            ObjectGroup.model_validate(payload)
    else:
        assert ObjectGroup.model_validate(payload).gen_count == gen


def test_inventory_transport_only_changes_response_format():
    forwarded, audit = [], []
    schema = constrained_inventory_schema(Judgment.model_json_schema())
    body = dict(response_format={"type": "json_object"}, model="qwen3.8-27b",
                temperature=0, max_tokens=8192,
                chat_template_kwargs={"enable_thinking": False},
                messages=[{"role": "user", "content": "synthetic test"}])

    def backend(request):
        forwarded.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    async def run():
        transport = InventorySchemaTransport(schema, audit.append,
                                              backend=httpx.MockTransport(backend))
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post("http://127.0.0.1:18047/v1/chat/completions", json=body)
            with pytest.raises(ValueError, match="Unsupported inventory request identity"):
                await client.post("http://127.0.0.1:18047/v1/chat/completions",
                                  json=dict(body, model="unsupported"))

    asyncio.run(run())
    assert len(forwarded) == 1 and len(audit) == 1
    expected = dict(body, response_format=dict(type="json_schema", json_schema=dict(
        name="native_inventory_nonempty_object_group", schema=schema, strict=True)))
    assert forwarded[0] == expected
