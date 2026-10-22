"""One Gate call, or one selected dimension per frame pair; bounded requests and audit logs."""

import asyncio
import base64
import hashlib
import json
import time
from math import isfinite
from pathlib import Path

import httpx
from pydantic import create_model

from .appearance.schema import Judgment as Appearance
from .environment.schema import Judgment as Environment
from .gate.schema import Judgment as Gate
from .inventory.schema import Judgment as Inventory
from .rate_limit import RequestPacer
from .schema import Metric, Record
from .spatial.schema import Judgment as Spatial

ROOT = Path(__file__).parent
SCHEMAS = {
    "gate": Gate,
    "inventory": Inventory,
    "appearance": Appearance,
    "spatial": Spatial,
    "environment": Environment,
}


def write_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def validate_evidence(dimension, judgment, read_indices, lighting_style):
    data = judgment.model_dump()

    def walk(value, allowed):
        if isinstance(value, dict):
            if "frame_indices" in value and not set(value["frame_indices"]) <= allowed:
                raise ValueError("Evidence must cite permitted Read frame indices")
            for child in value.values():
                walk(child, allowed)
        elif isinstance(value, list):
            for child in value:
                walk(child, allowed)

    if dimension == "gate":
        walk(data["observability"], set(read_indices))
        walk(data["action_following"], set(range(1, 13)))
    else:
        walk(data, set(read_indices))
    if (
        dimension == "environment"
        and not lighting_style
        and getattr(judgment, "lighting_style", None) is not None
        and judgment.lighting_style.status != "not_applicable"
    ):
        raise ValueError(
            "Disabled lighting_style must be not_applicable with null score"
        )


class Judge:
    def __init__(
        self,
        endpoint,
        model,
        api_key="",
        concurrency=4,
        timeout=240,
        retries=2,
        max_tokens=4096,
        extra_body=None,
        transport=None,
        temperature=0,
        qpm=None,
        tpm=None,
        tokens_per_request=20000,
        token_parameter="max_tokens",
        request_headers=None,
        request_params=None,
        resume=False,
    ):
        if (
            concurrency < 1
            or retries < 0
            or not isfinite(timeout)
            or timeout <= 0
            or max_tokens < 1
        ):
            raise ValueError("Invalid judge concurrency/retries/timeout/max_tokens")
        extra_body = extra_body or {}
        reserved = {
            "model",
            "messages",
            "stream",
            "response_format",
            "temperature",
            "max_tokens",
            "max_completion_tokens",
        }
        if reserved & extra_body.keys():
            raise ValueError(
                f"extra_body cannot override {sorted(reserved & extra_body.keys())}"
            )
        self.endpoint, self.model, self.api_key = endpoint, model, api_key
        self.retries, self.max_tokens, self.extra_body = retries, max_tokens, extra_body
        self.temperature = temperature
        if token_parameter not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("Unsupported output token parameter")
        self.token_parameter = token_parameter
        self.request_headers = dict(request_headers or {})
        self.request_params = dict(request_params or {})
        self.resume = resume
        self.pacer = RequestPacer(qpm, tpm, tokens_per_request)
        self.enforce_token_reservation = tpm is not None
        self.limit = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(timeout, 15)),
            trust_env=False,
            transport=transport,
        )
        self.common = (ROOT / "common.md").read_text()
        self.memory_common = (ROOT / "memory_common.md").read_text()
        self.rubrics = {
            dimension: (ROOT / dimension / "rubric.md").read_text()
            for dimension in SCHEMAS
        }

    async def close(self):
        await self.client.aclose()

    async def ask(
        self,
        dimension,
        context,
        images,
        output: Path,
        *,
        schema_override=None,
        rubric_override=None,
    ):
        schema = schema_override or SCHEMAS[dimension]
        selected = context.get("selected_metrics")
        if (
            schema_override is None
            and selected is not None
            and dimension not in ("gate", "inventory")
        ):
            fields = {
                key.split(".")[1]: (Metric, ...)
                for key in selected
                if key.startswith(dimension + ".")
            }
            schema = create_model(
                f"{dimension.title()}Judgment", __base__=Record, **fields
            )
        common = (
            self.memory_common
            if context.get("memory_frame") is not None
            else self.common
        )
        rubric = (
            rubric_override if rubric_override is not None else self.rubrics[dimension]
        )
        prompt = (
            rubric
            + "\nCASE AND SAMPLING:\n"
            + json.dumps(context, ensure_ascii=False)
            + "\nREQUIRED JSON SCHEMA:\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )
        content = [{"type": "text", "text": prompt}]
        for label, path in zip(
            ("I_gt: GT REFERENCE", "I_gen: GENERATED VIDEO"), images
        ):
            content.extend(
                [
                    {"type": "text", "text": label},
                    {
                        "type": "image_url",
                        "image_url": {
                            "detail": "high",
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(path.read_bytes()).decode(),
                        },
                    },
                ]
            )
        record = {
            "dimension": dimension,
            "model": self.model,
            "common_rubric": common,
            "rubric": rubric,
            "schema": schema.model_json_schema(),
            "context": context,
            "images": [
                {"label": label, "path": str(path)}
                for label, path in zip(("I_gt", "I_gen"), images)
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "token_parameter": self.token_parameter,
            "attempts": [],
        }
        signature = {
            "endpoint": self.endpoint,
            "params": self.request_params,
            "model": self.model,
            "system": common,
            "prompt": prompt,
            "images": [hashlib.sha256(path.read_bytes()).hexdigest() for path in images],
            "max_tokens": self.max_tokens,
            "token_parameter": self.token_parameter,
            "temperature": self.temperature,
            "extra_body": self.extra_body,
        }
        record["request_signature"] = hashlib.sha256(
            json.dumps(signature, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        record["image_sha256"] = signature["images"]
        record_path = output / f"{dimension}.json"
        if self.resume and record_path.exists():
            previous = json.loads(record_path.read_text())
            if previous.get("request_signature") != record["request_signature"]:
                raise ValueError("Saved judge request differs in model, inputs or protocol; use a new output directory")
            record = previous
            for event in reversed(record["attempts"]):
                if event.get("parsed") is not None:
                    judgment = schema.model_validate(event["parsed"])
                    validate_evidence(dimension, judgment, context["sampling"]["read_frame_indices"], context["case"]["lighting_style"])
                    return judgment
        repair = ""
        for attempt in range(self.retries + 1):
            body = {
                "model": self.model,
                self.token_parameter: self.max_tokens,
                "stream": False,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": common},
                    {
                        "role": "user",
                        "content": content
                        + ([{"type": "text", "text": repair}] if repair else []),
                    },
                ],
                **self.extra_body,
            }
            if self.temperature is not None:
                body["temperature"] = self.temperature
            event = {"attempt": len(record["attempts"]) + 1}
            if repair:
                event["validation_repair"] = repair
            retryable = True
            try:
                headers = (
                    {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                )
                headers.update(self.request_headers)
                async with self.limit:
                    await self.pacer.wait()
                    event["started_at"] = time.time()
                    write_json(record_path, {**record, "attempts": record["attempts"] + [event]})
                    response = await self.client.post(
                        self.endpoint, json=body, headers=headers, params=self.request_params
                    )
                event["elapsed_seconds"] = time.time() - event["started_at"]
                event["http_status"] = response.status_code
                if response.status_code != 200:
                    retryable = (
                        response.status_code in (408, 429)
                        or response.status_code >= 500
                    )
                    raise ValueError(f"VLM HTTP {response.status_code}")
                payload = response.json()
                event["response_model"] = payload.get("model")
                event["response_id"] = payload.get("id")
                event["usage"] = payload.get("usage")
                if self.enforce_token_reservation:
                    # This is a configuration failure, not a reason to repeat a charged request.
                    try:
                        self.pacer.observe(event["usage"])
                    except ValueError:
                        retryable = False
                        raise
                choice = payload["choices"][0]
                raw = choice["message"]["content"]
                event["response_text"] = raw
                event["finish_reason"] = choice.get("finish_reason")
                if choice.get("finish_reason") == "length":
                    raise ValueError("Response truncated; increase --max-tokens")
                # Strict JSON is intentional: invalid outputs are retried, never assigned zero.
                judgment = schema.model_validate_json(raw)
                validate_evidence(
                    dimension,
                    judgment,
                    context["sampling"]["read_frame_indices"],
                    context["case"]["lighting_style"],
                )
                event["parsed"] = judgment.model_dump()
                record["attempts"].append(event)
                write_json(output / f"{dimension}.json", record)
                return judgment
            except (
                httpx.HTTPError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
            ) as exc:
                detail = (
                    type(exc).__name__ if isinstance(exc, httpx.HTTPError) else str(exc)
                )
                if self.api_key:
                    detail = detail.replace(self.api_key, "[REDACTED]")
                event["error"] = detail[:2000]
                record["attempts"].append(event)
                write_json(output / f"{dimension}.json", record)
                if not retryable or attempt == self.retries:
                    raise RuntimeError(
                        f"{dimension} judge failed: {event['error']}"
                    ) from None
                repair = (
                    "Previous response failed validation. Return a COMPLETE JSON object with ALL "
                    "required top-level keys: "
                    + ", ".join(schema.model_fields)
                    + ". Include every metric, even when not_applicable or insufficient_evidence; "
                    "do not return only the corrected field. Error: " + detail[:1000]
                )
                await asyncio.sleep(min(2**attempt, 4))
        raise RuntimeError("Unreachable judge state")
