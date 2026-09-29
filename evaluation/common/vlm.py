import base64
import json
import re
import time
from pathlib import Path

import httpx


class EvaluationError(RuntimeError):
    pass


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


class VLM:
    def __init__(self, config, output, transport=None):
        self.config = config
        self.output = Path(output) / "vlm_calls"
        self.output.mkdir(parents=True, exist_ok=True)
        snapshot = Path(output) / "rubrics"
        self.rubric_dir = (
            snapshot
            if snapshot.is_dir()
            else Path(__file__).resolve().parents[1] / "prompt"
        )
        self.common = (self.rubric_dir / "common.md").read_text()
        self.client = httpx.Client(
            timeout=config.timeout_seconds,
            trust_env=config.trust_env,
            transport=transport,
        )
        self.calls = []

    def close(self):
        self.client.close()

    def ask(self, task, data, images, schema, validator=None):
        if len(images) > min(8, self.config.max_images_per_query):
            raise ValueError(
                "A VLM query may contain at most 8 images (or the lower configured limit)"
            )
        number = len(self.calls) + 1
        rubric = (self.rubric_dir / f"{task}.md").read_text()
        prompt = rubric + "\nINPUT:\n" + json.dumps(data, ensure_ascii=False)
        prompt += "\nREQUIRED JSON SCHEMA:\n" + json.dumps(
            schema.model_json_schema(), ensure_ascii=False
        )
        content = [{"type": "text", "text": prompt}]
        image_records = []
        for label, path in images:
            path = Path(path)
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"
                        + base64.b64encode(path.read_bytes()).decode("ascii")
                    },
                }
            )
            image_records.append({"label": label, "path": str(path)})
        record = {
            "call": number,
            "task": task,
            "input": data,
            "images": image_records,
            "rubric": rubric,
            "common_rubric": self.common,
            "response_schema": schema.model_json_schema(),
            "attempts": [],
        }
        self.calls.append(record)
        path = self.output / f"{number:04d}_{task}.json"
        repair = ""
        for attempt in range(self.config.retries + 1):
            body = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": self.common},
                    {
                        "role": "user",
                        "content": content
                        + ([{"type": "text", "text": repair}] if repair else []),
                    },
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": False,
                **self.config.extra_body,
            }
            if self.config.json_mode:
                body["response_format"] = {"type": "json_object"}
            started = time.monotonic()
            result = {"attempt": attempt + 1}
            try:
                headers = (
                    {"Authorization": f"Bearer {self.config.key()}"}
                    if self.config.key()
                    else {}
                )
                response = self.client.post(
                    self.config.endpoint, json=body, headers=headers
                )
                result["http_status"] = response.status_code
                if response.status_code != 200:
                    # Do not persist service error bodies, which may echo credentials/requests.
                    raise EvaluationError(f"VLM HTTP {response.status_code}")
                payload = response.json()
                choice = payload["choices"][0]
                result["usage"] = payload.get("usage")
                result["finish_reason"] = choice.get("finish_reason")
                raw = choice["message"]["content"]
                result["response_text"] = raw
                if not isinstance(raw, str):
                    raise ValueError(
                        "Expected a JSON string in choices[0].message.content"
                    )
                if choice.get("finish_reason") == "length":
                    raise ValueError("VLM response truncated; increase api.max_tokens")
                stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
                parsed = schema.model_validate(json.loads(stripped))
                if validator:
                    validator(parsed)
                result["parsed"] = parsed.model_dump()
                result["elapsed_seconds"] = time.monotonic() - started
                record["attempts"].append(result)
                write_json(path, record)
                return parsed
            except (
                httpx.HTTPError,
                EvaluationError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
            ) as exc:
                # Validation messages contain no API headers or base64 images.
                detail = (
                    str(exc)
                    if not isinstance(exc, httpx.HTTPError)
                    else type(exc).__name__
                )
                if self.config.key():
                    detail = detail.replace(self.config.key(), "[REDACTED]")
                result["error"] = detail[:3000]
                result["elapsed_seconds"] = time.monotonic() - started
                record["attempts"].append(result)
                write_json(path, record)
                retryable = (
                    result.get("http_status", 200) in (200, 408, 429)
                    or result.get("http_status", 0) >= 500
                )
                if attempt == self.config.retries or not retryable:
                    raise EvaluationError(f'{task} failed: {result["error"]}') from exc
                repair = (
                    "Previous response failed validation. Correct the JSON, preserving the rubric. Error: "
                    + detail[:1800]
                )
                if "http_status" not in result or result.get("http_status") != 200:
                    time.sleep(min(2**attempt, 4))
        raise EvaluationError("Unreachable retry state")
