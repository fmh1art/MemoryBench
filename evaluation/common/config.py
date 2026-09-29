import os
import tomllib
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class APIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endpoint: str = "http://127.0.0.1:8000/v1/chat/completions"
    model: str = "PLEASE_FILL_MODEL"
    api_key: str = ""
    api_key_env: str = "MEM_BENCH_API_KEY"
    timeout_seconds: float = Field(default=180, gt=0)
    retries: int = Field(default=2, ge=0, le=5)
    max_tokens: int = Field(default=8192, ge=256)
    temperature: float = Field(default=0, ge=0, le=2)
    json_mode: bool = True
    trust_env: bool = False
    # Example: {"chat_template_kwargs": {"enable_thinking": false}}
    extra_body: dict = Field(default_factory=dict)
    max_images_per_query: int = Field(default=8, ge=1, le=8)

    def key(self):
        return os.getenv(self.api_key_env, "") or self.api_key


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frames_per_video: int = Field(default=8, ge=1, le=8)
    action_frames: int = Field(default=6, ge=1, le=8)
    image_max_side: int = Field(default=768, ge=256, le=2048)


class WeightConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inventory: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    appearance: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    spatial: float = Field(default=1.0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def nonzero(self):
        if not any(self.model_dump().values()):
            raise ValueError("At least one metric weight must be positive")
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api: APIConfig = Field(default_factory=APIConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    weights: WeightConfig = Field(default_factory=WeightConfig)

    def public(self):
        data = self.model_dump()
        data["api"]["api_key"] = "[REDACTED]"
        # Extra body may contain deployment-specific credentials; persist only its keys.
        data["api"]["extra_body"] = {"configured_keys": sorted(self.api.extra_body)}
        return data


def load_config(path: Path):
    with path.open("rb") as f:
        config = Config.model_validate(tomllib.load(f))
    url = urlparse(config.api.endpoint)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username
        or url.password
        or url.query
    ):
        raise ValueError(
            "api.endpoint must be an HTTP(S) URL without credentials or query parameters"
        )
    if not config.api.model or config.api.model == "PLEASE_FILL_MODEL":
        raise ValueError("请先在 configs/local.toml 中填写 api.model 和 api.endpoint")
    reserved = {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "response_format",
        "stream",
    }
    if reserved.intersection(config.api.extra_body):
        raise ValueError(
            "api.extra_body must not override protocol/model/message fields"
        )
    if config.sampling.action_frames > config.api.max_images_per_query:
        raise ValueError("action_frames exceeds the per-query image limit")
    return config
