"""Machine-local API settings shared by evaluation and null review CLIs."""

import os
from pathlib import Path
import tomllib

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "api.local.toml"


def add_api_arguments(parser):
    parser.add_argument(
        "--api-config", type=Path,
        help="API TOML file; default: MEMORYBENCH_API_CONFIG or configs/api.local.toml",
    )
    parser.add_argument("--endpoint", help="Full compatible /v1/chat/completions URL")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", help="Environment variable containing the API key")


def resolve_api_arguments(args):
    """Explicit CLI values override local settings. Never store keys in TOML."""
    configured_path = args.api_config or os.environ.get("MEMORYBENCH_API_CONFIG")
    path = Path(configured_path) if configured_path else DEFAULT_CONFIG
    settings = {}
    if configured_path or path.exists():
        with path.open("rb") as source:
            settings = tomllib.load(source).get("api", {})
        if not isinstance(settings, dict):
            raise ValueError("API config must contain an [api] table")
        unknown = settings.keys() - {"endpoint", "model", "api_key_env", "extra_body"}
        if unknown:
            raise ValueError(f"Unknown API config fields: {sorted(unknown)}")
    for name in ("endpoint", "model", "api_key_env"):
        value = getattr(args, name, None) or settings.get(name)
        if name == "api_key_env" and value is None:
            value = "MEM_BENCH_API_KEY"
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Set --{name.replace('_', '-')} or api.{name} in {path}")
        setattr(args, name, value)
    extra_body = settings.get("extra_body", {})
    if not isinstance(extra_body, dict):
        raise ValueError("api.extra_body must be a table")
    args.api_extra_body = extra_body
    return args
