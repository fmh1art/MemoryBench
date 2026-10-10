"""API configuration tests requiring only the Python standard library."""

import argparse
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.api_config import add_api_arguments, resolve_api_arguments


class ApiConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = Path(self.directory.name) / "api.toml"
        self.config.write_text(
            '[api]\nendpoint="http://localhost:9000/v1/chat/completions"\n'
            'model="local-qwen"\napi_key_env="MACHINE_KEY"\n'
            '[api.extra_body.chat_template_kwargs]\nenable_thinking=false\n'
        )
        self.parser = argparse.ArgumentParser()
        add_api_arguments(self.parser)

    def resolve(self, *arguments):
        return resolve_api_arguments(self.parser.parse_args(arguments))

    def test_local_config_and_thinking(self):
        args = self.resolve("--api-config", str(self.config))
        self.assertEqual(args.endpoint, "http://localhost:9000/v1/chat/completions")
        self.assertEqual(args.model, "local-qwen")
        self.assertEqual(args.api_key_env, "MACHINE_KEY")
        self.assertEqual(args.api_extra_body, {"chat_template_kwargs": {"enable_thinking": False}})

    def test_cli_overrides_local_config(self):
        args = self.resolve("--api-config", str(self.config), "--endpoint", "http://other/v1/chat/completions", "--model", "other", "--api-key-env", "OTHER_KEY")
        self.assertEqual((args.endpoint, args.model, args.api_key_env), ("http://other/v1/chat/completions", "other", "OTHER_KEY"))

    def test_environment_config(self):
        with patch.dict(os.environ, {"MEMORYBENCH_API_CONFIG": str(self.config)}):
            self.assertEqual(self.resolve().model, "local-qwen")

    def test_legacy_cli_without_config(self):
        with patch("evaluation.api_config.DEFAULT_CONFIG", Path(self.directory.name) / "missing"), patch.dict(os.environ, {}, clear=True):
            args = self.resolve("--endpoint", "http://old", "--model", "old")
            self.assertEqual(args.api_extra_body, {})
            self.assertEqual(args.api_key_env, "MEM_BENCH_API_KEY")

    def test_missing_explicit_config_fails(self):
        with self.assertRaises(FileNotFoundError):
            self.resolve("--api-config", str(self.config.with_name("missing")))

    def test_incomplete_config_fails(self):
        self.config.write_text('[api]\nmodel="qwen"\n')
        with self.assertRaisesRegex(ValueError, "endpoint"):
            self.resolve("--api-config", str(self.config))

    def test_plaintext_key_rejected(self):
        self.config.write_text('[api]\napi_key="secret"\n')
        with self.assertRaisesRegex(ValueError, "Unknown API config fields"):
            self.resolve("--api-config", str(self.config))


if __name__ == "__main__":
    unittest.main()
