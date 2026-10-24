import unittest
from unittest import mock

from core import doctor


class MaskTest(unittest.TestCase):
    def test_shows_only_last_four(self):
        masked = doctor.mask("AIzaSyABCDEFGHIJ1234")
        self.assertEqual(masked, "…1234")

    def test_short_secret_fully_hidden(self):
        self.assertEqual(doctor.mask("abc"), "…")


class DefaultModelTest(unittest.TestCase):
    def test_env_override(self):
        with mock.patch.dict("os.environ", {"SUPERMEET_GEMINI_MODEL": "x-model"}):
            self.assertEqual(doctor.default_model(), "x-model")

    def test_matches_pipeline_default(self):
        with mock.patch.dict("os.environ", {}):
            import os
            os.environ.pop("SUPERMEET_GEMINI_MODEL", None)
            src = (doctor.ROOT / "batch" / "pipeline.py").read_text(encoding="utf-8")
            self.assertIn(f'"{doctor.default_model()}")', src)


class GeminiOnlineTest(unittest.TestCase):
    def call_with(self, code):
        from google.genai import errors
        exc = errors.ClientError(code, {"error": {"message": "boom"}})
        with mock.patch("google.genai.Client") as client:
            client.return_value.models.generate_content.side_effect = exc
            return doctor.check_gemini_online("k" * 12, "some-model")

    def test_error_codes_mapped(self):
        try:
            import google.genai  # noqa: F401
        except ImportError:
            self.skipTest("google-genai not installed")
        self.assertEqual(self.call_with(403)[0], doctor.FAIL)
        self.assertIn("key salah", self.call_with(400)[1])
        self.assertIn("model nggak ada", self.call_with(404)[1])
        icon, msg = self.call_with(429)
        self.assertEqual(icon, doctor.WARN)
        self.assertIn("limit", msg)


if __name__ == "__main__":
    unittest.main()
