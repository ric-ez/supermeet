"""The doctor's device line and macOS checks, with a fake Homebrew and no torch."""
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core import doctor


class DyldStatusTest(unittest.TestCase):
    def test_intel_prefix_needs_nothing(self):
        self.assertEqual(doctor.dyld_status("/usr/local", None)[0], doctor.OK)

    def test_apple_silicon_unset_warns_with_export_hint(self):
        icon, detail, hint = doctor.dyld_status("/opt/homebrew", None)
        self.assertEqual(icon, doctor.WARN)
        self.assertIn('export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib"', hint)

    def test_apple_silicon_set(self):
        icon, _, _ = doctor.dyld_status("/opt/homebrew", "/opt/homebrew/lib:/usr/lib")
        self.assertEqual(icon, doctor.OK)

    def test_set_to_something_else_warns(self):
        icon, _, _ = doctor.dyld_status("/opt/homebrew", "/somewhere/else")
        self.assertEqual(icon, doctor.WARN)


class BrewTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.prefix = Path(self._tmp.name)
        (self.prefix / "lib").mkdir()

    def test_prefix_from_env(self):
        with mock.patch.dict(os.environ, {"HOMEBREW_PREFIX": str(self.prefix)}):
            self.assertEqual(doctor.brew_prefix(), str(self.prefix))

    def test_pango_detected_by_dylib(self):
        self.assertFalse(doctor.pango_installed(str(self.prefix)))
        (self.prefix / "lib" / "libpango-1.0.0.dylib").write_bytes(b"")
        self.assertTrue(doctor.pango_installed(str(self.prefix)))


class DeviceChoiceTest(unittest.TestCase):
    def pick(self, usable, **env):
        base = {k: v for k, v in os.environ.items()
                if k not in ("SUPERMEET_DEVICE", "SUPERMEET_WHISPER_MODEL")}
        with mock.patch.dict(os.environ, {**base, **env}, clear=True):
            return doctor.device_choice(usable)

    def test_no_gpu_means_cpu_turbo(self):
        c = self.pick(False)
        self.assertEqual((c.device, c.model), ("cpu", "large-v3-turbo"))

    def test_gpu_means_cuda_large(self):
        c = self.pick(True)
        self.assertEqual((c.device, c.model), ("cuda", "large-v3"))

    def test_forced_cpu_on_a_gpu_box(self):
        self.assertEqual(self.pick(True, SUPERMEET_DEVICE="cpu").device, "cpu")

    def test_real_cuda_available_is_restored(self):
        from core import device
        before = device.cuda_available
        self.pick(False)
        self.assertIs(device.cuda_available, before)


class MainOutputTest(unittest.TestCase):
    def run_doctor(self, cuda, macos, brew_prefix=None, dyld=None):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        env = {"TRANSCRIBE_CLOUD_ROOT": tmp}       # keeps core.config from creating ./whisper
        if brew_prefix:
            env["HOMEBREW_PREFIX"] = brew_prefix
        if dyld:
            env["DYLD_FALLBACK_LIBRARY_PATH"] = dyld
        out = io.StringIO()
        with mock.patch.dict(os.environ, env), \
             mock.patch.object(doctor, "MACOS", macos), \
             mock.patch.object(doctor, "has_module", return_value=True), \
             mock.patch.object(doctor, "cuda_status", return_value=cuda), \
             contextlib.redirect_stdout(out):
            doctor.main([])
        return out.getvalue()

    def test_cpu_is_info_not_a_warning(self):
        text = self.run_doctor((False, doctor.NO_CUDA), macos=False)
        line = next(l for l in text.splitlines() if l.startswith(doctor.INFO))
        self.assertIn("CPU (large-v3-turbo)", line)
        self.assertNotIn("belum ada fallback CPU", text)

    def test_torch_missing_is_the_only_device_warning(self):
        text = self.run_doctor(None, macos=False)
        self.assertIn(f"{doctor.WARN} device", text)

    def test_broken_torch_warns(self):
        text = self.run_doctor((False, "torch gagal di-import"), macos=False)
        self.assertIn(f"{doctor.WARN} device", text)

    def test_mac_checks_with_fake_homebrew(self):
        with tempfile.TemporaryDirectory() as brew:
            (Path(brew) / "lib").mkdir()
            text = self.run_doctor((False, doctor.NO_CUDA), macos=True, brew_prefix=brew)
        self.assertIn("Homebrew", text)
        self.assertIn("brew install pango", text)
        self.assertIn("DYLD_FALLBACK_LIB", text)
        self.assertIn(f'export DYLD_FALLBACK_LIBRARY_PATH="{brew}/lib"', text)


if __name__ == "__main__":
    unittest.main()
