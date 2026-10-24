"""Device choice and the GPU lease on a machine with no NVIDIA GPU. No GPU, no network."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core import device, gpu


def pick(cuda: bool, **env) -> device.Choice:
    """pick() as if torch saw (or did not see) a GPU, with only `env` set."""
    clean = {k: v for k, v in os.environ.items()
             if k not in ("SUPERMEET_DEVICE", "SUPERMEET_WHISPER_MODEL")}
    with mock.patch.dict(os.environ, {**clean, **env}, clear=True), \
            mock.patch.object(device, "cuda_available", return_value=cuda):
        return device.pick()


class PickTest(unittest.TestCase):
    def test_auto_with_gpu_is_the_original_setup(self):
        c = pick(True)
        self.assertEqual((c.device, c.compute_type, c.model), ("cuda", "int8", "large-v3"))
        self.assertTrue(c.is_cuda)

    def test_auto_without_gpu_is_cpu_turbo(self):
        c = pick(False)
        self.assertEqual((c.device, c.compute_type, c.model),
                         ("cpu", "int8", "large-v3-turbo"))
        self.assertFalse(c.is_cuda)

    def test_forced_cuda_ignores_detection(self):
        self.assertEqual(pick(False, SUPERMEET_DEVICE="cuda").device, "cuda")

    def test_forced_cpu_on_a_gpu_machine(self):
        c = pick(True, SUPERMEET_DEVICE="cpu")
        self.assertEqual((c.device, c.model), ("cpu", "large-v3-turbo"))

    def test_device_value_is_case_and_space_insensitive(self):
        self.assertEqual(pick(True, SUPERMEET_DEVICE=" CPU ").device, "cpu")

    def test_model_override_wins_on_both(self):
        self.assertEqual(pick(True, SUPERMEET_WHISPER_MODEL="small").model, "small")
        self.assertEqual(pick(False, SUPERMEET_WHISPER_MODEL="small").model, "small")

    def test_invalid_device_exits(self):
        with self.assertRaises(SystemExit):
            pick(True, SUPERMEET_DEVICE="mps")


class LeaseWithoutGpuTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.lock = Path(tmp.name) / "gpu.lock"
        for p in (mock.patch.object(gpu, "LOCK_PATH", self.lock),
                  mock.patch.object(device, "cuda_available", return_value=False)):
            p.start()
            self.addCleanup(p.stop)

    def test_acquire_is_a_noop_that_never_touches_the_lockfile(self):
        with mock.patch.object(gpu, "_flock", side_effect=AssertionError("flocked")):
            handle = gpu.acquire("batch", wait=True, timeout=1)
        self.assertFalse(handle.held)
        self.assertFalse(self.lock.exists())
        handle.release()   # harmless

    def test_two_holders_do_not_block_each_other(self):
        a = gpu.acquire("realtime", wait=False)
        b = gpu.acquire("batch", wait=False)      # would raise GPUBusy on a real card
        self.assertFalse(a.held or b.held)

    def test_scoped_lease_yields_and_exits(self):
        with gpu.lease("kuliah", wait=False) as handle:
            self.assertEqual(handle.name, "kuliah")
        self.assertIsNone(gpu.holder())

    def test_with_a_gpu_the_second_taker_is_still_refused(self):
        with mock.patch.object(device, "cuda_available", return_value=True):
            first = gpu.acquire("realtime", wait=False)
            try:
                self.assertTrue(first.held)
                with self.assertRaises(gpu.GPUBusy):
                    gpu.acquire("batch", wait=False)
            finally:
                first.release()


class DescribeTest(unittest.TestCase):
    def test_lines(self):
        self.assertEqual(gpu.describe(device.Choice("cuda", "int8", "large-v3")),
                         "large-v3 on cuda (int8)")
        with mock.patch.dict(os.environ, {"SUPERMEET_DEVICE": "auto"}):
            self.assertEqual(gpu.describe(device.Choice("cpu", "int8", "large-v3-turbo")),
                             "large-v3-turbo on cpu (tidak ada GPU NVIDIA)")
        with mock.patch.dict(os.environ, {"SUPERMEET_DEVICE": "cpu"}):
            self.assertIn("dipaksa", gpu.describe(device.Choice("cpu", "int8", "large-v3-turbo")))


if __name__ == "__main__":
    unittest.main()
