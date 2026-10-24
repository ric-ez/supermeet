"""macOS-specific helpers in supermeet.py, exercised on any OS.

supermeet.py re-execs itself at import time (into .venv, and on a Mac with a
library path), so it cannot be imported in a test. The helper functions are
pulled out of its source and run in a namespace instead.
"""
import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parent.parent / "supermeet.py"


def load(*names):
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in wanted} == set(names), "function missing from supermeet.py"
    ns = {"os": os, "sys": mock.MagicMock(), "subprocess": subprocess}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(SRC), "exec"), ns)
    return ns


class DyldFallbackTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.prefix = Path(self._tmp.name) / "brew"
        (self.prefix / "lib").mkdir(parents=True)
        self.ns = load("_dyld_fallback")

    def run_fn(self, platform="darwin", env=None):
        env = {"HOMEBREW_PREFIX": str(self.prefix), **(env or {})}
        self.ns["sys"].platform = platform
        with mock.patch.dict(os.environ, env, clear=True):
            return self.ns["_dyld_fallback"]()

    def test_apple_silicon_style_prefix(self):
        value = self.run_fn()
        parts = value.split(":")
        self.assertEqual(parts[0], f"{self.prefix}/lib")
        # dyld's own defaults must survive, because the variable replaces them.
        self.assertIn("/usr/lib", parts)
        self.assertIn("/usr/local/lib", parts)

    def test_not_on_linux(self):
        self.assertIsNone(self.run_fn(platform="linux"))

    def test_respects_existing_value(self):
        self.assertIsNone(self.run_fn(env={"DYLD_FALLBACK_LIBRARY_PATH": "/x"}))

    def test_intel_prefix_needs_nothing(self):
        # /usr/local/lib is already in dyld's default fallback path.
        self.assertIsNone(self.run_fn(env={"HOMEBREW_PREFIX": "/usr/local"}))

    def test_missing_lib_dir(self):
        self.assertIsNone(self.run_fn(env={"HOMEBREW_PREFIX": str(self.prefix / "nope")}))


class LanIpsTest(unittest.TestCase):
    def run_fn(self, outputs, platform="linux", socket_ip=None):
        ns = load("lan_ips")
        ns["sys"] = mock.MagicMock(platform=platform)
        ns["_run_out"] = lambda *cmd: outputs.get(cmd, "")
        if socket_ip is None:
            return ns["lan_ips"]()
        with mock.patch("socket.socket") as sock:
            sock.return_value.__enter__.return_value.getsockname.return_value = (socket_ip, 0)
            return ns["lan_ips"]()

    def test_linux_hostname_i(self):
        out = self.run_fn({("hostname", "-I"): "192.168.1.10 10.0.0.5 fe80::1 \n"})
        self.assertEqual(out, ["192.168.1.10", "10.0.0.5"])

    def test_filters_loopback_linklocal_and_dupes(self):
        out = self.run_fn({("hostname", "-I"): "127.0.0.1 169.254.1.1 10.1.1.1 10.1.1.1"})
        self.assertEqual(out, ["10.1.1.1"])

    def test_mac_uses_ipconfig(self):
        out = self.run_fn({("ipconfig", "getifaddr", "en0"): "192.168.0.7\n",
                           ("ipconfig", "getifaddr", "en5"): "10.9.9.9\n"},
                          platform="darwin")
        self.assertEqual(out, ["192.168.0.7", "10.9.9.9"])

    def test_ifconfig_fallback(self):
        ifconfig = ("lo0: flags=8049<UP>\n\tinet 127.0.0.1 netmask 0xff000000\n"
                    "\tinet6 ::1 prefixlen 128\nen0: flags=8863<UP>\n"
                    "\tinet6 fe80::1%en0 prefixlen 64\n"
                    "\tinet 192.168.50.2 netmask 0xffffff00 broadcast 192.168.50.255\n")
        self.assertEqual(self.run_fn({("ifconfig",): ifconfig}, platform="darwin"),
                         ["192.168.50.2"])

    def test_old_linux_ifconfig_addr_prefix(self):
        self.assertEqual(self.run_fn({("ifconfig",): "  inet addr:172.20.0.4  Bcast:x\n"}),
                         ["172.20.0.4"])

    def test_socket_last_resort(self):
        self.assertEqual(self.run_fn({}, socket_ip="192.168.9.9"), ["192.168.9.9"])

    def test_nothing_found(self):
        self.assertEqual(self.run_fn({}, socket_ip="0.0.0.0"), [])


if __name__ == "__main__":
    unittest.main()
