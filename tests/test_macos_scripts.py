"""The shell scripts must stay portable to macOS (bash 3.2, BSD userland).

Real macOS cannot be run here, so these check the things that break there:
syntax, no GNU-only commands, and the portable replacements actually working
(the lock that replaced flock, the size helper, certs made through config files).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [ROOT / "setup.sh", ROOT / "make-cert.sh", ROOT / "batch" / "watcher.sh"]
BASH = shutil.which("bash")


def code_lines(path: Path):
    """Script lines without comments, for grepping constructs."""
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield n, line


@unittest.skipUnless(BASH, "bash not available")
class ScriptSyntaxTest(unittest.TestCase):
    def test_bash_n(self):
        for script in SCRIPTS:
            with self.subTest(script=script.name):
                r = subprocess.run([BASH, "-n", str(script)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)

    def test_env_bash_shebang(self):
        for script in SCRIPTS:
            with self.subTest(script=script.name):
                self.assertEqual(script.read_text().splitlines()[0], "#!/usr/bin/env bash")

    def test_no_gnu_or_bash4_only_constructs(self):
        banned = {
            r"\bflock\b": "flock (not on macOS)",
            r"stat\s+-c": "stat -c (GNU only)",
            r"readlink\s+-f": "readlink -f (GNU only)",
            r"date\s+-d\b": "date -d (GNU only)",
            r"%N": "date +%N (GNU only)",
            r"\bmapfile\b|\breadarray\b": "mapfile/readarray (bash 4)",
            r"declare\s+-A|local\s+-A": "associative arrays (bash 4)",
            r"local\s+-n|declare\s+-n": "namerefs (bash 4.3)",
            r"\$\{[A-Za-z_]+(,,|\^\^)\}": "case-modifying expansion (bash 4)",
            r"\[\[\s+-v\s": "[[ -v ]] (bash 4.2)",
            r"\|&": "|& (bash 4)",
            r"&>>": "&>> (bash 4)",
            r"\bwait\s+-n\b": "wait -n (bash 4.3)",
            r"\bsed\s+(-[A-Za-z]*\s+)*-i\b": "sed -i (differs on BSD)",
            r"-addext": "openssl -addext (missing in older LibreSSL)",
            r"-ext\s+subjectAltName": "openssl x509 -ext (OpenSSL only)",
        }
        for script in SCRIPTS:
            for n, line in code_lines(script):
                for pattern, why in banned.items():
                    with self.subTest(script=script.name, line=n, why=why):
                        self.assertIsNone(re.search(pattern, line), f"{script.name}:{n}: {why}")

    def test_hostname_dash_I_is_only_one_fallback(self):
        # `hostname -I` is fine as the first attempt; it must not be the only one.
        text = (ROOT / "make-cert.sh").read_text()
        self.assertIn("ipconfig getifaddr", text)
        self.assertIn("ifconfig", text)


def watcher_functions(*names):
    src = (ROOT / "batch" / "watcher.sh").read_text(encoding="utf-8")
    out = []
    for name in names:
        m = re.search(rf"^{name}\(\) \{{.*?^\}}|^{name}\(\) \{{[^\n]*\}}$", src, re.M | re.S)
        assert m, f"{name} not found in watcher.sh"
        out.append(m.group(0))
    return "\n".join(out)


@unittest.skipUnless(BASH, "bash not available")
class WatcherHelpersTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.lock = self.tmp / "lock.d"

    def sh(self, body, **env):
        funcs = watcher_functions("log", "file_size", "lock_holder_alive",
                                  "acquire_lock", "release_lock")
        script = f'LOCKDIR="{self.lock}"; CHILD=""\n{funcs}\n{body}\n'
        return subprocess.run([BASH, "-c", script], capture_output=True, text=True,
                              env={**os.environ, **env}, timeout=30)

    def spawn(self, argv0):
        p = subprocess.Popen([BASH, "-c", "sleep 60; sleep 0", argv0])
        self.addCleanup(lambda: (p.kill(), p.wait()))
        return p

    def test_file_size(self):
        f = self.tmp / "a.bin"
        f.write_bytes(b"x" * 12345)
        r = self.sh(f'file_size "{f}"; file_size "{self.tmp}/nope"')
        self.assertEqual(r.stdout.split(), ["12345", "0"])

    def test_lock_acquire_and_release(self):
        r = self.sh('acquire_lock && cat "$LOCKDIR/pid" && release_lock; '
                    f'[ -d "{self.lock}" ] && echo LEFT || echo GONE')
        lines = r.stdout.split()
        self.assertTrue(lines[0].isdigit(), r.stdout + r.stderr)
        self.assertEqual(lines[-1], "GONE")

    def test_second_watcher_is_refused(self):
        live = self.spawn("watcher.sh")
        self.lock.mkdir()
        (self.lock / "pid").write_text(str(live.pid))
        r = self.sh("acquire_lock; echo rc=$?")
        self.assertIn("rc=1", r.stdout)
        self.assertIn("Another watcher", r.stdout)
        self.assertEqual((self.lock / "pid").read_text(), str(live.pid))

    def test_dead_owner_is_taken_over(self):
        dead = subprocess.Popen([BASH, "-c", "exit 0"])
        dead.wait()
        self.lock.mkdir()
        (self.lock / "pid").write_text(str(dead.pid))
        r = self.sh('acquire_lock; echo rc=$?; cat "$LOCKDIR/pid"')
        self.assertIn("rc=0", r.stdout)
        self.assertIn("stale", r.stdout)

    def test_reused_pid_is_not_a_watcher(self):
        other = self.spawn("something-else")
        self.lock.mkdir()
        (self.lock / "pid").write_text(str(other.pid))
        r = self.sh("acquire_lock; echo rc=$?")
        self.assertIn("rc=0", r.stdout)

    def test_release_leaves_someone_elses_lock(self):
        self.lock.mkdir()
        (self.lock / "pid").write_text("999999")
        self.sh("release_lock")
        self.assertTrue(self.lock.is_dir())


@unittest.skipUnless(BASH and shutil.which("openssl"), "bash/openssl not available")
class MakeCertTest(unittest.TestCase):
    def test_issues_valid_certs_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            repo = tmp / "repo"
            repo.mkdir()
            shutil.copy(ROOT / "make-cert.sh", repo / "make-cert.sh")
            shim = tmp / "shim"
            shim.mkdir()
            hostname = shim / "hostname"
            hostname.write_text('#!/bin/sh\necho "192.168.1.10 10.0.0.5 fe80::1"\n')
            hostname.chmod(0o755)
            scratch = tmp / "scratch"
            scratch.mkdir()
            env = {**os.environ, "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}",
                   "TMPDIR": str(scratch)}

            def run():
                return subprocess.run([BASH, str(repo / "make-cert.sh")], env=env,
                                      capture_output=True, text=True, timeout=120)

            r = run()
            self.assertEqual(r.returncode, 0, r.stderr)
            certs = repo / "certs"
            v = subprocess.run(["openssl", "verify", "-CAfile", str(certs / "ca.pem"),
                                str(certs / "cert.pem")], capture_output=True, text=True)
            self.assertIn("OK", v.stdout)
            text = subprocess.run(["openssl", "x509", "-in", str(certs / "cert.pem"),
                                   "-noout", "-text"], capture_output=True, text=True).stdout
            san = text.split("Subject Alternative Name")[1].splitlines()[1]
            for want in ("DNS:localhost", "IP Address:127.0.0.1",
                         "IP Address:192.168.1.10", "IP Address:10.0.0.5"):
                self.assertIn(want, san)
            self.assertIn("TLS Web Server Authentication", text)
            self.assertIn("CA:FALSE", text)
            ca = subprocess.run(["openssl", "x509", "-in", str(certs / "ca.pem"),
                                 "-noout", "-text"], capture_output=True, text=True).stdout
            self.assertIn("CA:TRUE", ca)
            self.assertEqual(list(scratch.iterdir()), [], "temp files left behind")

            serial = (certs / "cert.pem").read_bytes()
            self.assertEqual(run().returncode, 0)
            self.assertEqual((certs / "cert.pem").read_bytes(), serial, "re-run reissued a good cert")


if __name__ == "__main__":
    unittest.main()
