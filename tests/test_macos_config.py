"""Cloud-folder auto-detection on a Mac (and that WSL still wins), against a fake HOME."""
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_config(scratch: Path):
    # Importing core.config creates ./whisper in the repo when no cloud root is
    # set; point it at the scratch dir for the import so the test leaves no trace.
    with mock.patch.dict(os.environ, {"TRANSCRIBE_CLOUD_ROOT": str(scratch)}):
        return importlib.import_module("core.config")


class DetectCloudRootTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()
        self.home = self.tmp / "home" / "me"
        self.home.mkdir(parents=True)
        self.config = _load_config(self.tmp / "unused-root")
        # An empty WSL root: nothing under it can match.
        self.wsl = [str(self.tmp / "mnt" / "c" / "Users" / "*" / "My Drive" / "whisper")]

    def detect(self, wsl_globs=None):
        return self.config.detect_cloud_root(
            home=self.home, wsl_globs=self.wsl if wsl_globs is None else wsl_globs)

    def mkdir(self, rel: str) -> Path:
        p = self.home / rel
        p.mkdir(parents=True)
        return p

    def test_nothing_found(self):
        self.assertIsNone(self.detect())

    def test_icloud(self):
        p = self.mkdir("Library/Mobile Documents/com~apple~CloudDocs/whisper")
        self.assertEqual(self.detect(), p)

    def test_old_google_drive_client(self):
        p = self.mkdir("Google Drive/My Drive/whisper")
        self.assertEqual(self.detect(), p)

    def test_google_drive_for_desktop(self):
        p = self.mkdir("Library/CloudStorage/GoogleDrive-me@example.com/My Drive/whisper")
        self.assertEqual(self.detect(), p)

    def test_google_beats_older_google_beats_icloud(self):
        icloud = self.mkdir("Library/Mobile Documents/com~apple~CloudDocs/whisper")
        old = self.mkdir("Google Drive/My Drive/whisper")
        new = self.mkdir("Library/CloudStorage/GoogleDrive-b@example.com/My Drive/whisper")
        self.assertEqual(self.detect(), new)
        new.rmdir()
        self.assertEqual(self.detect(), old)
        old.rmdir()
        self.assertEqual(self.detect(), icloud)

    def test_multiple_google_accounts_pick_sorted_first(self):
        self.mkdir("Library/CloudStorage/GoogleDrive-z@example.com/My Drive/whisper")
        a = self.mkdir("Library/CloudStorage/GoogleDrive-a@example.com/My Drive/whisper")
        self.assertEqual(self.detect(), a)

    def test_account_without_whisper_folder_is_skipped(self):
        (self.home / "Library/CloudStorage/GoogleDrive-a@example.com/My Drive").mkdir(parents=True)
        b = self.mkdir("Library/CloudStorage/GoogleDrive-b@example.com/My Drive/whisper")
        self.assertEqual(self.detect(), b)

    def test_a_file_named_whisper_is_not_a_folder(self):
        d = self.home / "Google Drive" / "My Drive"
        d.mkdir(parents=True)
        (d / "whisper").write_text("not a dir")
        self.assertIsNone(self.detect())

    def test_wsl_glob_wins_over_mac_folders(self):
        self.mkdir("Library/Mobile Documents/com~apple~CloudDocs/whisper")
        wsl = self.tmp / "mnt" / "c" / "Users" / "me" / "My Drive" / "whisper"
        wsl.mkdir(parents=True)
        self.assertEqual(self.detect(), wsl)

    def test_default_home_is_path_home(self):
        p = self.mkdir("Google Drive/My Drive/whisper")
        with mock.patch.object(Path, "home", return_value=self.home):
            self.assertEqual(
                self.config.detect_cloud_root(wsl_globs=self.wsl), p)

    def test_unresolvable_home_does_not_crash(self):
        with mock.patch.object(Path, "home", side_effect=RuntimeError("no home")):
            self.assertIsNone(self.config.detect_cloud_root(wsl_globs=self.wsl))

    def test_real_wsl_globs_come_first(self):
        self.assertTrue(self.config.CLOUD_GLOBS[0].startswith("/mnt/c/"))
        self.assertEqual(
            self.config.HOME_CLOUD_GLOBS[0],
            "Library/CloudStorage/GoogleDrive-*/My Drive/whisper")


if __name__ == "__main__":
    unittest.main()
