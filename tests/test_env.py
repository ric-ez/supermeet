import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core import env


class ParseTest(unittest.TestCase):
    def test_plain_pairs(self):
        self.assertEqual(env.parse("A=1\nB=two"), {"A": "1", "B": "two"})

    def test_comments_and_blank_lines_skipped(self):
        text = "# header\n\nA=1\n   # indented comment\n"
        self.assertEqual(env.parse(text), {"A": "1"})

    def test_export_prefix(self):
        self.assertEqual(env.parse("export A=1\nexport   B=2"), {"A": "1", "B": "2"})

    def test_quotes_stripped(self):
        self.assertEqual(env.parse('A="x y"\nB=\'z\''), {"A": "x y", "B": "z"})

    def test_hash_inside_quotes_kept(self):
        self.assertEqual(env.parse('A="x # y"'), {"A": "x # y"})

    def test_inline_comment_stripped(self):
        self.assertEqual(env.parse("A=abc  # note"), {"A": "abc"})

    def test_hash_without_space_is_part_of_value(self):
        self.assertEqual(env.parse("A=ab#cd"), {"A": "ab#cd"})

    def test_empty_value(self):
        self.assertEqual(env.parse("A=\nB=''"), {"A": "", "B": ""})

    def test_equals_in_value(self):
        self.assertEqual(env.parse("A=b=c"), {"A": "b=c"})

    def test_whitespace_around_key_and_value(self):
        self.assertEqual(env.parse("  A = 1  "), {"A": "1"})

    def test_malformed_lines_ignored(self):
        self.assertEqual(env.parse("no equals sign\n=novalue\nA=1"), {"A": "1"})

    def test_crlf_line_endings(self):
        self.assertEqual(env.parse("A=1\r\nB=2\r\n"), {"A": "1", "B": "2"})


class LoadTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / ".env"

    def test_loads_into_environ(self):
        self.path.write_text("SM_TEST_A=hello\n")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SM_TEST_A", None)
            env.load(self.path)
            self.assertEqual(os.environ["SM_TEST_A"], "hello")

    def test_real_env_wins_over_file(self):
        self.path.write_text("SM_TEST_A=from-file\n")
        with mock.patch.dict(os.environ, {"SM_TEST_A": "from-shell"}):
            env.load(self.path)
            self.assertEqual(os.environ["SM_TEST_A"], "from-shell")

    def test_empty_value_does_not_set(self):
        self.path.write_text("SM_TEST_A=\n")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SM_TEST_A", None)
            env.load(self.path)
            self.assertNotIn("SM_TEST_A", os.environ)

    def test_missing_file_is_fine(self):
        env.load(self.path)  # does not exist; must not raise

    def test_load_twice_is_harmless(self):
        self.path.write_text("SM_TEST_A=1\n")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SM_TEST_A", None)
            env.load(self.path)
            env.load(self.path)
            self.assertEqual(os.environ["SM_TEST_A"], "1")


if __name__ == "__main__":
    unittest.main()
