import unittest

from core.timestamps import fmt_ts


class FmtTsTest(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(fmt_ts(0), "00:00:00")

    def test_seconds_and_minutes(self):
        self.assertEqual(fmt_ts(59), "00:00:59")
        self.assertEqual(fmt_ts(60), "00:01:00")
        self.assertEqual(fmt_ts(754), "00:12:34")

    def test_hours(self):
        self.assertEqual(fmt_ts(3600), "01:00:00")
        self.assertEqual(fmt_ts(3661), "01:01:01")

    def test_over_a_day_does_not_wrap(self):
        self.assertEqual(fmt_ts(100 * 3600), "100:00:00")

    def test_fraction_is_truncated_not_rounded(self):
        self.assertEqual(fmt_ts(59.99), "00:00:59")


if __name__ == "__main__":
    unittest.main()
