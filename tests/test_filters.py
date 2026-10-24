import unittest

from core import filters


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


class NormTest(unittest.TestCase):
    def test_lowercase_and_punctuation_removed(self):
        self.assertEqual(filters.norm("Halo, Dunia!"), "halo dunia")


class RepetitionTest(unittest.TestCase):
    def test_repeated_phrase_from_start(self):
        self.assertTrue(filters.is_repetitive("ya ya ya ya"))

    def test_normal_sentence_not_repetitive(self):
        self.assertFalse(filters.is_repetitive("kita mulai rapat hari ini"))

    def test_leading_token_hides_loop_from_is_repetitive_only(self):
        text = "di BFC, BFC, BFC, BFC."
        self.assertFalse(filters.is_repetitive(text))
        self.assertTrue(filters.is_degenerate(text))

    def test_short_text_not_degenerate(self):
        self.assertFalse(filters.is_degenerate("ya ya ya"))


class HallucinationReasonTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(filters.hallucination_reason("...", 2.0), "empty")

    def test_boilerplate(self):
        self.assertEqual(
            filters.hallucination_reason("Terima kasih telah menonton", 2.0), "boilerplate")

    def test_repetition(self):
        self.assertEqual(filters.hallucination_reason("tidak tidak tidak tidak", 2.0),
                         "repetition")

    def test_low_rate_over_long_silence(self):
        self.assertEqual(filters.hallucination_reason("oke", 10.0), "low-rate")

    def test_low_rate_ignored_for_short_duration(self):
        self.assertIsNone(filters.hallucination_reason("oke", 2.0))

    def test_duplicate_of_previous(self):
        self.assertEqual(
            filters.hallucination_reason("Baik, lanjut.", 3.5, ["baik lanjut"]), "duplicate")

    def test_short_echo_of_earlier_utterance(self):
        recent = ["baik lanjut", "ada pertanyaan"]
        self.assertEqual(filters.hallucination_reason("Baik, lanjut.", 2.0, recent), "echo")

    def test_normal_speech_kept(self):
        text = "Hari ini kita bahas rencana peluncuran produk baru."
        self.assertIsNone(filters.hallucination_reason(text, 5.0))


class CollapseLoopsTest(unittest.TestCase):
    def test_loop_cut_to_one_copy(self):
        text = "aku bagi 10 juta, 10 juta, 10 juta, 10 juta, 10 juta, 10 juta lalu pergi"
        self.assertEqual(filters.collapse_loops(text), "aku bagi 10 juta, lalu pergi")

    def test_below_threshold_untouched(self):
        text = "ya ya ya ya sudah"
        self.assertEqual(filters.collapse_loops(text), text)

    def test_plain_text_untouched(self):
        text = "kalimat biasa saja"
        self.assertEqual(filters.collapse_loops(text), text)


class CleanSegmentsTest(unittest.TestCase):
    def test_drops_hallucination_and_reports_it(self):
        segs = [seg(0, 5, "Hari ini kita bahas anggaran kuartal ketiga."),
                seg(5, 8, "Terima kasih telah menonton")]
        kept, dropped = filters.clean_segments(segs)
        self.assertEqual([s["text"] for s in kept], [segs[0]["text"]])
        self.assertEqual(len(dropped), 1)
        reason, start, duration, text = dropped[0]
        self.assertEqual((reason, start, duration), ("boilerplate", 5, 3))

    def test_kept_segment_gets_loop_collapsed(self):
        loop = "aku bagi 10 juta, " + "10 juta, " * 5 + "lalu pergi ke pasar sekarang"
        kept, dropped = filters.clean_segments([seg(0, 8, loop)])
        self.assertEqual(dropped, [])
        self.assertEqual(kept[0]["text"], "aku bagi 10 juta, lalu pergi ke pasar sekarang")

    def test_input_segments_not_mutated(self):
        loop = "aku bagi 10 juta, " + "10 juta, " * 5 + "lalu pergi ke pasar sekarang"
        original = seg(0, 8, loop)
        filters.clean_segments([original])
        self.assertEqual(original["text"], loop)

    def test_back_to_back_duplicate_dropped(self):
        text = "Baik kita lanjut ke agenda berikutnya."
        kept, dropped = filters.clean_segments([seg(0, 5, text), seg(5, 10, text)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped[0][0], "duplicate")


if __name__ == "__main__":
    unittest.main()
