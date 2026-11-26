"""Tests for codeact_check: answer verification and stats reporting for the CodeAct demo.

Run: python3 test_codeact_check.py (stdlib only, no venv needed)
"""

import unittest

from codeact_check import EXPECTED_TOTAL, format_report, verify_answer


class TestVerifyAnswer(unittest.TestCase):
    def test_exact_int_passes(self):
        self.assertTrue(verify_answer(EXPECTED_TOTAL))

    def test_numeric_string_with_noise_passes(self):
        self.assertTrue(verify_answer(f"最終結果：{EXPECTED_TOTAL}"))

    def test_wrong_total_fails(self):
        self.assertFalse(verify_answer(EXPECTED_TOTAL + 1))

    def test_none_fails(self):
        self.assertFalse(verify_answer(None))


class TestFormatReport(unittest.TestCase):
    def test_report_carries_timings_and_tokens(self):
        out = format_report(
            ok=True,
            elapsed_s=144.28,
            input_tokens=2231,
            output_tokens=374,
        )
        self.assertIn("16566", out)
        self.assertIn("144.28", out)
        self.assertIn("2231", out)
        self.assertIn("374", out)
        # blended throughput must be present so runs stay comparable
        self.assertIn("tok/s", out)


if __name__ == "__main__":
    unittest.main()
