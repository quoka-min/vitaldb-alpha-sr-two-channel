#!/usr/bin/env python3
import unittest

import numpy as np
import pandas as pd

import vitaldb_api_pipeline_bilateral_v2 as v2


class BilateralProtocolTests(unittest.TestCase):
    def setUp(self):
        self.fs = v2.CFG.fs
        self.n = v2.CFG.window_sec * self.fs

    def test_exact_half_second_bilateral_is_rejected(self):
        ch1 = np.full(self.n, 20.0)
        ch2 = np.full(self.n, -20.0)
        a = self.fs
        b = a + int(0.5 * self.fs)
        ch1[a:b] = 0.0
        ch2[a:b] = 0.0
        result = v2.suppression_like_summary(ch1, ch2, 0, 120)
        self.assertEqual(result["bilateral_suppression_like_flag"], 1)
        self.assertEqual(result["suppression_like_bilateral_n"], 1)

    def test_unilateral_is_flagged_but_not_bilateral(self):
        ch1 = np.full(self.n, 20.0)
        ch2 = np.full(self.n, -20.0)
        a = self.fs
        b = a + self.fs
        ch1[a:b] = 0.0
        result = v2.suppression_like_summary(ch1, ch2, 0, 120)
        self.assertEqual(result["bilateral_suppression_like_flag"], 0)
        self.assertEqual(result["unilateral_suppression_like_flag"], 1)

    def test_sqi_is_summary_not_primary_gate(self):
        t = np.arange(self.n) / self.fs
        ch1 = 20 * np.sin(2 * np.pi * 10 * t)
        ch2 = 18 * np.sin(2 * np.pi * 10 * t + 0.2)
        sqi = pd.DataFrame({"time": np.arange(120), "value": np.full(120, 20.0)})
        candidates, audit = v2.technical_candidates(ch1, ch2, ch1, ch2, sqi, 0, 120)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["sqi_sensitivity_median_ge90"], 0)
        self.assertEqual(candidates[0]["sqi_sensitivity_all_observed_ge90"], 0)
        self.assertEqual(audit["accepted_for_blinded_review_n"], 1)

    def test_bilateral_candidate_is_skipped(self):
        t = np.arange(self.n) / self.fs
        ch1 = 20 * np.sin(2 * np.pi * 10 * t)
        ch2 = 18 * np.sin(2 * np.pi * 10 * t + 0.2)
        ch1[self.fs : 2 * self.fs] = 0.0
        ch2[self.fs : 2 * self.fs] = 0.0
        sqi = pd.DataFrame({"time": np.arange(120), "value": np.full(120, 100.0)})
        candidates, audit = v2.technical_candidates(ch1, ch2, ch1, ch2, sqi, 0, 120)
        self.assertEqual(candidates, [])
        self.assertEqual(audit["rejected_bilateral_suppression_like_n"], 1)


if __name__ == "__main__":
    unittest.main()
