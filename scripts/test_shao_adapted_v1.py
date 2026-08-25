#!/usr/bin/env python3
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

import vitaldb_shao_adapted_v1 as protocol


class ShaoAdaptedProtocolTests(unittest.TestCase):
    def test_fixed_window_is_bis60_plus_600_and_never_searched(self):
        target, start, end = protocol.fixed_window_times(101.3)
        self.assertAlmostEqual(target, 701.3)
        self.assertEqual(start, 702.0)
        self.assertEqual(end, 822.0)

    def test_equal_signal_and_power_average_are_both_reported(self):
        cfg = protocol.CFG
        time = np.arange(cfg.window_sec * cfg.fs) / cfg.fs
        ch1 = 20 * np.sin(2 * np.pi * 10 * time)
        ch2 = -20 * np.sin(2 * np.pi * 10 * time)
        result = protocol.spectra_for_window(ch1, ch2, 0)
        self.assertEqual(result["automated_spectral_available"], 1)
        self.assertTrue(np.isnan(result["alpha_8_12_db_equal_signal"]))
        self.assertGreater(result["alpha_8_12_db_mean_channel_power"], 0)
        self.assertEqual(result["equal_signal_polarity_cancellation_warning"], 1)

    def test_ten_hz_is_alpha_and_thirteen_hz_is_not_primary_alpha(self):
        cfg = protocol.CFG
        time = np.arange(cfg.window_sec * cfg.fs) / cfg.fs
        ten = 20 * np.sin(2 * np.pi * 10 * time)
        thirteen = 20 * np.sin(2 * np.pi * 13 * time)
        alpha = protocol.spectra_for_window(ten, ten, 0)
        beta = protocol.spectra_for_window(thirteen, thirteen, 0)
        self.assertGreater(alpha["alpha_8_12_db_equal_signal"], alpha["beta_13_30_db_equal_signal"])
        self.assertGreater(beta["beta_13_30_db_equal_signal"], beta["alpha_8_12_db_equal_signal"])

    def test_strict_sr_threshold_and_threshold_specific_classifiability(self):
        cfg = protocol.CFG
        short = pd.DataFrame({"time": np.arange(100), "value": np.r_[np.full(99, 10.0), 15.0]})
        metrics = protocol.manufacturer_sr_metrics(short, 0, 100, cfg)
        self.assertEqual(metrics["sr_gt10"], 1)
        self.assertTrue(np.isnan(metrics["sr_gt20"]))
        self.assertEqual(metrics["sr_gt20_classifiable"], 0)

        exact_ten = pd.DataFrame({"time": np.arange(700), "value": np.full(700, 10.0)})
        metrics = protocol.manufacturer_sr_metrics(exact_ten, 0, 700, cfg)
        self.assertEqual(metrics["sr_gt10"], 0)
        self.assertEqual(metrics["sr_gt20"], 0)

    def test_clinical_sentinel_and_same_day_proxy(self):
        sentinel = SimpleNamespace(
            adm=-3685366320,
            dis=-3685366320,
            opend=4374,
            icu_days=0,
            death_inhosp=0,
        )
        result = protocol.clinical_endpoints(sentinel)
        self.assertTrue(np.isnan(result["total_hospital_los_days"]))
        self.assertTrue(np.isnan(result["postoperative_hospital_los_days"]))

        same_day = SimpleNamespace(adm=-100000, dis=-2000, opend=3000, icu_days=2, death_inhosp=0)
        result = protocol.clinical_endpoints(same_day)
        self.assertEqual(result["postoperative_hospital_los_days"], 0)
        self.assertEqual(result["icu_admission"], 1)

    def test_reviewer_form_has_no_outcome_or_power(self):
        row = protocol.reviewer_row("REV-ABC", "images/REV-ABC.png")
        self.assertFalse(protocol.FORBIDDEN_REVIEW_COLUMNS.intersection(row))


if __name__ == "__main__":
    unittest.main()
