#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_shao_adapted_v1 as analysis


def synthetic_cases(n: int = 240, seed: int = 20260825) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    alpha = rng.normal(4.5, 2.2, n)
    surgery = rng.uniform(60, 300, n)
    observation = np.maximum(15, surgery - 12 + rng.normal(0, 8, n))
    repeated_group = np.array([f"P{index // 2:04d}" for index in range(n)])

    p10 = 1 / (1 + np.exp(-(-0.3 - 0.26 * alpha + 0.0025 * surgery)))
    p20 = 1 / (1 + np.exp(-(-1.2 - 0.22 * alpha + 0.0020 * surgery)))
    sr10 = rng.binomial(1, p10)
    sr20 = rng.binomial(1, p20)
    sr_max = np.where(sr20, rng.uniform(21, 55, n), np.where(sr10, rng.uniform(11, 20, n), rng.uniform(0, 10, n)))
    sr_twm = np.maximum(0, sr_max * rng.uniform(0.02, 0.25, n))
    sr_auc = sr_twm * observation

    icu_p = 1 / (1 + np.exp(-(-2.0 - 0.08 * alpha + 0.003 * surgery)))
    icu = rng.binomial(1, icu_p)
    death = np.zeros(n, dtype=int)
    death[rng.choice(n, size=3, replace=False)] = 1
    data = pd.DataFrame(
        {
            "study_id": [f"S{index:05d}" for index in range(n)],
            "patient_group_id": repeated_group,
            "automated_spectral_available": 1,
            "manual_blinded_review_completed": 1,
            "manual_window_eligible": 1,
            "alpha_8_12_db_equal_signal": alpha,
            "delta_0_5_4_db_equal_signal": alpha + rng.normal(4, 1, n),
            "theta_4_8_db_equal_signal": alpha + rng.normal(2, 1, n),
            "beta_13_30_db_equal_signal": alpha + rng.normal(-1, 1, n),
            "surgery_duration_min": surgery,
            "post_index_observation_duration_min": observation,
            "sr_gt10": sr10,
            "sr_gt20": sr20,
            "sr_max": sr_max,
            "sr_auc_percent_min": sr_auc,
            "sr_twm_percent": sr_twm,
            "sr_gt10_classifiable": 1,
            "sr_gt20_classifiable": 1,
            "icu_admission": icu,
            "icu_los_days": np.where(icu, rng.gamma(2, 1.5, n), 0),
            "postoperative_hospital_los_days": rng.gamma(2.5, 2.0, n),
            "total_hospital_los_days": rng.gamma(3.0, 2.0, n) + 1,
            "in_hospital_death": death,
        }
    )
    return data


class AnalyzeShaoAdaptedTests(unittest.TestCase):
    def test_full_run_provisional_and_outputs(self) -> None:
        data = synthetic_cases()
        data.loc[0, "manual_blinded_review_completed"] = 0
        data.loc[0, "manual_window_eligible"] = np.nan
        data.loc[1, "manual_window_eligible"] = 0
        data.loc[2, "automated_spectral_available"] = 0
        data.loc[3, "sr_gt20_classifiable"] = 0
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            input_csv = temp_path / "merged.csv"
            out = temp_path / "results"
            data.to_csv(input_csv, index=False)
            summary = analysis.run_analysis(input_csv, out)

            self.assertEqual(summary["analysis_status"], "provisional_manual_review_incomplete")
            self.assertEqual(summary["manual_ineligible_excluded_n"], 1)
            self.assertEqual(summary["analysis_n"], len(data) - 2)
            for filename in analysis.OUTPUT_FILES.values():
                self.assertTrue((out / filename).is_file(), filename)

            quartiles = pd.read_csv(out / analysis.OUTPUT_FILES["quartiles"])
            self.assertEqual(set(quartiles["endpoint"]), {"SR>10", "SR>20"})
            self.assertTrue(quartiles["wilson_95ci_low"].between(0, 1).all())
            self.assertTrue(quartiles["wilson_95ci_high"].between(0, 1).all())
            sr20_n = int(quartiles.loc[quartiles["endpoint"].eq("SR>20"), "classifiable_n"].sum())
            self.assertEqual(sr20_n, summary["analysis_n"] - 1)

            models = pd.read_csv(out / analysis.OUTPUT_FILES["logistic"])
            self.assertEqual(
                set(models["model"]),
                {
                    "unadjusted",
                    "surgery_duration_adjusted",
                    "observation_duration_adjusted",
                    "surgery_and_observation_duration_adjusted",
                },
            )
            self.assertEqual(set(models["effect_scale"]), {"per_1_db_lower_alpha", "per_1_sd_lower_alpha"})
            self.assertTrue(models["covariance"].str.startswith("cluster:patient_group_id").all())
            self.assertTrue(np.isfinite(models["odds_ratio"]).all())

            clinical = pd.read_csv(out / analysis.OUTPUT_FILES["clinical"])
            self.assertIn("ICU_LOS_days_among_ICU_users", set(clinical["endpoint"]))
            death_warnings = clinical.loc[clinical["endpoint"].eq("in_hospital_death"), "warning"].fillna("")
            self.assertTrue(death_warnings.str.contains("low outcome count").all())

            with (out / analysis.OUTPUT_FILES["summary"]).open(encoding="utf-8") as handle:
                disk_summary = json.load(handle)
            self.assertEqual(disk_summary["analysis_status"], summary["analysis_status"])
            self.assertIn("warnings", disk_summary)
            self.assertIn("clinical_outcome_models", disk_summary)
            self.assertIn("band_sr_spearman_correlations", disk_summary)

    def test_completed_review_is_final(self) -> None:
        data = synthetic_cases(120)
        prepared, metadata, issues = analysis.validate_and_prepare(data)
        self.assertEqual(metadata["analysis_status"], "final_manual_review_complete")
        self.assertEqual(len(prepared), 120)
        self.assertFalse(any(row["code"] == "manual_review_incomplete" for row in issues))

    def test_completed_review_requires_decision(self) -> None:
        data = synthetic_cases(40)
        data.loc[0, "manual_window_eligible"] = np.nan
        with self.assertRaisesRegex(ValueError, "requires a nonmissing"):
            analysis.validate_and_prepare(data)

    def test_duplicate_case_id_is_rejected(self) -> None:
        data = synthetic_cases(40)
        data.loc[1, "study_id"] = data.loc[0, "study_id"]
        with self.assertRaisesRegex(ValueError, "one row per case"):
            analysis.validate_and_prepare(data)

    def test_missing_schema_is_rejected(self) -> None:
        data = synthetic_cases(40).drop(columns=["sr_auc_percent_min"])
        with self.assertRaisesRegex(ValueError, "sr_auc_percent_min"):
            analysis.validate_and_prepare(data)

    def test_boolean_text_encodings(self) -> None:
        data = synthetic_cases(40)
        data["automated_spectral_available"] = "true"
        data["manual_blinded_review_completed"] = "yes"
        data["manual_window_eligible"] = "accepted"
        data["sr_gt10_classifiable"] = "1"
        data["sr_gt20_classifiable"] = "true"
        prepared, metadata, _ = analysis.validate_and_prepare(data)
        self.assertEqual(metadata["analysis_n"], 40)
        self.assertTrue(prepared["manual_window_eligible"].eq(1).all())

    def test_constant_alpha_collapses_to_one_group_without_splitting_ties(self) -> None:
        data = synthetic_cases(40)
        data["alpha_8_12_db_equal_signal"] = 5.0
        prepared, _, issues = analysis.validate_and_prepare(data)
        prepared = analysis.assign_alpha_quartiles(prepared, issues)
        self.assertEqual(prepared["alpha_quartile"].nunique(), 1)
        self.assertTrue(prepared["alpha_quartile"].eq("Q1_lowest").all())
        self.assertTrue(any(row["code"] == "collapsed_quantile_bins" for row in issues))

    def test_inconsistent_sr_thresholds_are_flagged(self) -> None:
        data = synthetic_cases(40)
        data.loc[0, ["sr_gt10", "sr_gt20"]] = [0, 1]
        _, _, issues = analysis.validate_and_prepare(data)
        self.assertTrue(any(row["code"] == "inconsistent_sr_thresholds" for row in issues))


if __name__ == "__main__":
    unittest.main(verbosity=2)
