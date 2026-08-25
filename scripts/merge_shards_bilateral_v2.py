#!/usr/bin/env python3
"""Merge protocol-v2 shard artifacts without implying completed review."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def concat(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for path in root.glob(f"**/{name}"):
        try:
            frame = pd.read_csv(path)
            if len(frame):
                frames.append(frame)
        except pd.errors.EmptyDataError:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cohort = concat(root, "cohort.csv")
    if len(cohort):
        cohort = cohort.drop_duplicates("caseid", keep="last").sort_values("caseid")
    reviews = concat(root, "reviewer_form.csv")
    if len(reviews):
        reviews = reviews.drop_duplicates("review_id", keep="last")
    keys = concat(root, "restricted_key.csv")
    if len(keys):
        keys = keys.drop_duplicates("review_id", keep="last")

    cohort.to_csv(out / "cohort_bilateral_v2_pending_review.csv", index=False)
    reviews.to_csv(out / "reviewer_form_blinded.csv", index=False)
    keys.to_csv(out / "restricted_key_do_not_share_with_reviewers.csv", index=False)

    image_out = out / "blinded_review_images"
    image_out.mkdir(exist_ok=True)
    for source in root.glob("**/images/*.png"):
        target = image_out / source.name
        if not target.exists():
            shutil.copy2(source, target)

    summary = {
        "protocol_version": "bilateral-suppression-like-v2",
        "cohort_n": int(len(cohort)),
        "status_counts": cohort.status.value_counts(dropna=False).to_dict() if len(cohort) else {},
        "review_rows": int(len(reviews)),
        "restricted_key_rows": int(len(keys)),
        "image_n": int(len(list(image_out.glob("*.png")))),
        "manual_blinded_review_completed": False,
        "analysis_cohort_finalized": False,
        "primary_sqi_gate_applied": False,
        "sqi_sensitivity_subsets": ["sqi_sensitivity_median_ge90", "sqi_sensitivity_all_observed_ge90"],
        "bilateral_definition": "both filtered channels simultaneously within -5 to +5 uV continuously for >=0.5 s",
        "bilateral_suppression_like_auto_rejected": True,
        "unilateral_suppression_like_auto_rejected": False,
        "filter_design": "Butterworth band-pass",
        "filter_order": 4,
        "zero_phase": True,
        "bandpass_low_hz": 0.5,
        "bandpass_high_hz": 45.0,
        "notch_applied": False,
        "source": "Fresh VitalDB public API or byte-identical cached API track files; all spectra recomputed",
    }
    for column in [
        "search_window_n", "rejected_any_technical_n", "rejected_missing_n",
        "rejected_abs_amplitude_n", "rejected_peak_to_peak_n", "rejected_flatline_n",
        "rejected_bilateral_suppression_like_n", "accepted_for_blinded_review_n",
        "accepted_unilateral_suppression_like_n",
    ]:
        summary[f"sum_{column}"] = (
            int(pd.to_numeric(cohort[column], errors="coerce").fillna(0).sum()) if column in cohort else 0
        )
    (out / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
