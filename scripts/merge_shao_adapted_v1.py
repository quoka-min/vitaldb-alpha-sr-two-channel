#!/usr/bin/env python3
"""Merge and validate Shao-adapted shard artifacts.

The merged cohort remains pre-adjudication.  This script never converts blank
reviewer decisions into completed manual review.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


PROTOCOL_VERSION = "shao-adapted-bis60plus10m-fixed120-v1"


CORE_DICTIONARY = {
    "study_id": ("identifier", "Pseudonymous case-level identifier", "text"),
    "patient_group_id": ("identifier", "Pseudonymous patient-level grouping identifier for repeated operations", "text"),
    "protocol_version": ("provenance", "Applied immutable protocol version", "text"),
    "status": ("flow", "Case-level extraction status", "category"),
    "manual_blinded_review_completed": ("alpha eligibility", "Whether blinded alpha-window review was completed", "0/1"),
    "manual_window_eligible": ("alpha eligibility", "Final blinded eligibility; blank until adjudication", "0/1/blank"),
    "fixed_window_start_sec": ("timing", "First 2-s grid point at or after first BIS<=60 after TCI + 600 s", "s from case start"),
    "fixed_window_end_sec": ("timing", "Fixed alpha-window end", "s from case start"),
    "automated_spectral_available": ("technical", "Provisional spectrum can be numerically calculated; not manual eligibility", "0/1"),
    "delta_0_5_4_db_equal_signal": ("EEG exposure", "Delta absolute power from samplewise equal-weight two-channel signal", "dB re 1 uV^2"),
    "theta_4_8_db_equal_signal": ("EEG exposure", "Theta absolute power from samplewise equal-weight two-channel signal", "dB re 1 uV^2"),
    "alpha_8_12_db_equal_signal": ("EEG exposure", "Primary alpha absolute power from samplewise equal-weight two-channel signal", "dB re 1 uV^2"),
    "alpha_8_13_sensitivity_db_equal_signal": ("EEG sensitivity exposure", "Monitor-band alpha absolute power", "dB re 1 uV^2"),
    "beta_13_30_db_equal_signal": ("EEG exposure", "Beta absolute power from samplewise equal-weight two-channel signal", "dB re 1 uV^2"),
    "alpha_8_12_db_mean_channel_power": ("EEG sensitivity exposure", "Equal average of channel-level linear alpha power, transformed to dB", "dB re 1 uV^2"),
    "sr_gt10": ("SR outcome", "Manufacturer SR strictly greater than 10 at least once after index+63 s", "0/1/blank"),
    "sr_gt20": ("SR outcome", "Manufacturer SR strictly greater than 20 at least once after index+63 s", "0/1/blank"),
    "sr_gt10_classifiable": ("SR outcome QC", "Threshold-specific positive or adequate negative follow-up", "0/1"),
    "sr_gt20_classifiable": ("SR outcome QC", "Threshold-specific positive or adequate negative follow-up", "0/1"),
    "sr_max": ("SR burden", "Maximum finite manufacturer SR in primary observation window", "%"),
    "sr_auc_percent_min": ("SR burden", "Gap-aware trapezoidal manufacturer SR AUC", "%*min"),
    "sr_twm_percent": ("SR burden", "SR AUC divided by integrated observation time", "%"),
    "sr_excess10_auc_percent_min": ("SR burden", "AUC above SR=10", "%*min"),
    "sr_excess20_auc_percent_min": ("SR burden", "AUC above SR=20", "%*min"),
    "surgery_duration_min": ("covariate", "Operation end minus operation start", "min"),
    "post_index_observation_duration_min": ("follow-up", "Operation end minus alpha-window end+63 s", "min"),
    "icu_admission": ("clinical outcome", "Proxy for postoperative ICU use: icu_days>0", "0/1"),
    "icu_los_days": ("clinical outcome", "VitalDB postoperative ICU length of stay; original icu_days", "days"),
    "postoperative_hospital_los_days": ("clinical outcome", "Integer calendar-day proxy from operation end to discharge", "days"),
    "total_hospital_los_days": ("clinical outcome", "Discharge marker minus admission marker", "days"),
    "in_hospital_death": ("clinical outcome", "Death during index hospitalization", "0/1"),
}


def read_csvs(root: Path, filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in root.glob(f"**/{filename}"):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if len(frame):
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_manifests(root: Path) -> list[dict]:
    manifests = []
    for path in root.glob("**/run_manifest.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        manifests.append(data)
    return manifests


def require_single_value(manifests: list[dict], key: str):
    values = {json.dumps(item.get(key), sort_keys=True) for item in manifests}
    if len(values) != 1:
        raise RuntimeError(f"manifest mismatch for {key}: {values}")
    return manifests[0].get(key)


def data_dictionary(columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        domain, description, unit = CORE_DICTIONARY.get(
            column, ("supporting/QC", "Pipeline supporting or quality-control variable; see protocol README", "as named")
        )
        rows.append(
            {
                "variable": column,
                "domain": domain,
                "description": description,
                "unit_or_coding": unit,
                "source": "VitalDB /cases, BIS tracks, Orchestra PPF20_RATE, or deterministic derivation",
                "manual_eeg_review_required_for_variable": int(column == "manual_window_eligible"),
            }
        )
    return pd.DataFrame(rows)


def write_readme(path: Path, summary: dict) -> None:
    content = f"""# VitalDB Shao-adapted fixed 120-second package

Protocol: `{PROTOCOL_VERSION}`

This package was rebuilt from the VitalDB public API and does not reuse prior
derived alpha or SR values.  The primary fixed EEG window starts on the first
2-second grid point at or after 600 seconds following the first BIS<=60 after
the first positive propofol TCI rate and lasts 120 seconds.  No later window is
searched when this fixed window is unsuitable.

The automated cohort is **provisional**.  `manual_blinded_review_completed=0`
means artifact, physiologic burst suppression, and transition-free eligibility
has not been confirmed.  Manufacturer SR endpoints and administrative clinical
outcomes are electronic variables and do not require whole-cohort EEG review.

Primary band definitions are delta 0.5-4, theta 4-8, alpha 8-12, and beta
13-30 Hz.  The primary channel derivation is the samplewise equal-weight signal
`(EEG1+EEG2)/2`; equal averaging of the two channel-level linear powers is
provided as a montage/polarity sensitivity analysis.

SR observation begins 63 seconds after the alpha-window end and ends at
operation end.  `sr_gt10` and `sr_gt20` use strict `>` operators.  SR AUC is
gap-aware and reported in percent-minutes; time-weighted mean SR divides the
AUC by integrated observation time.

Clinical outcomes:

- ICU admission proxy: `icu_days>0`
- ICU length of stay: original nonnegative `icu_days`
- postoperative hospital stay: integer calendar-day proxy from operation end
  to discharge; not an exact elapsed-time measurement
- total hospital stay: `(dis-adm)/86400`
- in-hospital death: `death_inhosp`

VitalDB does not provide ICU entry/exit timestamps, planned versus unplanned ICU
status, or death timing/cause.  These limitations must be retained in reports.

Merge counts: cohort={summary.get('cohort_n', 0)}, provisional spectra={summary.get('provisional_spectral_n', 0)}, blinded review rows={summary.get('review_rows', 0)}.
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    args = parser.parse_args()
    root = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifests = load_manifests(root)
    if len(manifests) != args.expected_shards:
        raise RuntimeError(f"expected {args.expected_shards} manifests, found {len(manifests)}")
    if any(item.get("protocol_version") != PROTOCOL_VERSION for item in manifests):
        raise RuntimeError("wrong protocol version in shard manifests")
    indices = sorted(int(item["shard_index"]) for item in manifests)
    if indices != list(range(args.expected_shards)):
        raise RuntimeError(f"missing or duplicate shard index: {indices}")
    for key in (
        "protocol_config_sha256", "source_cases_sha256", "source_tracks_sha256",
        "protocol_version", "shard_count",
    ):
        require_single_value(manifests, key)

    cohort = read_csvs(root, "cohort.csv")
    reviews = read_csvs(root, "reviewer_form.csv")
    keys = read_csvs(root, "restricted_key.csv")
    if not len(cohort):
        raise RuntimeError("no cohort rows found")
    if "caseid" in cohort.columns or "subjectid" in cohort.columns:
        raise RuntimeError("actual source IDs leaked into cohort artifact")
    if cohort.study_id.duplicated().any():
        raise RuntimeError("duplicate study_id in merged cohort")
    if len(reviews) and reviews.review_id.duplicated().any():
        raise RuntimeError("duplicate review_id in reviewer form")
    if len(keys) and keys.review_id.duplicated().any():
        raise RuntimeError("duplicate review_id in restricted key")
    forbidden = {
        "study_id", "patient_group_id", "alpha_8_12_db_equal_signal", "sr_gt10", "sr_gt20",
        "sr_auc_percent_min", "icu_admission", "in_hospital_death",
    }.intersection(reviews.columns)
    if forbidden:
        raise RuntimeError(f"reviewer form contains unblinded columns: {sorted(forbidden)}")

    cohort = cohort.sort_values("study_id")
    if len(reviews):
        reviews = reviews.sort_values("review_id")
    if len(keys):
        keys = keys.sort_values("review_id")
    cohort.to_csv(out / "cohort_all_fixed_window.csv", index=False)
    provisional = cohort.loc[pd.to_numeric(cohort.get("automated_spectral_available"), errors="coerce").eq(1)].copy()
    provisional.to_csv(out / "automatic_candidates_provisional.csv", index=False)
    reviews.to_csv(out / "reviewer_form_blinded.csv", index=False)
    keys.to_csv(out / "restricted_key_do_not_share_with_reviewers.csv", index=False)
    data_dictionary(list(cohort.columns)).to_csv(out / "data_dictionary.csv", index=False)

    image_out = out / "blinded_review_images"
    image_out.mkdir(exist_ok=True)
    for source in root.glob("**/images/*.png"):
        target = image_out / source.name
        if target.exists():
            raise RuntimeError(f"duplicate review image: {source.name}")
        shutil.copy2(source, target)

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "cohort_n": int(len(cohort)),
        "status_counts": cohort.status.value_counts(dropna=False).to_dict(),
        "provisional_spectral_n": int(len(provisional)),
        "review_rows": int(len(reviews)),
        "restricted_key_rows": int(len(keys)),
        "review_image_n": int(len(list(image_out.glob("*.png")))),
        "manual_blinded_review_completed": False,
        "analysis_cohort_finalized": False,
        "source_cases_sha256": manifests[0].get("source_cases_sha256"),
        "source_tracks_sha256": manifests[0].get("source_tracks_sha256"),
        "protocol_config_sha256": manifests[0].get("protocol_config_sha256"),
        "sr_gt_operator": ">",
        "sr_primary_end": "operation_end",
    }
    (out / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(out / "README_KO.md", summary)


if __name__ == "__main__":
    main()
