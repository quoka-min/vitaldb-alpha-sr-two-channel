#!/usr/bin/env python3
"""Case-level analysis for the Shao-adapted VitalDB protocol (version 1).

The input is the merged, one-row-per-case CSV produced by the collection
workflow.  Automatically extractable fixed windows that have not yet received
manual blinded adjudication are retained only for a clearly labelled
*provisional* analysis.  A completed manual decision of ineligible always
removes the case.

This script deliberately does not infer that automated screening constitutes
manual blinded review.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.proportion import proportion_confint


PROTOCOL_VERSION = "shao-adapted-bis60plus10m-fixed120-v1"

CORE_COLUMNS = [
    "study_id",
    "automated_spectral_available",
    "manual_blinded_review_completed",
    "manual_window_eligible",
    "alpha_8_12_db_equal_signal",
    "delta_0_5_4_db_equal_signal",
    "theta_4_8_db_equal_signal",
    "beta_13_30_db_equal_signal",
    "surgery_duration_min",
    "post_index_observation_duration_min",
    "sr_gt10",
    "sr_gt20",
    "sr_max",
    "sr_auc_percent_min",
    "sr_twm_percent",
    "sr_gt10_classifiable",
    "sr_gt20_classifiable",
    "icu_admission",
    "icu_los_days",
    "postoperative_hospital_los_days",
    "total_hospital_los_days",
    "in_hospital_death",
]

NUMERIC_COLUMNS = [
    "alpha_8_12_db_equal_signal",
    "delta_0_5_4_db_equal_signal",
    "theta_4_8_db_equal_signal",
    "beta_13_30_db_equal_signal",
    "surgery_duration_min",
    "post_index_observation_duration_min",
    "sr_gt10",
    "sr_gt20",
    "sr_max",
    "sr_auc_percent_min",
    "sr_twm_percent",
    "icu_admission",
    "icu_los_days",
    "postoperative_hospital_los_days",
    "total_hospital_los_days",
    "in_hospital_death",
]

BOOLEAN_COLUMNS = [
    "automated_spectral_available",
    "manual_blinded_review_completed",
    "manual_window_eligible",
    "sr_gt10_classifiable",
    "sr_gt20_classifiable",
]

BAND_COLUMNS = {
    "delta_0_5_4": "delta_0_5_4_db_equal_signal",
    "theta_4_8": "theta_4_8_db_equal_signal",
    "alpha_8_12": "alpha_8_12_db_equal_signal",
    "beta_13_30": "beta_13_30_db_equal_signal",
}

SR_CONTINUOUS_OUTCOMES = {
    "sr_max_percent": "sr_max",
    "sr_auc_percent_min": "sr_auc_percent_min",
    "sr_time_weighted_mean_percent": "sr_twm_percent",
}

OUTPUT_FILES = {
    "dataset": "analysis_case_level.csv",
    "quartiles": "alpha_quartile_sr_incidence.csv",
    "logistic": "sr_logistic_models.csv",
    "quartile_models": "alpha_quartile_logistic_models.csv",
    "correlations": "band_sr_spearman_correlations.csv",
    "auc_ols": "sr_auc_log1p_ols.csv",
    "clinical": "clinical_outcome_models.csv",
    "warnings": "analysis_warnings.csv",
    "summary": "analysis_summary.json",
}


def _bool_value(value: Any) -> float:
    """Return 1/0/nan for common CSV boolean encodings."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return float(value)
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
        return float(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "eligible", "accept", "accepted"}:
        return 1.0
    if text in {"0", "false", "f", "no", "n", "ineligible", "reject", "rejected"}:
        return 0.0
    if text in {"", "na", "nan", "none", "null", "pending", "not_reviewed"}:
        return np.nan
    return np.nan


def _coerce_bool(series: pd.Series) -> pd.Series:
    return series.map(_bool_value).astype(float)


def _finite_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.where(np.isfinite(out), np.nan)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _join_notes(notes: Iterable[str]) -> str:
    return "; ".join(dict.fromkeys(str(note) for note in notes if note))


def validate_and_prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Validate schema, normalize values, and select the analysis candidates."""
    missing = [column for column in CORE_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(missing))
    if raw.empty:
        raise ValueError("Input CSV contains no rows.")
    if raw["study_id"].isna().any():
        raise ValueError("study_id contains missing values; one stable ID is required per case.")
    duplicated = raw["study_id"].astype(str).duplicated(keep=False)
    if duplicated.any():
        examples = raw.loc[duplicated, "study_id"].astype(str).unique()[:5]
        raise ValueError(
            "Input must contain one row per case; duplicated study_id values include: "
            + ", ".join(examples)
        )

    data = raw.copy()
    issue_rows: list[dict[str, Any]] = []
    for column in BOOLEAN_COLUMNS:
        original_nonmissing = data[column].notna() & data[column].astype(str).str.strip().ne("")
        data[column] = _coerce_bool(data[column])
        invalid_n = int((original_nonmissing & data[column].isna()).sum())
        if invalid_n:
            issue_rows.append({
                "scope": "input",
                "severity": "warning",
                "code": "unrecognized_boolean",
                "message": f"{column}: {invalid_n} value(s) were not recognized and were treated as missing.",
            })
    for column in NUMERIC_COLUMNS:
        original_nonmissing = data[column].notna() & data[column].astype(str).str.strip().ne("")
        data[column] = _finite_numeric(data[column])
        invalid_n = int((original_nonmissing & data[column].isna()).sum())
        if invalid_n:
            issue_rows.append({
                "scope": "input",
                "severity": "warning",
                "code": "nonnumeric_value",
                "message": f"{column}: {invalid_n} nonnumeric/nonfinite value(s) were treated as missing.",
            })

    for column in ("sr_gt10", "sr_gt20", "icu_admission", "in_hospital_death"):
        invalid_binary_n = int((data[column].notna() & ~data[column].isin([0, 1])).sum())
        if invalid_binary_n:
            issue_rows.append({
                "scope": "input",
                "severity": "warning",
                "code": "invalid_binary_outcome",
                "message": (
                    f"{column}: {invalid_binary_n} value(s) were outside 0/1 and will be excluded "
                    "from relevant binary models."
                ),
            })

    threshold_inconsistency = (
        data["sr_gt10_classifiable"].eq(1)
        & data["sr_gt20_classifiable"].eq(1)
        & data["sr_gt20"].eq(1)
        & ~data["sr_gt10"].eq(1)
    )
    if threshold_inconsistency.any():
        issue_rows.append({
            "scope": "input",
            "severity": "warning",
            "code": "inconsistent_sr_thresholds",
            "message": (
                f"{int(threshold_inconsistency.sum())} classifiable row(s) had SR>20=1 but SR>10!=1; "
                "the supplied endpoint values were not silently changed."
            ),
        })

    completed_without_decision = data["manual_blinded_review_completed"].eq(1) & data[
        "manual_window_eligible"
    ].isna()
    if completed_without_decision.any():
        raise ValueError(
            "manual_blinded_review_completed=1 requires a nonmissing manual_window_eligible decision "
            f"({int(completed_without_decision.sum())} inconsistent row(s))."
        )

    automated = data["automated_spectral_available"].eq(1)
    alpha_available = data["alpha_8_12_db_equal_signal"].notna()
    manual_completed = data["manual_blinded_review_completed"].eq(1)
    manual_ineligible = manual_completed & data["manual_window_eligible"].eq(0)
    manual_eligible = manual_completed & data["manual_window_eligible"].eq(1)
    manual_pending = automated & ~manual_completed

    if (automated & ~alpha_available).any():
        issue_rows.append({
            "scope": "cohort",
            "severity": "warning",
            "code": "spectral_flag_without_alpha",
            "message": (
                f"{int((automated & ~alpha_available).sum())} row(s) had automated_spectral_available=1 "
                "but no finite primary alpha value and were excluded."
            ),
        })

    include = automated & alpha_available & ~manual_ineligible
    selected = data.loc[include].copy()
    if selected.empty:
        raise ValueError("No provisionally or manually eligible fixed-window cases remain for analysis.")

    pending_n = int((manual_pending & alpha_available).sum())
    analysis_status = "provisional_manual_review_incomplete" if pending_n else "final_manual_review_complete"
    selected["analysis_status"] = analysis_status
    selected["manual_review_state"] = np.select(
        [
            selected["manual_blinded_review_completed"].eq(1)
            & selected["manual_window_eligible"].eq(1),
            selected["manual_blinded_review_completed"].eq(1)
            & selected["manual_window_eligible"].eq(0),
        ],
        ["eligible", "ineligible"],
        default="pending",
    )
    selected["surgery_duration_hr"] = selected["surgery_duration_min"] / 60.0
    selected["post_index_observation_duration_hr"] = (
        selected["post_index_observation_duration_min"] / 60.0
    )

    if pending_n:
        issue_rows.append({
            "scope": "cohort",
            "severity": "warning",
            "code": "manual_review_incomplete",
            "message": (
                f"{pending_n} included fixed-window candidate(s) have no completed blinded manual review. "
                "All results are provisional and automated screening is not manual adjudication."
            ),
        })

    for column in [
        "surgery_duration_min",
        "post_index_observation_duration_min",
        "sr_auc_percent_min",
        "icu_los_days",
        "postoperative_hospital_los_days",
        "total_hospital_los_days",
    ]:
        negative_n = int(selected[column].lt(0).sum())
        if negative_n:
            issue_rows.append({
                "scope": "input",
                "severity": "warning",
                "code": "negative_duration_or_auc",
                "message": f"{column}: {negative_n} negative value(s) will be excluded from relevant models.",
            })

    cluster_column = next(
        (
            column
            for column in ("subject_group_id", "patient_group_id")
            if column in selected.columns and selected[column].notna().any()
        ),
        None,
    )
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": analysis_status,
        "input_rows": int(len(data)),
        "automated_spectral_available_n": int(automated.sum()),
        "automated_without_primary_alpha_n": int((automated & ~alpha_available).sum()),
        "manual_review_completed_n_among_automated": int((automated & manual_completed).sum()),
        "manual_eligible_n_among_automated": int((automated & manual_eligible).sum()),
        "manual_ineligible_excluded_n": int((automated & manual_ineligible).sum()),
        "manual_review_pending_n_among_included": pending_n,
        "analysis_n": int(len(selected)),
        "cluster_id_column": cluster_column,
    }
    return selected, metadata, issue_rows


def assign_alpha_quartiles(data: pd.DataFrame, issue_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Add quantile categories without arbitrarily splitting tied values."""
    out = data.copy()
    alpha = out["alpha_8_12_db_equal_signal"]
    try:
        raw_codes, bins = pd.qcut(alpha, q=4, labels=False, retbins=True, duplicates="drop")
    except ValueError:
        raw_codes = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)
        bins = np.asarray([alpha.min(), alpha.max()])
    n_groups = int(pd.Series(raw_codes).dropna().nunique())
    if n_groups == 0:
        raw_codes = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)
        n_groups = 1
    if n_groups < 4:
        issue_rows.append({
            "scope": "alpha_quartiles",
            "severity": "warning",
            "code": "collapsed_quantile_bins",
            "message": (
                f"Only {n_groups} distinct alpha quantile group(s) could be formed because of tied or "
                "insufficient alpha values; identical values were not split arbitrarily."
            ),
        })
    labels: dict[int, str] = {}
    for index in range(max(n_groups, 1)):
        if index == 0:
            labels[index] = "Q1_lowest"
        elif index == n_groups - 1:
            labels[index] = f"Q{index + 1}_highest"
        else:
            labels[index] = f"Q{index + 1}"
    out["alpha_quartile"] = pd.Series(raw_codes, index=out.index).map(labels)
    out["alpha_quartile_number"] = pd.Series(raw_codes, index=out.index) + 1
    return out


def quartile_incidence(data: pd.DataFrame, status: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    endpoints = [
        ("SR>10", "sr_gt10", "sr_gt10_classifiable"),
        ("SR>20", "sr_gt20", "sr_gt20_classifiable"),
    ]
    quartiles = (
        data[["alpha_quartile_number", "alpha_quartile"]]
        .dropna()
        .drop_duplicates()
        .sort_values("alpha_quartile_number")
    )
    for endpoint, outcome, classifiable in endpoints:
        for item in quartiles.itertuples(index=False):
            complete_quartile = data.loc[
                data["alpha_quartile_number"].eq(item.alpha_quartile_number)
            ]
            group = data.loc[
                data["alpha_quartile_number"].eq(item.alpha_quartile_number)
                & data[classifiable].eq(1)
                & data[outcome].isin([0, 1])
            ]
            n = int(len(group))
            events = int(group[outcome].sum()) if n else 0
            if n:
                low, high = proportion_confint(events, n, alpha=0.05, method="wilson")
                incidence = events / n
            else:
                incidence = low = high = np.nan
            rows.append({
                "protocol_version": PROTOCOL_VERSION,
                "analysis_status": status,
                "endpoint": endpoint,
                "alpha_quartile": item.alpha_quartile,
                "alpha_quartile_number": int(item.alpha_quartile_number),
                "quartile_total_n": int(len(complete_quartile)),
                "alpha_min_db": complete_quartile["alpha_8_12_db_equal_signal"].min(),
                "alpha_median_db": complete_quartile["alpha_8_12_db_equal_signal"].median(),
                "alpha_max_db": complete_quartile["alpha_8_12_db_equal_signal"].max(),
                "classifiable_n": n,
                "events_n": events,
                "nonevents_n": n - events,
                "incidence_proportion": incidence,
                "incidence_percent": incidence * 100 if n else np.nan,
                "wilson_95ci_low": low,
                "wilson_95ci_high": high,
                "wilson_95ci_low_percent": low * 100 if n else np.nan,
                "wilson_95ci_high_percent": high * 100 if n else np.nan,
            })
    return pd.DataFrame(rows)


def _analysis_groups(frame: pd.DataFrame, cluster_column: str | None) -> tuple[pd.Series | None, str, int]:
    if not cluster_column or cluster_column not in frame.columns or not frame[cluster_column].notna().any():
        return None, "HC1", 0
    groups = frame[cluster_column].astype("string")
    missing = groups.isna()
    if missing.any():
        unique_missing = pd.Series(
            [f"__missing_case_{idx}" for idx in frame.index[missing]], index=frame.index[missing], dtype="string"
        )
        groups.loc[missing] = unique_missing
    n_groups = int(groups.nunique())
    if n_groups < 2:
        return None, "HC1", n_groups
    return groups.astype(str), f"cluster:{cluster_column}", n_groups


def _model_note(events: int | None, n: int, fit_notes: Sequence[str]) -> str:
    notes = list(fit_notes)
    if n < 30:
        notes.append("small sample (n<30)")
    if events is not None:
        nonevents = n - events
        if events < 10 or nonevents < 10:
            notes.append(f"low outcome count (events={events}, nonevents={nonevents})")
    return _join_notes(notes)


def _binary_model(
    data: pd.DataFrame,
    outcome: str,
    classifiable: str | None,
    adjustment_columns: Sequence[str],
    model_name: str,
    endpoint_name: str,
    status: str,
    cluster_column: str | None,
) -> list[dict[str, Any]]:
    columns = [outcome, "alpha_8_12_db_equal_signal", *adjustment_columns]
    if classifiable:
        columns.append(classifiable)
    if cluster_column:
        columns.append(cluster_column)
    frame = data[columns].copy()
    if classifiable:
        frame = frame.loc[frame[classifiable].eq(1)]
    frame = frame.loc[frame[outcome].isin([0, 1])]
    complete = [outcome, "alpha_8_12_db_equal_signal", *adjustment_columns]
    frame = frame.dropna(subset=complete)
    n = int(len(frame))
    events = int(frame[outcome].sum()) if n else 0
    alpha_sd = float(frame["alpha_8_12_db_equal_signal"].std(ddof=1)) if n > 1 else np.nan
    base = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": status,
        "endpoint": endpoint_name,
        "model": model_name,
        "n": n,
        "events_n": events,
        "nonevents_n": n - events,
        "alpha_sd_db": alpha_sd,
        "adjustment_variables": ",".join(adjustment_columns) if adjustment_columns else "none",
    }
    if n == 0 or events == 0 or events == n or not np.isfinite(alpha_sd) or alpha_sd <= 0:
        reason = "model not estimable: no observations, no outcome variation, or no alpha variation"
        return [
            {
                **base,
                "effect_scale": scale,
                "odds_ratio": np.nan,
                "ci_95_low": np.nan,
                "ci_95_high": np.nan,
                "p_value": np.nan,
                "covariance": "not_fitted",
                "cluster_n": 0,
                "warning": _model_note(events, n, [reason]),
            }
            for scale in ("per_1_db_lower_alpha", "per_1_sd_lower_alpha")
        ]

    frame = frame.copy()
    frame["lower_alpha_db"] = -frame["alpha_8_12_db_equal_signal"]
    design_columns = ["lower_alpha_db", *adjustment_columns]
    design = sm.add_constant(frame[design_columns].astype(float), has_constant="add")
    fit_notes: list[str] = []
    groups, covariance, cluster_n = _analysis_groups(frame, cluster_column)
    if covariance.startswith("cluster") and cluster_n <= design.shape[1]:
        fit_notes.append(f"few clusters relative to model parameters (clusters={cluster_n})")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = sm.GLM(frame[outcome].astype(float), design, family=sm.families.Binomial())
            if groups is not None:
                fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            else:
                fit = model.fit(cov_type="HC1")
        for warning in caught:
            text = str(warning.message)
            if "separation" in text.lower() or "overflow" in text.lower():
                fit_notes.append(text)
        beta = float(fit.params["lower_alpha_db"])
        se = float(fit.bse["lower_alpha_db"])
        p_value = float(fit.pvalues["lower_alpha_db"])
        ci_low, ci_high = (float(value) for value in fit.conf_int().loc["lower_alpha_db"])
        if not all(np.isfinite([beta, se, ci_low, ci_high])):
            fit_notes.append("nonfinite estimate or standard error; possible separation/instability")
        if abs(beta) > 5 or se > 5:
            fit_notes.append("extreme coefficient/SE; inspect possible separation or sparse data")
    except Exception as exc:  # statsmodels emits several exception types for separation/singularity
        reason = f"model fitting failed ({type(exc).__name__}: {exc})"
        return [
            {
                **base,
                "effect_scale": scale,
                "odds_ratio": np.nan,
                "ci_95_low": np.nan,
                "ci_95_high": np.nan,
                "p_value": np.nan,
                "covariance": covariance,
                "cluster_n": cluster_n,
                "warning": _model_note(events, n, [reason]),
            }
            for scale in ("per_1_db_lower_alpha", "per_1_sd_lower_alpha")
        ]

    result: list[dict[str, Any]] = []
    for scale, multiplier in (("per_1_db_lower_alpha", 1.0), ("per_1_sd_lower_alpha", alpha_sd)):
        try:
            odds_ratio = float(np.exp(beta * multiplier))
            low = float(np.exp(ci_low * multiplier))
            high = float(np.exp(ci_high * multiplier))
        except (OverflowError, FloatingPointError):
            odds_ratio = low = high = np.nan
            fit_notes.append("odds-ratio transform overflowed; possible separation/instability")
        result.append({
            **base,
            "effect_scale": scale,
            "odds_ratio": odds_ratio,
            "ci_95_low": low,
            "ci_95_high": high,
            "p_value": p_value,
            "covariance": covariance,
            "cluster_n": cluster_n,
            "warning": _model_note(events, n, fit_notes),
        })
    return result


def sr_logistic_models(data: pd.DataFrame, status: str, cluster_column: str | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    models = [
        ("unadjusted", []),
        ("surgery_duration_adjusted", ["surgery_duration_hr"]),
        ("observation_duration_adjusted", ["post_index_observation_duration_hr"]),
        (
            "surgery_and_observation_duration_adjusted",
            ["surgery_duration_hr", "post_index_observation_duration_hr"],
        ),
    ]
    for endpoint_name, outcome, classifiable in [
        ("SR>10", "sr_gt10", "sr_gt10_classifiable"),
        ("SR>20", "sr_gt20", "sr_gt20_classifiable"),
    ]:
        for model_name, covariates in models:
            rows.extend(
                _binary_model(
                    data,
                    outcome,
                    classifiable,
                    covariates,
                    model_name,
                    endpoint_name,
                    status,
                    cluster_column,
                )
            )
    return pd.DataFrame(rows)


def quartile_logistic_models(
    data: pd.DataFrame, status: str, cluster_column: str | None
) -> pd.DataFrame:
    """Compare lower alpha quartiles with the highest quartile after surgery-time adjustment."""
    rows: list[dict[str, Any]] = []
    for endpoint_name, outcome, classifiable in [
        ("SR>10", "sr_gt10", "sr_gt10_classifiable"),
        ("SR>20", "sr_gt20", "sr_gt20_classifiable"),
    ]:
        columns = [outcome, classifiable, "alpha_quartile_number", "surgery_duration_hr"]
        if cluster_column:
            columns.append(cluster_column)
        frame = data[columns].copy()
        frame = frame.loc[frame[classifiable].eq(1) & frame[outcome].isin([0, 1])]
        frame = frame.dropna(subset=[outcome, "alpha_quartile_number", "surgery_duration_hr"])
        n = int(len(frame))
        events = int(frame[outcome].sum()) if n else 0
        quartiles = sorted(int(value) for value in frame["alpha_quartile_number"].unique())
        reference = max(quartiles) if quartiles else None
        base = {
            "protocol_version": PROTOCOL_VERSION,
            "analysis_status": status,
            "endpoint": endpoint_name,
            "model": "alpha_quartile_surgery_duration_adjusted",
            "reference_quartile": f"Q{reference}_highest" if reference else None,
            "n": n,
            "events_n": events,
            "nonevents_n": n - events,
        }
        if n == 0 or events == 0 or events == n or len(quartiles) < 2:
            rows.append({
                **base,
                "contrast": "not_estimable",
                "odds_ratio": np.nan,
                "ci_95_low": np.nan,
                "ci_95_high": np.nan,
                "p_value": np.nan,
                "covariance": "not_fitted",
                "cluster_n": 0,
                "warning": _model_note(events, n, ["quartile model not estimable"]),
            })
            continue
        frame = frame.copy()
        dummy_names = []
        for quartile in quartiles:
            if quartile == reference:
                continue
            name = f"Q{quartile}_vs_Q{reference}"
            frame[name] = frame["alpha_quartile_number"].eq(quartile).astype(float)
            dummy_names.append(name)
        design = sm.add_constant(frame[[*dummy_names, "surgery_duration_hr"]].astype(float), has_constant="add")
        groups, covariance, cluster_n = _analysis_groups(frame, cluster_column)
        fit_notes: list[str] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = sm.GLM(frame[outcome].astype(float), design, family=sm.families.Binomial())
                fit = (
                    model.fit(cov_type="cluster", cov_kwds={"groups": groups})
                    if groups is not None
                    else model.fit(cov_type="HC1")
                )
            for warning in caught:
                text = str(warning.message)
                if "separation" in text.lower() or "overflow" in text.lower():
                    fit_notes.append(text)
            for name in dummy_names:
                low, high = (float(value) for value in fit.conf_int().loc[name])
                beta = float(fit.params[name])
                se = float(fit.bse[name])
                local_notes = list(fit_notes)
                if abs(beta) > 5 or se > 5 or not all(np.isfinite([beta, se, low, high])):
                    local_notes.append("extreme/nonfinite estimate; inspect possible separation or sparse cells")
                rows.append({
                    **base,
                    "contrast": name,
                    "odds_ratio": float(np.exp(beta)),
                    "ci_95_low": float(np.exp(low)),
                    "ci_95_high": float(np.exp(high)),
                    "p_value": float(fit.pvalues[name]),
                    "covariance": covariance,
                    "cluster_n": cluster_n,
                    "warning": _model_note(events, n, local_notes),
                })
        except Exception as exc:
            rows.append({
                **base,
                "contrast": "fit_failed",
                "odds_ratio": np.nan,
                "ci_95_low": np.nan,
                "ci_95_high": np.nan,
                "p_value": np.nan,
                "covariance": covariance,
                "cluster_n": cluster_n,
                "warning": _model_note(
                    events, n, [f"quartile model fitting failed ({type(exc).__name__}: {exc})"]
                ),
            })
    return pd.DataFrame(rows)


def spearman_correlations(data: pd.DataFrame, status: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for band_name, band_column in BAND_COLUMNS.items():
        for outcome_name, outcome_column in SR_CONTINUOUS_OUTCOMES.items():
            frame = data[[band_column, outcome_column]].dropna()
            n = int(len(frame))
            note = ""
            if n < 3 or frame[band_column].nunique() < 2 or frame[outcome_column].nunique() < 2:
                rho = p_value = np.nan
                note = "correlation not estimable: n<3 or a constant variable"
            else:
                result = spearmanr(frame[band_column], frame[outcome_column], nan_policy="omit")
                rho, p_value = float(result.statistic), float(result.pvalue)
                if n < 30:
                    note = "small sample (n<30)"
            rows.append({
                "protocol_version": PROTOCOL_VERSION,
                "analysis_status": status,
                "band": band_name,
                "band_column": band_column,
                "outcome": outcome_name,
                "outcome_column": outcome_column,
                "n": n,
                "spearman_rho": rho,
                "p_value": p_value,
                "warning": note,
            })
    return pd.DataFrame(rows)


def _continuous_model(
    data: pd.DataFrame,
    outcome: str,
    subset: pd.Series | None,
    adjustment_columns: Sequence[str],
    model_name: str,
    endpoint_name: str,
    status: str,
    cluster_column: str | None,
    log1p_outcome: bool,
) -> list[dict[str, Any]]:
    columns = [outcome, "alpha_8_12_db_equal_signal", *adjustment_columns]
    if cluster_column:
        columns.append(cluster_column)
    frame = data.loc[subset, columns].copy() if subset is not None else data[columns].copy()
    frame = frame.dropna(subset=[outcome, "alpha_8_12_db_equal_signal", *adjustment_columns])
    negative_n = int(frame[outcome].lt(0).sum()) if log1p_outcome else 0
    if log1p_outcome:
        frame = frame.loc[frame[outcome].ge(0)]
    n = int(len(frame))
    alpha_sd = float(frame["alpha_8_12_db_equal_signal"].std(ddof=1)) if n > 1 else np.nan
    base = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": status,
        "endpoint": endpoint_name,
        "model": model_name,
        "n": n,
        "alpha_sd_db": alpha_sd,
        "outcome_transform": "log1p" if log1p_outcome else "none",
        "adjustment_variables": ",".join(adjustment_columns) if adjustment_columns else "none",
    }
    initial_notes = [f"excluded {negative_n} negative outcome value(s)"] if negative_n else []
    if n < 3 or not np.isfinite(alpha_sd) or alpha_sd <= 0:
        return [
            {
                **base,
                "effect_scale": scale,
                "beta": np.nan,
                "ci_95_low": np.nan,
                "ci_95_high": np.nan,
                "p_value": np.nan,
                "multiplicative_change_in_outcome_plus_1": np.nan,
                "percent_change_in_outcome_plus_1": np.nan,
                "covariance": "not_fitted",
                "cluster_n": 0,
                "warning": _model_note(None, n, [*initial_notes, "model not estimable"]),
            }
            for scale in ("per_1_db_lower_alpha", "per_1_sd_lower_alpha")
        ]
    frame = frame.copy()
    frame["lower_alpha_db"] = -frame["alpha_8_12_db_equal_signal"]
    y = np.log1p(frame[outcome]) if log1p_outcome else frame[outcome]
    design = sm.add_constant(frame[["lower_alpha_db", *adjustment_columns]].astype(float), has_constant="add")
    groups, covariance, cluster_n = _analysis_groups(frame, cluster_column)
    fit_notes = list(initial_notes)
    try:
        model = sm.OLS(y.astype(float), design)
        fit = (
            model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            if groups is not None
            else model.fit(cov_type="HC1")
        )
        beta = float(fit.params["lower_alpha_db"])
        p_value = float(fit.pvalues["lower_alpha_db"])
        ci_low, ci_high = (float(value) for value in fit.conf_int().loc["lower_alpha_db"])
        if not all(np.isfinite([beta, ci_low, ci_high])):
            fit_notes.append("nonfinite estimate or confidence interval")
    except Exception as exc:
        return [
            {
                **base,
                "effect_scale": scale,
                "beta": np.nan,
                "ci_95_low": np.nan,
                "ci_95_high": np.nan,
                "p_value": np.nan,
                "multiplicative_change_in_outcome_plus_1": np.nan,
                "percent_change_in_outcome_plus_1": np.nan,
                "covariance": covariance,
                "cluster_n": cluster_n,
                "warning": _model_note(None, n, [*fit_notes, f"model fitting failed ({type(exc).__name__}: {exc})"]),
            }
            for scale in ("per_1_db_lower_alpha", "per_1_sd_lower_alpha")
        ]
    rows: list[dict[str, Any]] = []
    for scale, multiplier in (("per_1_db_lower_alpha", 1.0), ("per_1_sd_lower_alpha", alpha_sd)):
        scaled_beta = beta * multiplier
        scaled_low = ci_low * multiplier
        scaled_high = ci_high * multiplier
        ratio = float(np.exp(scaled_beta)) if log1p_outcome else np.nan
        rows.append({
            **base,
            "effect_scale": scale,
            "beta": scaled_beta,
            "ci_95_low": scaled_low,
            "ci_95_high": scaled_high,
            "p_value": p_value,
            "multiplicative_change_in_outcome_plus_1": ratio,
            "percent_change_in_outcome_plus_1": (ratio - 1) * 100 if log1p_outcome else np.nan,
            "covariance": covariance,
            "cluster_n": cluster_n,
            "warning": _model_note(None, n, fit_notes),
        })
    return rows


def auc_ols_model(data: pd.DataFrame, status: str, cluster_column: str | None) -> pd.DataFrame:
    rows = _continuous_model(
        data,
        "sr_auc_percent_min",
        None,
        ["surgery_duration_hr"],
        "log1p_auc_surgery_duration_adjusted",
        "SR AUC (percent-minutes)",
        status,
        cluster_column,
        log1p_outcome=True,
    )
    return pd.DataFrame(rows)


def clinical_models(data: pd.DataFrame, status: str, cluster_column: str | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for endpoint_name, outcome in [
        ("postoperative_ICU_admission", "icu_admission"),
        ("in_hospital_death", "in_hospital_death"),
    ]:
        for model_name, covariates in [
            ("unadjusted", []),
            ("surgery_duration_adjusted", ["surgery_duration_hr"]),
        ]:
            binary_rows = _binary_model(
                data,
                outcome,
                None,
                covariates,
                model_name,
                endpoint_name,
                status,
                cluster_column,
            )
            for row in binary_rows:
                row["model_family"] = "binomial_logistic"
            rows.extend(binary_rows)

    continuous_endpoints = [
        (
            "ICU_LOS_days_among_ICU_users",
            "icu_los_days",
            data["icu_admission"].eq(1),
        ),
        (
            "postoperative_hospital_LOS_calendar_days_among_in_hospital_survivors",
            "postoperative_hospital_los_days",
            data["in_hospital_death"].eq(0),
        ),
        (
            "total_hospital_LOS_days_among_in_hospital_survivors",
            "total_hospital_los_days",
            data["in_hospital_death"].eq(0),
        ),
    ]
    for endpoint_name, outcome, subset in continuous_endpoints:
        for model_name, covariates in [
            ("log1p_unadjusted", []),
            ("log1p_surgery_duration_adjusted", ["surgery_duration_hr"]),
        ]:
            continuous_rows = _continuous_model(
                data,
                outcome,
                subset,
                covariates,
                model_name,
                endpoint_name,
                status,
                cluster_column,
                log1p_outcome=True,
            )
            for row in continuous_rows:
                row["model_family"] = "OLS_log1p"
            rows.extend(continuous_rows)
    return pd.DataFrame(rows)


def _collect_model_warnings(
    tables: Sequence[tuple[str, pd.DataFrame]], issue_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out = list(issue_rows)
    for scope, table in tables:
        if "warning" not in table.columns:
            continue
        for index, row in table.loc[table["warning"].fillna("").astype(str).str.len().gt(0)].iterrows():
            label_parts = [str(row.get(key, "")) for key in ("endpoint", "model", "effect_scale", "contrast")]
            label = " / ".join(part for part in label_parts if part and part != "nan")
            out.append({
                "scope": scope,
                "severity": "warning",
                "code": "model_or_estimate_warning",
                "message": f"{label}: {row['warning']}",
            })
    return out


def run_analysis(input_csv: Path | str, output_dir: Path | str) -> dict[str, Any]:
    input_path = Path(input_csv)
    out = Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input case-level CSV not found: {input_path}")
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path, low_memory=False)
    data, metadata, issue_rows = validate_and_prepare(raw)
    data = assign_alpha_quartiles(data, issue_rows)
    status = str(metadata["analysis_status"])
    cluster_column = metadata.get("cluster_id_column")

    quartiles = quartile_incidence(data, status)
    logistic = sr_logistic_models(data, status, cluster_column)
    quartile_models = quartile_logistic_models(data, status, cluster_column)
    correlations = spearman_correlations(data, status)
    auc_ols = auc_ols_model(data, status, cluster_column)
    clinical = clinical_models(data, status, cluster_column)

    all_warnings = _collect_model_warnings(
        [
            ("sr_logistic_models", logistic),
            ("alpha_quartile_logistic_models", quartile_models),
            ("band_sr_spearman_correlations", correlations),
            ("sr_auc_log1p_ols", auc_ols),
            ("clinical_outcome_models", clinical),
        ],
        issue_rows,
    )
    warning_table = pd.DataFrame(all_warnings, columns=["scope", "severity", "code", "message"])

    data.to_csv(out / OUTPUT_FILES["dataset"], index=False)
    quartiles.to_csv(out / OUTPUT_FILES["quartiles"], index=False)
    logistic.to_csv(out / OUTPUT_FILES["logistic"], index=False)
    quartile_models.to_csv(out / OUTPUT_FILES["quartile_models"], index=False)
    correlations.to_csv(out / OUTPUT_FILES["correlations"], index=False)
    auc_ols.to_csv(out / OUTPUT_FILES["auc_ols"], index=False)
    clinical.to_csv(out / OUTPUT_FILES["clinical"], index=False)
    warning_table.to_csv(out / OUTPUT_FILES["warnings"], index=False)

    summary = {
        **metadata,
        "input_file": str(input_path.resolve()),
        "primary_alpha_definition": "8-12 Hz absolute power in dB from equal-signal two-channel derivation",
        "endpoint_definitions": {
            "SR>10": "manufacturer-reported SR strictly greater than 10; threshold-specific classifiability",
            "SR>20": "manufacturer-reported SR strictly greater than 20; threshold-specific classifiability",
            "SR_AUC": "integral of manufacturer SR over post-index observation, percent-minutes",
            "ICU_admission": "postoperative ICU-use proxy defined by VitalDB icu_days>0",
            "ICU_LOS": "VitalDB postoperative ICU length of stay in days, analyzed only among ICU users",
            "postoperative_hospital_LOS": "integer calendar-day proxy from operation end to discharge; association models restricted to in-hospital survivors",
            "total_hospital_LOS": "(dis-adm)/86400 days; association models restricted to in-hospital survivors",
            "in_hospital_death": "death during index hospitalization (binary)",
        },
        "modeling_notes": [
            "Alpha is modeled continuously as lower power per 1 dB and per sample SD.",
            "SR logistic models are unadjusted, surgery-duration adjusted, observation-duration adjusted, and adjusted for both durations.",
            "Quartile models compare lower alpha groups with the highest group and adjust only for surgery duration.",
            "SR AUC and LOS outcomes use OLS on log1p(outcome); exponentiated coefficients refer to outcome+1, not the raw outcome.",
            "Cluster-robust covariance is used when subject_group_id or patient_group_id is available; otherwise HC1 is used.",
            "Clinical outcome models are exploratory and sparse-event warnings must be considered.",
            "Hospital-LOS models exclude in-hospital deaths so death is not treated as ordinary live discharge; death is modeled separately.",
        ],
        "output_files": OUTPUT_FILES,
        "warning_n": int(len(warning_table)),
        "warnings": warning_table.to_dict(orient="records"),
        "quartile_incidence": quartiles.to_dict(orient="records"),
        "sr_logistic_models": logistic.to_dict(orient="records"),
        "alpha_quartile_logistic_models": quartile_models.to_dict(orient="records"),
        "band_sr_spearman_correlations": correlations.to_dict(orient="records"),
        "sr_auc_log1p_ols": auc_ols.to_dict(orient="records"),
        "clinical_outcome_models": clinical.to_dict(orient="records"),
    }
    with (out / OUTPUT_FILES["summary"]).open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, ensure_ascii=False, indent=2, allow_nan=False)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze merged case-level results for Shao-adapted VitalDB protocol v1."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Merged one-row-per-case CSV.")
    source.add_argument("--cohort", help="Alias for --input, retained for compatibility.")
    parser.add_argument("--out", required=True, help="Directory for CSV and JSON analysis outputs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_csv = args.input or args.cohort
    try:
        summary = run_analysis(input_csv, args.out)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(
        json.dumps(
            {
                "protocol_version": summary["protocol_version"],
                "analysis_status": summary["analysis_status"],
                "analysis_n": summary["analysis_n"],
                "warning_n": summary["warning_n"],
                "output_directory": str(Path(args.out).resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
