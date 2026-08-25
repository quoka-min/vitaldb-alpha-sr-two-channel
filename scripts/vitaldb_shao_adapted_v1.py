#!/usr/bin/env python3
"""Fresh VitalDB extraction for the Shao-adapted fixed 120-second protocol.

The primary landmark is the first manufacturer BIS value <=60 after the first
positive propofol TCI pump rate.  The single candidate window begins on the
first 2-second grid point at or after landmark+600 seconds and lasts 120
seconds.  A failed window is never replaced with a later, more alpha-rich
window.

Automated output is deliberately *pre-adjudication*.  Artifact, physiologic
suppression, and induction/emergence/suppression transitions require blinded
visual review of the raw two-channel EEG and spectrogram.  Manufacturer SR and
administrative clinical outcomes do not require whole-cohort EEG adjudication.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import dpss

import vitaldb_api_pipeline as base


PROTOCOL_VERSION = "shao-adapted-bis60plus10m-fixed120-v1"
REQUIRED_TRACKS = (
    "BIS/EEG1_WAV",
    "BIS/EEG2_WAV",
    "BIS/BIS",
    "BIS/SQI",
    "BIS/SR",
    "Orchestra/PPF20_RATE",
)
SHAO_CITATION = (
    "Shao YR et al. Anesth Analg. 2020;131:1529-1539. "
    "doi:10.1213/ANE.0000000000004781"
)


@dataclass(frozen=True)
class Config:
    fs: int = 128
    landmark_offset_sec: int = 600
    window_sec: int = 120
    epoch_sec: int = 2
    grid_sec: int = 2
    filter_order: int = 4
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 45.0
    finite_fraction_min_for_provisional_spectrum: float = 0.98
    raw_abs_amplitude_review_flag_uv: float = 500.0
    raw_peak_to_peak_review_flag_uv: float = 800.0
    filtered_sd_review_flag_uv: float = 0.5
    suppression_like_amplitude_uv: float = 5.0
    suppression_like_min_sec: float = 0.5
    time_bandwidth: float = 3.0
    tapers: int = 5
    sr_memory_lag_sec: int = 63
    sr_max_bridge_gap_sec: float = 2.5
    sr_negative_min_observed_sec: int = 600
    sr_negative_min_coverage: float = 0.90
    clinical_time_sentinel_limit_days: int = 3650
    http_retries: int = 5


CFG = Config()
BANDS = {
    "delta_0_5_4": (0.5, 4.0),
    "theta_4_8": (4.0, 8.0),
    "alpha_8_12": (8.0, 12.0),
    "alpha_8_13_sensitivity": (8.0, 13.0),
    "beta_13_30": (13.0, 30.0),
}
FORBIDDEN_REVIEW_COLUMNS = {
    "study_id",
    "patient_group_id",
    "alpha_8_12_db_equal_signal",
    "sr_gt10",
    "sr_gt20",
    "sr_max",
    "icu_admission",
    "icu_los_days",
    "postoperative_hospital_los_days",
    "total_hospital_los_days",
    "in_hospital_death",
}


def finite_num(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def pseudonym(prefix: str, value: Any, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{prefix}:{value}".encode("utf-8")).hexdigest()[:16].upper()
    return f"{prefix}-{digest}"


def config_hash(cfg: Config = CFG) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    payload = path.read_bytes()
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    return hashlib.sha256(payload).hexdigest()


def fixed_window_times(first_bis60_sec: float, cfg: Config = CFG) -> tuple[float, float, float]:
    target = float(first_bis60_sec + cfg.landmark_offset_sec)
    start = math.ceil(target / cfg.grid_sec) * cfg.grid_sec
    return target, float(start), float(start + cfg.window_sec)


def eligible_cases(vdb: base.VitalDB) -> pd.DataFrame:
    cases = vdb.cases.copy()
    cases["age"] = pd.to_numeric(cases.get("age"), errors="coerce")
    cases["asa"] = pd.to_numeric(cases.get("asa"), errors="coerce")
    general = cases.get("ane_type", pd.Series(index=cases.index, dtype=object)).eq("General")
    cases = cases.loc[
        (cases.age >= 18)
        & cases.asa.between(1, 4)
        & general
        & cases.opstart.notna()
        & cases.opend.notna()
        & cases.aneend.notna()
    ].copy()
    available = set(cases.caseid.astype(int))
    for track in REQUIRED_TRACKS:
        available &= set(vdb.trks.loc[vdb.trks.tname.eq(track), "caseid"].astype(int))
    return cases.loc[cases.caseid.astype(int).isin(available)].sort_values("caseid")


def slice_samples(signal: np.ndarray, start_sec: float, duration_sec: float, fs: int) -> np.ndarray:
    first = int(round(start_sec * fs))
    stop = first + int(round(duration_sec * fs))
    if first < 0 or stop > len(signal):
        return np.asarray([], dtype=float)
    return np.asarray(signal[first:stop], dtype=float)


def longest_true_run(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool)
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int(np.max(ends - starts)) if len(starts) else 0


def filter_wave(raw: np.ndarray, cfg: Config = CFG) -> np.ndarray:
    """Zero-phase fourth-order Butterworth 0.5-45 Hz filter; no notch.

    Interpolation is a numerical bridge for filtering only.  Original missing
    samples are restored to NaN and missingness remains part of technical QC.
    """
    raw = np.asarray(raw, dtype=float)
    good = np.isfinite(raw)
    if good.sum() < cfg.fs * cfg.window_sec:
        return np.full_like(raw, np.nan)
    index = np.arange(len(raw))
    filled = np.interp(index, index[good], raw[good])
    sos = butter(
        cfg.filter_order,
        [cfg.bandpass_low_hz / (cfg.fs / 2), cfg.bandpass_high_hz / (cfg.fs / 2)],
        btype="band",
        output="sos",
    )
    filtered = sosfiltfilt(sos, filled)
    filtered[~good] = np.nan
    return filtered


def channel_window_qc(raw: np.ndarray, filtered: np.ndarray, start: float, cfg: Config = CFG) -> dict[str, Any]:
    raw_window = slice_samples(raw, start, cfg.window_sec, cfg.fs)
    filtered_window = slice_samples(filtered, start, cfg.window_sec, cfg.fs)
    expected = cfg.window_sec * cfg.fs
    exact = len(raw_window) == expected and len(filtered_window) == expected
    finite = np.isfinite(raw_window)
    finite_fraction = float(finite.mean()) if len(raw_window) else 0.0
    raw_values = raw_window[finite]
    filtered_values = filtered_window[np.isfinite(filtered_window)]
    abs_max = float(np.max(np.abs(raw_values))) if len(raw_values) else np.nan
    peak_to_peak = float(np.ptp(raw_values)) if len(raw_values) else np.nan
    filtered_sd = float(np.std(filtered_values)) if len(filtered_values) else np.nan
    max_missing_run = longest_true_run(~finite) / cfg.fs if len(raw_window) else np.nan
    provisional = bool(exact and finite_fraction >= cfg.finite_fraction_min_for_provisional_spectrum)
    return {
        "exact_sample_count": int(exact),
        "sample_count": int(len(raw_window)),
        "finite_fraction": finite_fraction,
        "longest_missing_run_sec": max_missing_run,
        "raw_abs_max_uv": abs_max,
        "raw_peak_to_peak_uv": peak_to_peak,
        "filtered_sd_uv": filtered_sd,
        "review_flag_extreme_abs_amplitude": int(np.isfinite(abs_max) and abs_max > cfg.raw_abs_amplitude_review_flag_uv),
        "review_flag_extreme_peak_to_peak": int(np.isfinite(peak_to_peak) and peak_to_peak > cfg.raw_peak_to_peak_review_flag_uv),
        "review_flag_near_flat_sd": int(np.isfinite(filtered_sd) and filtered_sd < cfg.filtered_sd_review_flag_uv),
        "provisional_spectrum_technical_pass": int(provisional),
    }


def low_amplitude_review_flags(filtered1: np.ndarray, filtered2: np.ndarray, start: float, cfg: Config = CFG) -> dict[str, Any]:
    x1 = slice_samples(filtered1, start, cfg.window_sec, cfg.fs)
    x2 = slice_samples(filtered2, start, cfg.window_sec, cfg.fs)
    expected = cfg.window_sec * cfg.fs
    if len(x1) != expected or len(x2) != expected:
        return {
            "suppression_like_ch1_flag": 0,
            "suppression_like_ch2_flag": 0,
            "suppression_like_bilateral_flag": 0,
            "suppression_like_is_review_aid_only": 1,
        }
    minimum = int(math.ceil(cfg.suppression_like_min_sec * cfg.fs))
    low1 = np.isfinite(x1) & (np.abs(x1) <= cfg.suppression_like_amplitude_uv)
    low2 = np.isfinite(x2) & (np.abs(x2) <= cfg.suppression_like_amplitude_uv)
    return {
        "suppression_like_ch1_flag": int(longest_true_run(low1) >= minimum),
        "suppression_like_ch2_flag": int(longest_true_run(low2) >= minimum),
        "suppression_like_bilateral_flag": int(longest_true_run(low1 & low2) >= minimum),
        "suppression_like_is_review_aid_only": 1,
    }


def fill_small_missing(epoch: np.ndarray) -> np.ndarray:
    values = np.asarray(epoch, dtype=float)
    good = np.isfinite(values)
    if good.all() or not good.any():
        return values
    index = np.arange(len(values))
    return np.interp(index, index[good], values[good])


def epoch_psd(epoch: np.ndarray, cfg: Config = CFG) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(epoch, dtype=float)
    if len(x) != cfg.epoch_sec * cfg.fs or not np.isfinite(x).all():
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = x - np.mean(x)
    tapers = dpss(len(x), cfg.time_bandwidth, cfg.tapers)
    psd = np.mean(
        [np.abs(np.fft.rfft(x * taper)) ** 2 / (cfg.fs * np.sum(taper * taper)) for taper in tapers],
        axis=0,
    )
    if len(psd) > 2:
        psd[1:-1] *= 2
    frequency = np.fft.rfftfreq(len(x), 1 / cfg.fs)
    return frequency, psd


def integrate_band(frequency: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    include = (frequency >= low) & (frequency <= high)
    if np.sum(include) < 2:
        return np.nan
    return float(np.trapezoid(psd[include], frequency[include]))


def spectra_for_window(filtered1: np.ndarray, filtered2: np.ndarray, start: float, cfg: Config = CFG) -> dict[str, Any]:
    """Return primary equal-voltage-signal and sensitivity channel-power averages."""
    window1 = slice_samples(filtered1, start, cfg.window_sec, cfg.fs)
    window2 = slice_samples(filtered2, start, cfg.window_sec, cfg.fs)
    expected = cfg.window_sec * cfg.fs
    if len(window1) != expected or len(window2) != expected:
        return {"automated_spectral_available": 0}

    correlation = np.corrcoef(
        fill_small_missing(window1), fill_small_missing(window2)
    )[0, 1]
    accum: dict[str, dict[str, list[float]]] = {
        band: {"eeg1": [], "eeg2": [], "equal_signal": [], "mean_channel_power": []}
        for band in BANDS
    }
    interpolated = 0
    total = 0
    epoch_n = cfg.window_sec // cfg.epoch_sec
    for index in range(epoch_n):
        first = index * cfg.epoch_sec * cfg.fs
        stop = first + cfg.epoch_sec * cfg.fs
        e1_raw = window1[first:stop]
        e2_raw = window2[first:stop]
        interpolated += int((~np.isfinite(e1_raw)).sum() + (~np.isfinite(e2_raw)).sum())
        total += len(e1_raw) + len(e2_raw)
        e1 = fill_small_missing(e1_raw)
        e2 = fill_small_missing(e2_raw)
        frequency, p1 = epoch_psd(e1, cfg)
        _, p2 = epoch_psd(e2, cfg)
        _, peq = epoch_psd((e1 + e2) / 2.0, cfg)
        if not len(frequency):
            return {"automated_spectral_available": 0}
        for name, (low, high) in BANDS.items():
            b1 = integrate_band(frequency, p1, low, high)
            b2 = integrate_band(frequency, p2, low, high)
            beq = integrate_band(frequency, peq, low, high)
            accum[name]["eeg1"].append(b1)
            accum[name]["eeg2"].append(b2)
            accum[name]["equal_signal"].append(beq)
            accum[name]["mean_channel_power"].append((b1 + b2) / 2.0)

    output: dict[str, Any] = {
        "automated_spectral_available": 1,
        "spectral_epoch_n": int(epoch_n),
        "spectral_interpolated_sample_fraction": float(interpolated / total) if total else np.nan,
        "eeg_channel_correlation": float(correlation),
        "equal_signal_polarity_cancellation_warning": int(np.isfinite(correlation) and correlation < 0),
    }
    for band, derivations in accum.items():
        for derivation, values in derivations.items():
            linear = float(np.mean(values))
            output[f"{band}_linear_uv2_{derivation}"] = linear
            output[f"{band}_db_{derivation}"] = float(10 * np.log10(linear)) if linear > 0 else np.nan
    output["initial_alpha_db"] = output["alpha_8_12_db_equal_signal"]
    return output


def sqi_summary(sqi: pd.DataFrame, start: float, end: float) -> dict[str, Any]:
    values = pd.to_numeric(
        sqi.loc[sqi.time.between(start, end, inclusive="left"), "value"], errors="coerce"
    ).to_numpy(float)
    values = values[np.isfinite(values)]
    return {
        "sqi_n_obs": int(len(values)),
        "sqi_min": float(np.min(values)) if len(values) else np.nan,
        "sqi_median": float(np.median(values)) if len(values) else np.nan,
        "sqi_mean": float(np.mean(values)) if len(values) else np.nan,
        "sqi_fraction_ge90": float(np.mean(values >= 90)) if len(values) else np.nan,
        "sqi_is_stability_criterion": 0,
        "sqi_is_primary_alpha_gate": 0,
    }


def _event_runs(times: np.ndarray, above: np.ndarray, max_gap: float) -> int:
    count = 0
    prior_time = np.nan
    prior_above = False
    for current_time, current_above in zip(times, above):
        if current_above and (not prior_above or not np.isfinite(prior_time) or current_time - prior_time > max_gap):
            count += 1
        prior_time = current_time
        prior_above = bool(current_above)
    return count


def manufacturer_sr_metrics(sr: pd.DataFrame, start: float, end: float, cfg: Config = CFG) -> dict[str, Any]:
    expected_sec = max(0.0, float(end - start)) if np.isfinite(start) and np.isfinite(end) else 0.0
    selected = sr.loc[sr.time.between(start, end, inclusive="both")].copy() if expected_sec > 0 else pd.DataFrame()
    if len(selected):
        selected["time"] = pd.to_numeric(selected.time, errors="coerce")
        selected["value"] = pd.to_numeric(selected.value, errors="coerce")
        selected = selected.dropna(subset=["time", "value"]).sort_values("time").drop_duplicates("time", keep="last")
        in_range = selected.value.between(0, 100)
        invalid_n = int((~in_range).sum())
        selected = selected.loc[in_range]
    else:
        invalid_n = 0
        selected = pd.DataFrame(columns=["time", "value"])
    times = selected.time.to_numpy(float)
    values = selected.value.to_numpy(float)
    unique_seconds = len(np.unique(np.rint(times).astype(int))) if len(times) else 0
    coverage = min(1.0, unique_seconds / expected_sec) if expected_sec > 0 else 0.0
    gt10 = values > 10
    gt20 = values > 20
    event10 = bool(gt10.any()) if len(values) else False
    event20 = bool(gt20.any()) if len(values) else False
    negative_classifiable = unique_seconds >= cfg.sr_negative_min_observed_sec and coverage >= cfg.sr_negative_min_coverage

    auc = excess10 = excess20 = integrated_sec = 0.0
    if len(values) >= 2:
        dt = np.diff(times)
        use = (dt > 0) & (dt <= cfg.sr_max_bridge_gap_sec)
        integrated_sec = float(np.sum(dt[use]))
        auc = float(np.sum((values[:-1][use] + values[1:][use]) * 0.5 * dt[use]) / 60.0)
        v10 = np.maximum(values - 10.0, 0.0)
        v20 = np.maximum(values - 20.0, 0.0)
        excess10 = float(np.sum((v10[:-1][use] + v10[1:][use]) * 0.5 * dt[use]) / 60.0)
        excess20 = float(np.sum((v20[:-1][use] + v20[1:][use]) * 0.5 * dt[use]) / 60.0)
    twm = auc * 60.0 / integrated_sec if integrated_sec > 0 else np.nan
    return {
        "sr_expected_observation_sec": expected_sec,
        "sr_valid_observed_sec": int(unique_seconds),
        "sr_coverage_fraction": coverage,
        "sr_invalid_out_of_range_n": invalid_n,
        "sr_integrated_duration_sec": integrated_sec,
        "sr_gt10_classifiable": int(event10 or negative_classifiable),
        "sr_gt20_classifiable": int(event20 or negative_classifiable),
        "sr_gt10": int(event10) if (event10 or negative_classifiable) else np.nan,
        "sr_gt20": int(event20) if (event20 or negative_classifiable) else np.nan,
        "sr_gt10_event_count": _event_runs(times, gt10, cfg.sr_max_bridge_gap_sec) if len(values) else 0,
        "sr_gt20_event_count": _event_runs(times, gt20, cfg.sr_max_bridge_gap_sec) if len(values) else 0,
        "sr_gt10_observed_sec": int(np.sum(gt10)) if len(values) else 0,
        "sr_gt20_observed_sec": int(np.sum(gt20)) if len(values) else 0,
        "sr_max": float(np.max(values)) if len(values) else np.nan,
        "sr_mean_observed": float(np.mean(values)) if len(values) else np.nan,
        "sr_auc_percent_min": auc if integrated_sec > 0 else np.nan,
        "sr_twm_percent": twm,
        "sr_excess10_auc_percent_min": excess10 if integrated_sec > 0 else np.nan,
        "sr_excess20_auc_percent_min": excess20 if integrated_sec > 0 else np.nan,
        "sr_threshold_operator": ">",
        "sr_primary_sqi_gate_applied": 0,
        "sr_manual_eeg_review_applied": 0,
    }


def prior_sr_flags(sr: pd.DataFrame, start: float, end: float) -> dict[str, Any]:
    selected = sr.loc[sr.time.between(start, end, inclusive="left")].copy()
    values = pd.to_numeric(selected.get("value"), errors="coerce").to_numpy(float) if len(selected) else np.asarray([])
    values = values[np.isfinite(values) & (values >= 0) & (values <= 100)]
    return {
        "prior_sr_gt10": int(bool(len(values) and np.any(values > 10))),
        "prior_sr_gt20": int(bool(len(values) and np.any(values > 20))),
        "prior_sr_max": float(np.max(values)) if len(values) else np.nan,
    }


def _valid_clinical_time(value: Any, cfg: Config = CFG) -> bool:
    number = finite_num(value)
    limit = cfg.clinical_time_sentinel_limit_days * 86400.0
    return bool(np.isfinite(number) and abs(number) <= limit)


def clinical_endpoints(row: Any, cfg: Config = CFG) -> dict[str, Any]:
    adm = finite_num(getattr(row, "adm", np.nan))
    discharge = finite_num(getattr(row, "dis", np.nan))
    opend = finite_num(getattr(row, "opend", np.nan))
    icu_raw = finite_num(getattr(row, "icu_days", np.nan))
    death_raw = finite_num(getattr(row, "death_inhosp", np.nan))
    time_valid = _valid_clinical_time(adm, cfg) and _valid_clinical_time(discharge, cfg)

    total_los = (discharge - adm) / 86400.0 if time_valid else np.nan
    if np.isfinite(total_los) and total_los < 0:
        total_qc = "negative_interval"
        total_los = np.nan
    elif np.isfinite(total_los):
        total_qc = "valid_calendar_day_interval"
    else:
        total_qc = "invalid_or_sentinel_source"

    raw_postop = (discharge - opend) / 86400.0 if time_valid and np.isfinite(opend) else np.nan
    if not np.isfinite(raw_postop):
        postop_los = np.nan
        postop_qc = "invalid_or_sentinel_source"
    elif raw_postop >= 0:
        postop_los = float(math.ceil(raw_postop - 1e-12))
        postop_qc = "valid_calendar_day_proxy"
    elif raw_postop > -1:
        postop_los = 0.0
        postop_qc = "same_calendar_day_negative_clock_proxy_set_zero"
    else:
        postop_los = np.nan
        postop_qc = "negative_more_than_one_day_or_sentinel"

    icu_valid = np.isfinite(icu_raw) and icu_raw >= 0
    icu_los = icu_raw if icu_valid else np.nan
    icu_admission = int(icu_raw > 0) if icu_valid else np.nan
    death = int(death_raw) if death_raw in (0, 1) else np.nan
    icu_vs_total_flag = int(
        np.isfinite(icu_los) and np.isfinite(total_los) and icu_los > total_los
    )
    return {
        "adm_sec_raw": adm,
        "dis_sec_raw": discharge,
        "opend_sec_raw": opend,
        "icu_days_raw": icu_raw,
        "death_inhosp_raw": death_raw,
        "icu_admission": icu_admission,
        "icu_los_days": icu_los,
        "postoperative_hospital_los_raw_interval_days": raw_postop,
        "postoperative_hospital_los_days": postop_los,
        "postoperative_hospital_los_qc": postop_qc,
        "total_hospital_los_days": total_los,
        "total_hospital_los_qc": total_qc,
        "icu_los_exceeds_total_los_qc_flag": icu_vs_total_flag,
        "in_hospital_death": death,
        "clinical_outcomes_are_manual_eeg_review_independent": 1,
    }


def reviewer_row(review_id: str, image_file: str) -> dict[str, Any]:
    row = {
        "review_id": review_id,
        "image_file": image_file,
        "window_definition": "first_BIS_le60_after_TCI_plus_600s_fixed_120s",
        "both_channels_interpretable": "",
        "artifact_present": "",
        "physiologic_suppression_present": "",
        "transition_present": "",
        "eligible": "",
        "reviewer_confidence": "",
        "reviewer_comment": "",
    }
    overlap = FORBIDDEN_REVIEW_COLUMNS.intersection(row)
    if overlap:
        raise AssertionError(f"outcome leakage into reviewer form: {sorted(overlap)}")
    return row


def process_case(vdb: base.VitalDB, row: Any, out: Path, salt: str, cfg: Config = CFG) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    caseid = int(row.caseid)
    subjectid = getattr(row, "subjectid", caseid)
    study_id = pseudonym("STUDY", caseid, salt)
    patient_group_id = pseudonym("PATIENT", subjectid, salt)
    result: dict[str, Any] = {
        "study_id": study_id,
        "patient_group_id": patient_group_id,
        "protocol_version": PROTOCOL_VERSION,
        "manual_blinded_review_completed": 0,
        "manual_window_eligible": np.nan,
        "analysis_stage": "pre_adjudication_fixed_window_candidate",
        **clinical_endpoints(row, cfg),
    }
    for column in (
        "age", "sex", "height", "weight", "bmi", "asa", "emop", "department",
        "optype", "ane_type", "anestart", "aneend", "opstart", "opend",
        "preop_cr", "preop_alb", "preop_hb", "intraop_ppf", "intraop_mdz",
    ):
        if hasattr(row, column):
            result[column] = getattr(row, column)
    opstart = finite_num(getattr(row, "opstart", np.nan))
    opend = finite_num(getattr(row, "opend", np.nan))
    aneend = finite_num(getattr(row, "aneend", np.nan))
    result["surgery_duration_min"] = (opend - opstart) / 60.0 if np.isfinite(opstart) and np.isfinite(opend) and opend >= opstart else np.nan

    try:
        bis = vdb.numeric(caseid, "BIS/BIS")
        sqi = vdb.numeric(caseid, "BIS/SQI")
        sr = vdb.numeric(caseid, "BIS/SR")
        propofol_rate = vdb.numeric(caseid, "Orchestra/PPF20_RATE")
        tci_start = base.first_time(propofol_rate, lambda value: value > 0)
        first_bis60 = base.first_time(bis, lambda value: value <= 60, after=tci_start)
        result.update(
            propofol_tci_start_sec=tci_start,
            first_bis_le60_after_tci_sec=first_bis60,
            first_bis_le60_is_loc_proxy_only=1,
        )
        if not np.isfinite(tci_start):
            result["status"] = "no_positive_propofol_tci_rate"
            return result, [], []
        if not np.isfinite(first_bis60):
            result["status"] = "no_bis_le60_after_tci"
            return result, [], []

        target, window_start, window_end = fixed_window_times(first_bis60, cfg)
        outcome_start = window_end + cfg.sr_memory_lag_sec
        result.update(
            fixed_window_target_sec=target,
            fixed_window_start_sec=window_start,
            fixed_window_end_sec=window_end,
            fixed_window_grid_alignment_delay_sec=window_start - target,
            fixed_window_duration_sec=cfg.window_sec,
            post_index_observation_start_sec=outcome_start,
            post_index_observation_end_sec=opend,
            post_index_observation_duration_min=max(0.0, (opend - outcome_start) / 60.0) if np.isfinite(opend) else np.nan,
            alternative_window_searched=0,
        )
        if not np.isfinite(aneend) or window_end > aneend:
            result["status"] = "fixed_window_not_completed_before_anesthesia_end"
            return result, [], []

        raw1 = vdb.waveform(caseid, "BIS/EEG1_WAV")
        raw2 = vdb.waveform(caseid, "BIS/EEG2_WAV")
        result["eeg1_total_sample_n"] = int(len(raw1))
        result["eeg2_total_sample_n"] = int(len(raw2))
        result["eeg_channels_time_synchronized_by_vitaldb"] = 1
        filtered1 = filter_wave(raw1, cfg)
        filtered2 = filter_wave(raw2, cfg)
        qc1 = channel_window_qc(raw1, filtered1, window_start, cfg)
        qc2 = channel_window_qc(raw2, filtered2, window_start, cfg)
        result.update({f"eeg1_{key}": value for key, value in qc1.items()})
        result.update({f"eeg2_{key}": value for key, value in qc2.items()})
        result.update(low_amplitude_review_flags(filtered1, filtered2, window_start, cfg))
        result.update(sqi_summary(sqi, window_start, window_end))
        technical_pass = bool(qc1["provisional_spectrum_technical_pass"] and qc2["provisional_spectrum_technical_pass"])
        if technical_pass:
            result.update(spectra_for_window(filtered1, filtered2, window_start, cfg))
        else:
            result["automated_spectral_available"] = 0

        result.update(prior_sr_flags(sr, first_bis60, outcome_start))
        result.update(manufacturer_sr_metrics(sr, outcome_start, opend, cfg))
        result["sr_outcome_end_is_operation_end"] = 1
        result["status"] = "manual_visual_review_pending" if technical_pass else "fixed_window_technically_nonevaluable"
        if not technical_pass:
            return result, [], []

        review_id = pseudonym("REV", caseid, salt)
        image_file = f"images/{review_id}.png"
        # The base renderer receives this config duck-typed; it does not show BIS SR or band power.
        base.make_review_image(
            out / "blinded_review" / image_file,
            review_id,
            raw1,
            raw2,
            filtered1,
            filtered2,
            sqi,
            window_start,
            cfg,
        )
        review = reviewer_row(review_id, image_file)
        key = {
            "review_id": review_id,
            "study_id": study_id,
            "protocol_version": PROTOCOL_VERSION,
            "fixed_window_start_sec": window_start,
            "fixed_window_end_sec": window_end,
            **{key: value for key, value in result.items() if key not in {"review_id"}},
        }
        result["review_id"] = review_id
        return result, [review], [key]
    except Exception as exc:
        result.update(status="processing_error", error=f"{type(exc).__name__}: {exc}")
        return result, [], []


def _load_records(path: Path) -> list[dict[str, Any]]:
    try:
        return pd.read_csv(path).to_dict("records") if path.exists() else []
    except pd.errors.EmptyDataError:
        return []


def write_csv_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_state(out: Path, cohort: dict[str, dict], reviews: dict[str, dict], keys: dict[str, dict]) -> None:
    rows = [cohort[key] for key in sorted(cohort)]
    write_csv_atomic(rows, out / "checkpoint.csv")
    write_csv_atomic(rows, out / "cohort.csv")
    write_csv_atomic(list(reviews.values()), out / "blinded_review" / "reviewer_form.csv")
    write_csv_atomic(list(keys.values()), out / "restricted_key.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--salt", default="VitalDB-Shao-adapted-fixed120-v1-public-pseudonym")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    vdb = base.VitalDB(cache)
    cases = eligible_cases(vdb).iloc[args.shard_index :: args.shard_count]
    if args.limit:
        cases = cases.head(args.limit)

    cohort_by_id: dict[str, dict] = {}
    reviews_by_id: dict[str, dict] = {}
    keys_by_id: dict[str, dict] = {}
    if args.resume:
        for record in _load_records(out / "checkpoint.csv"):
            cohort_by_id[str(record["study_id"])] = record
        for record in _load_records(out / "blinded_review" / "reviewer_form.csv"):
            reviews_by_id[str(record["review_id"])] = record
        for record in _load_records(out / "restricted_key.csv"):
            keys_by_id[str(record["review_id"])] = record
    done = {
        study_id for study_id, record in cohort_by_id.items()
        if record.get("status") != "processing_error"
    }

    processed = 0
    for ordinal, row in enumerate(cases.itertuples(index=False), 1):
        study_id = pseudonym("STUDY", int(row.caseid), args.salt)
        if study_id in done:
            continue
        result, review_rows, key_rows = process_case(vdb, row, out, args.salt, CFG)
        cohort_by_id[study_id] = result
        for record in review_rows:
            reviews_by_id[str(record["review_id"])] = record
        for record in key_rows:
            keys_by_id[str(record["review_id"])] = record
        processed += 1
        if processed % 5 == 0:
            write_state(out, cohort_by_id, reviews_by_id, keys_by_id)
            print(f"shard {args.shard_index}: source row {ordinal}/{len(cases)}", flush=True)
    write_state(out, cohort_by_id, reviews_by_id, keys_by_id)

    cohort_frame = pd.DataFrame(cohort_by_id.values())
    cases_path = cache / "cases.csv.gz"
    tracks_path = cache / "trks.csv.gz"
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_config": asdict(CFG),
        "protocol_config_sha256": config_hash(CFG),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "cases_assigned": int(len(cases)),
        "cohort_rows_written": int(len(cohort_frame)),
        "status_counts": cohort_frame.status.value_counts(dropna=False).to_dict() if len(cohort_frame) else {},
        "source": "Fresh VitalDB public Web API or byte-identical within-job cache",
        "source_cases_sha256": file_hash(cases_path) if cases_path.exists() else None,
        "source_tracks_sha256": file_hash(tracks_path) if tracks_path.exists() else None,
        "primary_band_hz": {"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 12], "beta": [13, 30]},
        "channel_derivation_primary": "samplewise equal-voltage average (EEG1+EEG2)/2 before PSD",
        "channel_derivation_sensitivity": "equal average of EEG1 and EEG2 linear band powers before dB",
        "window_is_fixed_not_searched": True,
        "visual_review_required_for_alpha_window_eligibility": True,
        "manual_blinded_review_completed": False,
        "analysis_cohort_finalized": False,
        "manufacturer_sr_manual_review_required": False,
        "clinical_outcomes_manual_eeg_review_required": False,
        "shao_citation": SHAO_CITATION,
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
