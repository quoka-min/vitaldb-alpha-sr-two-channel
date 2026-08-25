#!/usr/bin/env python3
"""VitalDB reanalysis with bilateral suppression-like screening (protocol v2).

This is a deliberately separate pipeline version.  It preserves the original
fresh-API implementation and implements the subsequently prespecified rule:

* synchronised BIS EEG1/EEG2 waveforms at 128 Hz;
* zero-phase 0.5--45 Hz band-pass filtering (no redundant 60-Hz notch);
* 120-s candidates composed of 60 non-overlapping 2-s epochs;
* BIS/SQI summarised over the whole candidate, never used as an independent
  per-epoch gate or as evidence that suppression is absent;
* <= +/-5 uV for at least 0.5 s is an automated ``suppression-like`` flag;
* a temporally overlapping bilateral flag of at least 0.5 s automatically
  rejects a candidate; a unilateral flag is retained for blinded review.

Automated output is *pending blinded review*.  It must never be labelled as a
completed manual review or as a final analysis cohort.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

import vitaldb_api_pipeline as base


PROTOCOL_VERSION = "bilateral-suppression-like-v2"
CFG = base.Config()
FILTER_SPEC = {
    "filter_design": "Butterworth band-pass",
    "filter_order": 4,
    "zero_phase": True,
    "bandpass_low_hz": 0.5,
    "bandpass_high_hz": 45.0,
    "notch_applied": False,
}


def applied_config() -> dict:
    """Return the actually applied configuration without a legacy notch field."""
    config = asdict(CFG)
    config.pop("notch_hz", None)
    config.update(FILTER_SPEC)
    return config


def filter_wave(raw: np.ndarray, cfg: base.Config = CFG) -> np.ndarray:
    """Apply a fourth-order Butterworth zero-phase 0.5--45 Hz band-pass.

    Missing samples are interpolated only to make continuous zero-phase
    filtering numerically possible.  Their positions are restored to NaN so
    that the original-sample completeness checks remain operative.
    """
    raw = np.asarray(raw, dtype=float)
    good = np.isfinite(raw)
    if good.sum() < cfg.fs * 10:
        return np.full_like(raw, np.nan)
    idx = np.arange(len(raw))
    filled = np.interp(idx, idx[good], raw[good])
    sos = butter(
        4,
        [cfg.bandpass_low_hz / (cfg.fs / 2), cfg.bandpass_high_hz / (cfg.fs / 2)],
        btype="band",
        output="sos",
    )
    out = sosfiltfilt(sos, filled)
    out[~good] = np.nan
    return out


def epoch_quality(raw: np.ndarray, filt: np.ndarray, start: float, cfg: base.Config = CFG) -> dict:
    """Return transparent operational QC results for one 2-s epoch."""
    a = max(0, int(round(start * cfg.fs)))
    b = min(len(raw), a + cfg.epoch_sec * cfg.fs)
    r = np.asarray(raw[a:b], float)
    x = np.asarray(filt[a:b], float)
    expected = cfg.epoch_sec * cfg.fs
    finite_fraction = float(np.isfinite(r).mean()) if len(r) else 0.0
    rv = r[np.isfinite(r)]
    xv = x[np.isfinite(x)]
    abs_max = float(np.max(np.abs(rv))) if rv.size else np.nan
    peak_to_peak = float(np.ptp(rv)) if rv.size else np.nan
    filtered_sd = float(np.std(xv)) if xv.size else np.nan
    fail_missing = bool(len(r) != expected or finite_fraction < cfg.finite_fraction_min)
    fail_abs = bool(not rv.size or abs_max > cfg.abs_amplitude_max_uv)
    fail_ptp = bool(not rv.size or peak_to_peak > cfg.peak_to_peak_max_uv)
    fail_flatline = bool(not xv.size or filtered_sd < cfg.flat_sd_min_uv)
    return {
        "fail_missing": fail_missing,
        "fail_abs_amplitude": fail_abs,
        "fail_peak_to_peak": fail_ptp,
        "fail_flatline": fail_flatline,
        "pass": not (fail_missing or fail_abs or fail_ptp or fail_flatline),
        "finite_fraction": finite_fraction,
        "abs_max_uv": abs_max,
        "peak_to_peak_uv": peak_to_peak,
        "filtered_sd_uv": filtered_sd,
    }


def _runs(mask: np.ndarray, offset_sample: int, cfg: base.Config = CFG) -> list[dict]:
    delta = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(delta == 1)
    ends = np.flatnonzero(delta == -1)
    min_samples = int(math.ceil(cfg.suppression_min_sec * cfg.fs))
    out = []
    for a, b in zip(starts, ends):
        if b - a >= min_samples:
            out.append(
                {
                    "start_sec": float((offset_sample + a) / cfg.fs),
                    "end_sec": float((offset_sample + b) / cfg.fs),
                    "duration_sec": float((b - a) / cfg.fs),
                }
            )
    return out


def suppression_like_summary(
    f1: np.ndarray, f2: np.ndarray, start: float, end: float, cfg: base.Config = CFG
) -> dict:
    """Summarise channel-wise and simultaneous bilateral low-amplitude runs."""
    a = max(0, int(round(start * cfg.fs)))
    b = min(len(f1), len(f2), int(round(end * cfg.fs)))
    x1 = np.asarray(f1[a:b], float)
    x2 = np.asarray(f2[a:b], float)
    low1 = np.isfinite(x1) & (np.abs(x1) <= cfg.suppression_amplitude_uv)
    low2 = np.isfinite(x2) & (np.abs(x2) <= cfg.suppression_amplitude_uv)
    ch1 = _runs(low1, a, cfg)
    ch2 = _runs(low2, a, cfg)
    bilateral = _runs(low1 & low2, a, cfg)
    return {
        "suppression_like_ch1_n": len(ch1),
        "suppression_like_ch2_n": len(ch2),
        "suppression_like_bilateral_n": len(bilateral),
        "suppression_like_ch1_sec": float(sum(r["duration_sec"] for r in ch1)),
        "suppression_like_ch2_sec": float(sum(r["duration_sec"] for r in ch2)),
        "suppression_like_bilateral_sec": float(sum(r["duration_sec"] for r in bilateral)),
        "unilateral_suppression_like_flag": int(bool(ch1 or ch2) and not bilateral),
        "bilateral_suppression_like_flag": int(bool(bilateral)),
        "ch1_runs": ch1,
        "ch2_runs": ch2,
        "bilateral_runs": bilateral,
    }


def sqi_window_summary(sqi: pd.DataFrame, start: float, end: float, cfg: base.Config = CFG) -> dict:
    """Summarise the single manufacturer SQI track over the whole window."""
    d = sqi.loc[sqi.time.between(start, end, inclusive="left"), ["time", "value"]].copy()
    d["value"] = pd.to_numeric(d.value, errors="coerce")
    d = d.loc[np.isfinite(d.value)]
    v = d.value.to_numpy(float)
    observed_seconds = len(np.unique(np.floor(d.time.to_numpy(float)).astype(int))) if len(d) else 0
    return {
        "sqi_window_n_obs": int(len(v)),
        "sqi_window_coverage_fraction": float(min(1.0, observed_seconds / cfg.window_sec)),
        "sqi_window_min": float(np.min(v)) if len(v) else np.nan,
        "sqi_window_p10": float(np.percentile(v, 10)) if len(v) else np.nan,
        "sqi_window_median": float(np.median(v)) if len(v) else np.nan,
        "sqi_window_mean": float(np.mean(v)) if len(v) else np.nan,
        "sqi_window_fraction_ge90": float(np.mean(v >= cfg.sqi_window_threshold)) if len(v) else np.nan,
        # Prespecified sensitivity subsets; neither flag gates the primary screen.
        "sqi_sensitivity_median_ge90": int(bool(len(v) and np.median(v) >= cfg.sqi_window_threshold)),
        "sqi_sensitivity_all_observed_ge90": int(bool(len(v) and np.all(v >= cfg.sqi_window_threshold))),
    }


def technical_candidates(raw1, raw2, f1, f2, sqi, anchor, incision, cfg: base.Config = CFG):
    """Return first eligible candidates and an auditable nonexclusive reject tally."""
    grid0 = math.ceil(anchor / cfg.search_step_sec) * cfg.search_step_sec
    starts = np.arange(grid0, incision - cfg.window_sec + 1e-9, cfg.search_step_sec)
    epoch_cache: dict[float, tuple[dict, dict]] = {}
    audit = {
        "search_window_n": 0,
        "rejected_any_technical_n": 0,
        "rejected_missing_n": 0,
        "rejected_abs_amplitude_n": 0,
        "rejected_peak_to_peak_n": 0,
        "rejected_flatline_n": 0,
        "rejected_bilateral_suppression_like_n": 0,
        "accepted_for_blinded_review_n": 0,
        "accepted_unilateral_suppression_like_n": 0,
    }
    accepted: list[dict] = []
    last = -np.inf
    for s in starts:
        if s < last + cfg.visual_candidate_separation_sec:
            continue
        audit["search_window_n"] += 1
        times = [float(s + i * cfg.epoch_sec) for i in range(cfg.window_sec // cfg.epoch_sec)]
        pairs = []
        for t in times:
            if t not in epoch_cache:
                epoch_cache[t] = (epoch_quality(raw1, f1, t, cfg), epoch_quality(raw2, f2, t, cfg))
            pairs.append(epoch_cache[t])
        flags = {
            name: any(q[name] for pair in pairs for q in pair)
            for name in ["fail_missing", "fail_abs_amplitude", "fail_peak_to_peak", "fail_flatline"]
        }
        if any(flags.values()):
            audit["rejected_any_technical_n"] += 1
            for name, key in [
                ("fail_missing", "rejected_missing_n"),
                ("fail_abs_amplitude", "rejected_abs_amplitude_n"),
                ("fail_peak_to_peak", "rejected_peak_to_peak_n"),
                ("fail_flatline", "rejected_flatline_n"),
            ]:
                audit[key] += int(flags[name])
            continue
        e = float(s + cfg.window_sec)
        suppression = suppression_like_summary(f1, f2, float(s), e, cfg)
        if suppression["bilateral_suppression_like_flag"]:
            audit["rejected_bilateral_suppression_like_n"] += 1
            continue
        candidate = {
            "start_sec": float(s),
            "end_sec": e,
            **{k: v for k, v in suppression.items() if not k.endswith("_runs")},
            **sqi_window_summary(sqi, float(s), e, cfg),
        }
        candidate["unilateral_manual_review_pending"] = candidate["unilateral_suppression_like_flag"]
        accepted.append(candidate)
        audit["accepted_for_blinded_review_n"] += 1
        audit["accepted_unilateral_suppression_like_n"] += candidate["unilateral_suppression_like_flag"]
        last = float(s)
        if len(accepted) >= cfg.max_visual_candidates:
            break
    return accepted, audit


def _fill_short_missing(epoch: np.ndarray) -> np.ndarray:
    x = np.asarray(epoch, float)
    good = np.isfinite(x)
    if good.all():
        return x
    if not good.any():
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[good], x[good])


def spectra_for_window(f1, f2, start, cfg: base.Config = CFG):
    """Calculate band powers, interpolating only QC-permitted missing samples."""
    bands = {
        "delta": (cfg.delta_low_hz, cfg.delta_high_hz),
        "theta": (cfg.theta_low_hz, cfg.theta_high_hz),
        "alpha": (cfg.alpha_low_hz, cfg.alpha_high_hz),
        "beta": (cfg.beta_low_hz, cfg.beta_high_hz),
    }
    powers = {name: [] for name in bands}
    interpolated = 0
    total = 0
    for i in range(cfg.window_sec // cfg.epoch_sec):
        a = int(round((start + i * cfg.epoch_sec) * cfg.fs))
        b = a + cfg.epoch_sec * cfg.fs
        e1 = np.asarray(f1[a:b], float)
        e2 = np.asarray(f2[a:b], float)
        interpolated += int((~np.isfinite(e1)).sum() + (~np.isfinite(e2)).sum())
        total += len(e1) + len(e2)
        e1 = _fill_short_missing(e1)
        e2 = _fill_short_missing(e2)
        freq, p1 = base.epoch_psd(e1, cfg)
        _, p2 = base.epoch_psd(e2, cfg)
        for name, (lo, hi) in bands.items():
            m = (freq >= lo) & (freq <= hi)
            powers[name].append((float(np.trapezoid(p1[m], freq[m])), float(np.trapezoid(p2[m], freq[m]))))
    out = {"psd_interpolated_sample_fraction": float(interpolated / total) if total else np.nan}
    for name, vals in powers.items():
        p = np.asarray(vals)
        out[f"{name}_db_eeg1"] = float(10 * np.log10(np.mean(p[:, 0])))
        out[f"{name}_db_eeg2"] = float(10 * np.log10(np.mean(p[:, 1])))
        out[f"{name}_db_two_channel"] = float(10 * np.log10(np.mean(p)))
    return out


def process_case(vdb: base.VitalDB, row, out: Path, salt: str, cfg: base.Config = CFG):
    cid = int(row.caseid)
    result = {"caseid": cid, "status": "processing", "protocol_version": PROTOCOL_VERSION}
    try:
        bis = vdb.numeric(cid, "BIS/BIS")
        sqi = vdb.numeric(cid, "BIS/SQI")
        sr = vdb.numeric(cid, "BIS/SR")
        ce = vdb.numeric(cid, "Orchestra/PPF20_CE")
        tci = base.first_time(ce, lambda x: x > 0)
        anchor = base.first_time(bis, lambda x: x <= 60, after=tci)
        result.update(tci_start_sec=tci, first_bis60_sec=anchor)
        if not np.isfinite(anchor):
            result.update(status="no_first_bis60")
            return result, [], []
        incision = base.finite_num(row.opstart)
        aneend = base.finite_num(row.aneend)
        if not np.isfinite(incision) or incision - anchor < cfg.window_sec:
            result.update(status="no_preincision_room")
            return result, [], []
        raw1 = vdb.waveform(cid, "BIS/EEG1_WAV")
        raw2 = vdb.waveform(cid, "BIS/EEG2_WAV")
        f1 = filter_wave(raw1, cfg)
        f2 = filter_wave(raw2, cfg)
        candidates, audit = technical_candidates(raw1, raw2, f1, f2, sqi, anchor, incision, cfg)
        result.update(audit)
        if not candidates:
            result.update(status="no_technical_candidate")
            return result, [], []
        reviews = []
        keys = []
        for order, candidate in enumerate(candidates, 1):
            start = candidate["start_sec"]
            end = candidate["end_sec"]
            rid = base.review_id(cid, order, salt)
            spectra = spectra_for_window(f1, f2, start, cfg)
            metrics = base.outcome_metrics(sr, sqi, anchor, start, end, aneend, cfg)
            image = f"images/{rid}.png"
            base.make_review_image(out / "blinded_review" / image, rid, raw1, raw2, f1, f2, sqi, start, cfg)
            reviews.append(
                {
                    "review_id": rid,
                    "candidate_order": order,
                    "image_file": image,
                    "decision": "",
                    "artifact_present": "",
                    "physiologic_suppression_present": "",
                    "transition_present": "",
                    "both_channels_interpretable": "",
                    "reviewer_confidence": "",
                    "comment": "",
                }
            )
            keys.append(
                {
                    "review_id": rid,
                    "caseid": cid,
                    "candidate_order": order,
                    "candidate_start_sec": start,
                    "candidate_end_sec": end,
                    **{k: v for k, v in candidate.items() if k not in {"start_sec", "end_sec"}},
                    **spectra,
                    **metrics,
                }
            )
        first = keys[0]
        keep = [
            "candidate_start_sec",
            "candidate_end_sec",
            "psd_interpolated_sample_fraction",
            "suppression_like_ch1_n",
            "suppression_like_ch2_n",
            "suppression_like_bilateral_n",
            "suppression_like_ch1_sec",
            "suppression_like_ch2_sec",
            "suppression_like_bilateral_sec",
            "unilateral_suppression_like_flag",
            "unilateral_manual_review_pending",
            "bilateral_suppression_like_flag",
            "sqi_window_n_obs",
            "sqi_window_coverage_fraction",
            "sqi_window_min",
            "sqi_window_p10",
            "sqi_window_median",
            "sqi_window_mean",
            "sqi_window_fraction_ge90",
            "sqi_sensitivity_median_ge90",
            "sqi_sensitivity_all_observed_ge90",
        ] + [
            f"{band}_db_{channel}"
            for band in ["delta", "theta", "alpha", "beta"]
            for channel in ["eeg1", "eeg2", "two_channel"]
        ] + [
            "pre_sr10",
            "post_outcome_start_sec",
            "post_outcome_end_sec",
            "post_valid_sec",
            "post_sr10",
            "post_sr10_event_count",
            "post_max_sr",
            "post_mean_sr",
            "post_sr_auc_percent_min",
            "post_classifiable",
        ]
        result.update({k: first.get(k) for k in keep})
        result.update(
            status="visual_review_pending",
            review_id=first["review_id"],
            visual_candidate_n=len(keys),
            manual_blinded_review_completed=0,
        )
        for c in [
            "age", "sex", "height", "weight", "bmi", "asa", "emop", "department", "optype",
            "anestart", "aneend", "opstart", "opend", "preop_cr", "preop_alb", "preop_hb",
            "icu_days", "death_inhosp",
        ]:
            if hasattr(row, c):
                result[c] = getattr(row, c)
        return result, reviews, keys
    except Exception as exc:
        result.update(status="processing_error", error=f"{type(exc).__name__}: {exc}")
        return result, [], []


def write_csv_atomic(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _load_records(path: Path) -> list[dict]:
    try:
        return pd.read_csv(path).to_dict("records") if path.exists() else []
    except pd.errors.EmptyDataError:
        return []


def _write_state(out: Path, cohort_by_case: dict, reviews_by_id: dict, keys_by_id: dict):
    rows = [cohort_by_case[k] for k in sorted(cohort_by_case)]
    write_csv_atomic(rows, out / "checkpoint.csv")
    write_csv_atomic(rows, out / "cohort.csv")
    write_csv_atomic(list(reviews_by_id.values()), out / "blinded_review" / "reviewer_form.csv")
    write_csv_atomic(list(keys_by_id.values()), out / "restricted_key.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--salt", default="VitalDB-alpha-SR-blind-bilateral-v2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vdb = base.VitalDB(Path(args.cache))
    cases = base.cohort(vdb).iloc[args.shard_index :: args.shard_count]
    if args.limit:
        cases = cases.head(args.limit)
    cohort_by_case: dict[int, dict] = {}
    reviews_by_id: dict[str, dict] = {}
    keys_by_id: dict[str, dict] = {}
    if args.resume:
        for r in _load_records(out / "checkpoint.csv"):
            cohort_by_case[int(r["caseid"])] = r
        for r in _load_records(out / "blinded_review" / "reviewer_form.csv"):
            reviews_by_id[str(r["review_id"])] = r
        for r in _load_records(out / "restricted_key.csv"):
            keys_by_id[str(r["review_id"])] = r
    done = {cid for cid, r in cohort_by_case.items() if r.get("status") != "processing_error"}
    processed_since_write = 0
    for ordinal, row in enumerate(cases.itertuples(index=False), 1):
        cid = int(row.caseid)
        if cid in done:
            continue
        result, reviews, keys = process_case(vdb, row, out, args.salt, CFG)
        cohort_by_case[cid] = result
        for r in reviews:
            reviews_by_id[str(r["review_id"])] = r
        for r in keys:
            keys_by_id[str(r["review_id"])] = r
        processed_since_write += 1
        if processed_since_write % 5 == 0:
            _write_state(out, cohort_by_case, reviews_by_id, keys_by_id)
            print(f"shard {args.shard_index}: source-row {ordinal}/{len(cases)}", flush=True)
    _write_state(out, cohort_by_case, reviews_by_id, keys_by_id)
    cohort_df = pd.DataFrame(cohort_by_case.values())
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "cohort_rows": int(len(cohort_df)),
        "status_counts": cohort_df.status.value_counts(dropna=False).to_dict() if len(cohort_df) else {},
        "review_rows": int(len(reviews_by_id)),
        "manual_blinded_review_completed": False,
        "candidate_sqi_is_window_summary_only": True,
        "candidate_sqi_used_as_epoch_gate": False,
        "bilateral_suppression_like_auto_rejected": True,
        "unilateral_suppression_like_auto_rejected": False,
        **FILTER_SPEC,
    }
    for c in [
        "search_window_n", "rejected_any_technical_n", "rejected_missing_n",
        "rejected_abs_amplitude_n", "rejected_peak_to_peak_n", "rejected_flatline_n",
        "rejected_bilateral_suppression_like_n", "accepted_for_blinded_review_n",
        "accepted_unilateral_suppression_like_n",
    ]:
        summary[f"sum_{c}"] = int(pd.to_numeric(cohort_df.get(c), errors="coerce").fillna(0).sum()) if c in cohort_df else 0
    (out / "screening_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "config": applied_config(),
        **FILTER_SPEC,
        "filter_implementation": "scipy.signal.butter(output='sos') + scipy.signal.sosfiltfilt",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "cases_assigned": len(cases),
        "source": "Fresh VitalDB public API or byte-identical cached API track files",
        "visual_review_required": True,
        "manual_blinded_review_completed": False,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
