#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from vitaldb_api_pipeline import CFG, VitalDB, filter_wave, quality_valid_sr


def make_image(path: Path, validation_id: str, raw1: np.ndarray, raw2: np.ndarray,
               sqi: pd.DataFrame, start: float, end: float) -> None:
    fs = CFG.fs; a = max(0, int(start * fs)); b = min(len(raw1), int(end * fs))
    f1 = filter_wave(raw1, CFG); f2 = filter_wave(raw2, CFG)
    t = np.arange(a, b) / fs - end
    fig, ax = plt.subplots(5, 1, figsize=(12, 9), sharex=True,
                           gridspec_kw={"height_ratios": [1.1, 1.1, 1.5, 1.5, .7]})
    ax[0].plot(t, raw1[a:b], lw=.3, color="#17365D"); ax[0].set_ylabel("EEG1 raw\nµV")
    ax[1].plot(t, raw2[a:b], lw=.3, color="#2F75B5"); ax[1].set_ylabel("EEG2 raw\nµV")
    for j, x in enumerate([f1[a:b], f2[a:b]], start=2):
        f, tt, p = spectrogram(np.nan_to_num(x), fs=fs, nperseg=fs*2, noverlap=fs, scaling="density")
        mask = (f >= .5) & (f <= 30)
        ax[j].pcolormesh(tt + t[0], f[mask], 10*np.log10(p[mask] + 1e-12), shading="auto",
                         cmap="viridis", vmin=-20, vmax=30)
        ax[j].set_ylim(.5, 30); ax[j].set_ylabel(f"EEG{j-1}\nHz")
    q = sqi.loc[sqi.time.between(start, end)]
    ax[4].plot(q.time - end, q.value, color="#548235", lw=1)
    ax[4].axhline(80, color="#C65911", ls="--", lw=.8); ax[4].set_ylim(0, 105)
    ax[4].set_ylabel("SQI"); ax[4].set_xlabel("Seconds before sampled manufacturer-SR time")
    for x in ax: x.grid(alpha=.15)
    fig.suptitle(f"Blinded SR validation {validation_id} | review the displayed 63-second interval", fontsize=12)
    checklist = ("Record: (1) both channels interpretable Y/N; (2) artifact Y/N; "
                 "(3) physiologic suppression Y/N/uncertain; (4) suppression fraction ≥10% Y/N/uncertain;\n"
                 "(5) category: 0, <5%, 5–<10%, 10–<20%, ≥20%; (6) confidence 1–5. Do not view manufacturer SR before locking decisions.")
    fig.text(.01, .004, checklist, fontsize=8.2, ha="left", va="bottom")
    fig.tight_layout(rect=[0, .035, 1, .965]); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True); ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.manifest).iloc[a.shard_index::a.shard_count].copy()
    vdb = VitalDB(Path(a.cache)); key_rows = []; failures = []
    for n, r in enumerate(d.itertuples(index=False), 1):
        try:
            cid = int(r.caseid); sr = vdb.numeric(cid, "BIS/SR"); sqi = vdb.numeric(cid, "BIS/SQI")
            raw1 = vdb.waveform(cid, "BIS/EEG1_WAV"); raw2 = vdb.waveform(cid, "BIS/EEG2_WAV")
            valid = quality_valid_sr(sr, sqi, float(r.post_outcome_start_sec), float(r.post_outcome_end_sec), CFG)
            pv = valid.loc[valid.valid & np.isfinite(valid.value)].copy()
            if pv.empty: raise RuntimeError("No quality-valid SR sample in outcome interval")
            peak = pv.loc[pv.value.idxmax()]
            end = float(peak.time); start = max(float(r.post_outcome_start_sec), end - CFG.outcome_memory_sec)
            if end - start < CFG.outcome_memory_sec - 1: raise RuntimeError("Less than 62 seconds available")
            make_image(out / "images" / f"{r.validation_id}.png", r.validation_id, raw1, raw2, sqi, start, end)
            key_rows.append({"validation_id": r.validation_id, "validation_set": r.validation_set, "caseid": cid,
                             "segment_start_sec": start, "segment_end_sec": end,
                             "manufacturer_sr_at_sample_time": float(peak.value),
                             "manufacturer_segment_sr_ge10": int(float(peak.value) >= 10),
                             "manufacturer_case_any_sr_ge10": int(r.post_sr10),
                             "manufacturer_case_post_max_sr": float(r.post_max_sr),
                             "manufacturer_case_post_mean_sr": float(r.post_mean_sr),
                             "pre_window_sr_ge10": int(r.pre_sr10)})
        except Exception as exc:
            failures.append({"validation_id": r.validation_id, "caseid": int(r.caseid), "error": f"{type(exc).__name__}: {exc}"})
        if n % 5 == 0: print(f"shard {a.shard_index}: {n}/{len(d)}", flush=True)
    pd.DataFrame(key_rows).to_csv(out / "restricted_manufacturer_key.csv", index=False)
    pd.DataFrame(failures).to_csv(out / "failures.csv", index=False)
    (out / "run_manifest.json").write_text(json.dumps({"shard_index": a.shard_index, "shard_count": a.shard_count,
        "assigned": len(d), "completed": len(key_rows), "failed": len(failures),
        "reference_standard": "Blinded human raw-EEG adjudication; manufacturer SR is index test, not truth"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
