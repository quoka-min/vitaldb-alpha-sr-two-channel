#!/usr/bin/env python3
"""Fresh VitalDB API extraction for early post-induction alpha and later SR.

The pipeline downloads every required track from the public VitalDB API. It
does not reuse alpha, BIS-anchor, or SR variables from the legacy REVISION2
package. Automated screening produces visually pending candidate windows;
manual blinded review remains required for transition dynamics.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt, spectrogram
from scipy.signal.windows import dpss


API = "https://api.vitaldb.net"
TRACKS = ["BIS/EEG1_WAV", "BIS/EEG2_WAV", "BIS/BIS", "BIS/SQI", "BIS/SR", "Orchestra/PPF20_CE"]


@dataclass(frozen=True)
class Config:
    fs: int = 128
    epoch_sec: int = 2
    window_sec: int = 120
    search_step_sec: int = 2
    sqi_window_threshold: float = 90
    sqi_outcome_threshold: float = 80
    finite_fraction_min: float = 0.98
    abs_amplitude_max_uv: float = 500
    peak_to_peak_max_uv: float = 800
    flat_sd_min_uv: float = 0.5
    suppression_amplitude_uv: float = 5
    suppression_min_sec: float = 0.5
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 45
    notch_hz: float = 60
    delta_low_hz: float = 0.5
    delta_high_hz: float = 4
    theta_low_hz: float = 4
    theta_high_hz: float = 8
    alpha_low_hz: float = 8
    alpha_high_hz: float = 13
    beta_low_hz: float = 13
    beta_high_hz: float = 30
    time_bandwidth: float = 3
    tapers: int = 5
    outcome_memory_sec: int = 63
    outcome_quality_coverage: float = 0.90
    outcome_quality_good_fraction: float = 0.90
    negative_valid_min_sec: int = 600
    max_visual_candidates: int = 2
    visual_candidate_separation_sec: int = 120
    http_retries: int = 5


CFG = Config()


def fetch(url: str, cache: Path, cfg: Config = CFG) -> bytes:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.stat().st_size:
        return cache.read_bytes()
    last = None
    for attempt in range(cfg.http_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VitalDB-alpha-SR-research/1.0"})
            with urllib.request.urlopen(req, timeout=120) as response:
                payload = response.read()
            if not payload:
                raise IOError("empty response")
            tmp = cache.with_suffix(cache.suffix + ".tmp")
            tmp.write_bytes(payload)
            os.replace(tmp, cache)
            return payload
        except Exception as exc:
            last = exc
            time.sleep((2 ** attempt) + random.random())
    raise RuntimeError(f"download failed {url}: {last}")


def csv_payload(payload: bytes) -> pd.DataFrame:
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    return pd.read_csv(io.BytesIO(payload))


class VitalDB:
    def __init__(self, cache: Path):
        self.cache = cache
        self.cases = csv_payload(fetch(f"{API}/cases", cache / "cases.csv.gz"))
        self.trks = csv_payload(fetch(f"{API}/trks", cache / "trks.csv.gz"))
        self.lookup = {(int(r.caseid), str(r.tname)): str(r.tid) for r in self.trks.itertuples()}

    def has(self, caseid: int, track: str) -> bool:
        return (int(caseid), track) in self.lookup

    def track(self, caseid: int, track: str) -> pd.DataFrame:
        tid = self.lookup.get((int(caseid), track))
        if not tid:
            return pd.DataFrame(columns=["time", "value"])
        raw = fetch(f"{API}/{tid}", self.cache / "tracks" / f"{tid}.csv.gz")
        frame = csv_payload(raw).iloc[:, :2].copy()
        frame.columns = ["time", "value"]
        frame["time"] = pd.to_numeric(frame.time, errors="coerce")
        frame["value"] = pd.to_numeric(frame.value, errors="coerce")
        return frame

    def numeric(self, caseid: int, track: str) -> pd.DataFrame:
        d = self.track(caseid, track).dropna(subset=["time"]).sort_values("time")
        return d.drop_duplicates("time", keep="last").reset_index(drop=True)

    def waveform(self, caseid: int, track: str) -> np.ndarray:
        return self.track(caseid, track).value.to_numpy(dtype=float)


def finite_num(value, default=np.nan):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def first_time(track: pd.DataFrame, predicate, after: float = -np.inf) -> float:
    d = track.loc[(track.time >= after) & predicate(track.value)]
    return float(d.time.iloc[0]) if len(d) else np.nan


def filter_wave(raw: np.ndarray, cfg: Config = CFG) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    good = np.isfinite(raw)
    if good.sum() < cfg.fs * 10:
        return np.full_like(raw, np.nan)
    idx = np.arange(len(raw))
    filled = np.interp(idx, idx[good], raw[good])
    sos = butter(4, [cfg.bandpass_low_hz/(cfg.fs/2), cfg.bandpass_high_hz/(cfg.fs/2)], btype="band", output="sos")
    b, a = iirnotch(cfg.notch_hz/(cfg.fs/2), Q=30)
    out = filtfilt(b, a, sosfiltfilt(sos, filled))
    out[~good] = np.nan
    return out


def sqi_epoch_ok(sqi: pd.DataFrame, start: float, end: float, cfg: Config = CFG) -> bool:
    v = sqi.loc[sqi.time.between(start, end, inclusive="left"), "value"].to_numpy(float)
    v = v[np.isfinite(v)]
    return bool(v.size >= 1 and np.all(v >= cfg.sqi_window_threshold))


def raw_epoch_ok(raw: np.ndarray, filt: np.ndarray, start: float, cfg: Config = CFG) -> bool:
    a = max(0, int(round(start * cfg.fs)))
    b = min(len(raw), a + cfg.epoch_sec * cfg.fs)
    r, x = raw[a:b], filt[a:b]
    if len(r) != cfg.epoch_sec * cfg.fs or np.isfinite(r).mean() < cfg.finite_fraction_min:
        return False
    rv = r[np.isfinite(r)]
    xv = x[np.isfinite(x)]
    return bool(rv.size and xv.size and np.max(np.abs(rv)) <= cfg.abs_amplitude_max_uv and np.ptp(rv) <= cfg.peak_to_peak_max_uv and np.std(xv) >= cfg.flat_sd_min_uv)


def has_suppression(filt: np.ndarray, start: float, end: float, cfg: Config = CFG) -> bool:
    a, b = int(round(start*cfg.fs)), int(round(end*cfg.fs))
    x = filt[max(0,a):min(len(filt),b)]
    low = np.isfinite(x) & (np.abs(x) <= cfg.suppression_amplitude_uv)
    delta = np.diff(np.r_[False, low, False].astype(np.int8))
    starts, ends = np.flatnonzero(delta == 1), np.flatnonzero(delta == -1)
    return bool(np.any((ends-starts) >= int(cfg.suppression_min_sec*cfg.fs)))


def technical_candidates(raw1, raw2, f1, f2, sqi, anchor, incision, cfg: Config = CFG):
    grid0 = math.ceil(anchor/cfg.search_step_sec)*cfg.search_step_sec
    starts = np.arange(grid0, incision-cfg.window_sec+1e-9, cfg.search_step_sec)
    epoch_ok = {}
    def ok_epoch(t):
        if t not in epoch_ok:
            epoch_ok[t] = sqi_epoch_ok(sqi,t,t+cfg.epoch_sec,cfg) and raw_epoch_ok(raw1,f1,t,cfg) and raw_epoch_ok(raw2,f2,t,cfg)
        return epoch_ok[t]
    accepted=[]
    last=-np.inf
    for s in starts:
        if s < last + cfg.visual_candidate_separation_sec:
            continue
        epochs=[s+i*cfg.epoch_sec for i in range(cfg.window_sec//cfg.epoch_sec)]
        if not all(ok_epoch(float(t)) for t in epochs):
            continue
        e=s+cfg.window_sec
        if has_suppression(f1,s,e,cfg) or has_suppression(f2,s,e,cfg):
            continue
        accepted.append(float(s)); last=float(s)
        if len(accepted)>=cfg.max_visual_candidates:
            break
    return accepted


def epoch_psd(epoch: np.ndarray, cfg: Config = CFG):
    x=np.asarray(epoch,float); x=x-np.mean(x)
    taps=dpss(len(x),cfg.time_bandwidth,cfg.tapers)
    psd=np.mean([np.abs(np.fft.rfft(x*t))**2/(cfg.fs*np.sum(t*t)) for t in taps],axis=0)
    if len(psd)>2: psd[1:-1]*=2
    f=np.fft.rfftfreq(len(x),1/cfg.fs)
    return f, psd


def spectra_for_window(raw1, raw2, f1, f2, start, cfg: Config = CFG):
    bands={"delta":(cfg.delta_low_hz,cfg.delta_high_hz),"theta":(cfg.theta_low_hz,cfg.theta_high_hz),"alpha":(cfg.alpha_low_hz,cfg.alpha_high_hz),"beta":(cfg.beta_low_hz,cfg.beta_high_hz)}
    powers={name:[] for name in bands}
    for i in range(cfg.window_sec//cfg.epoch_sec):
        a=int(round((start+i*cfg.epoch_sec)*cfg.fs)); b=a+cfg.epoch_sec*cfg.fs
        freq,p1=epoch_psd(f1[a:b],cfg); _,p2=epoch_psd(f2[a:b],cfg)
        for name,(lo,hi) in bands.items():
            m=(freq>=lo)&(freq<=hi)
            powers[name].append((float(np.trapezoid(p1[m],freq[m])),float(np.trapezoid(p2[m],freq[m]))))
    out={}
    for name,vals in powers.items():
        p=np.asarray(vals)
        out[f"{name}_db_eeg1"]=float(10*np.log10(np.mean(p[:,0])))
        out[f"{name}_db_eeg2"]=float(10*np.log10(np.mean(p[:,1])))
        out[f"{name}_db_two_channel"]=float(10*np.log10(np.mean(p)))
    return out


def quality_valid_sr(sr: pd.DataFrame, sqi: pd.DataFrame, start: float, end: float, cfg: Config = CFG):
    d=sr.loc[sr.time.between(start,end,inclusive="both")].copy()
    if d.empty: return d.assign(valid=False)
    qt=sqi.time.to_numpy(float); qv=sqi.value.to_numpy(float)
    valid=[]
    for t in d.time.to_numpy(float):
        m=(qt>t-cfg.outcome_memory_sec)&(qt<=t)&np.isfinite(qv)
        vals=qv[m]
        coverage=min(1.0,len(np.unique(np.rint(qt[m]).astype(int)))/cfg.outcome_memory_sec) if vals.size else 0
        good=float(np.mean(vals>=cfg.sqi_outcome_threshold)) if vals.size else 0
        current=vals[-1] if vals.size else np.nan
        valid.append(coverage>=cfg.outcome_quality_coverage and good>=cfg.outcome_quality_good_fraction and current>=cfg.sqi_outcome_threshold)
    d["valid"]=valid
    return d


def outcome_metrics(sr,sqi,anchor,wstart,wend,aneend,cfg: Config=CFG):
    pre=quality_valid_sr(sr,sqi,anchor,wstart,cfg)
    post_start=wend+cfg.outcome_memory_sec
    post=quality_valid_sr(sr,sqi,post_start,aneend,cfg)
    pv=post.loc[post.valid & np.isfinite(post.value)].copy()
    above=(pv.value.to_numpy(float)>=10) if len(pv) else np.array([],bool)
    event_count=int(np.sum(above & ~np.r_[False,above[:-1]])) if len(above) else 0
    return {
      "pre_sr10":int(bool(((pre.valid)&(pre.value>=10)).any())) if len(pre) else 0,
      "post_outcome_start_sec":post_start,"post_outcome_end_sec":aneend,
      "post_valid_sec":int(len(pv)),"post_sr10":int(bool(above.any())) if len(above) else 0,
      "post_sr10_event_count":event_count,"post_max_sr":float(pv.value.max()) if len(pv) else np.nan,
      "post_mean_sr":float(pv.value.mean()) if len(pv) else np.nan,
      "post_sr_auc_percent_min":float(pv.value.sum()/60) if len(pv) else np.nan,
      "post_classifiable":int(bool(len(pv)>=cfg.negative_valid_min_sec or (len(above) and above.any()))),
    }


def make_review_image(path: Path, review_id: str, raw1,raw2,f1,f2,sqi,start,cfg:Config=CFG):
    context_start=max(0,start-60); context_end=start+cfg.window_sec+60
    a=int(context_start*cfg.fs); b=min(len(raw1),int(context_end*cfg.fs))
    t=np.arange(a,b)/cfg.fs-start
    fig,ax=plt.subplots(5,1,figsize=(12,8),sharex=True,gridspec_kw={"height_ratios":[1,1,1.6,1.6,.7]})
    ax[0].plot(t,raw1[a:b],lw=.25,color="#17365D"); ax[0].set_ylabel("EEG ch1\n(uV)")
    ax[1].plot(t,raw2[a:b],lw=.25,color="#2F75B5"); ax[1].set_ylabel("EEG ch2\n(uV)")
    for j,x in enumerate([f1[a:b],f2[a:b]],start=2):
        f,tt,s=spectrogram(np.nan_to_num(x),fs=cfg.fs,nperseg=cfg.fs*2,noverlap=cfg.fs,scaling="density")
        m=(f>=.5)&(f<=30); ax[j].pcolormesh(tt+context_start-start,f[m],10*np.log10(s[m]+1e-12),shading="auto",cmap="viridis",vmin=-20,vmax=30); ax[j].set_ylim(.5,30); ax[j].set_ylabel(f"Ch{j-1}\nHz")
    q=sqi.loc[sqi.time.between(context_start,context_end)]; ax[4].plot(q.time-start,q.value,color="#548235",lw=1); ax[4].axhline(90,color="#C65911",ls="--",lw=.8); ax[4].set_ylim(0,105); ax[4].set_ylabel("SQI"); ax[4].set_xlabel("Seconds relative to candidate-window start")
    for x in ax: x.axvspan(0,cfg.window_sec,color="#FFD966",alpha=.12); x.axvline(0,color="#C65911",lw=1); x.axvline(cfg.window_sec,color="#C65911",lw=1); x.grid(alpha=.12)
    fig.suptitle(f"Blinded review {review_id} | shaded interval = candidate 120 s",fontsize=11)
    fig.tight_layout(rect=[0,0,1,.97]); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=110); plt.close(fig)


def review_id(caseid:int,order:int,salt:str)->str:
    return "REV-"+hashlib.sha256(f"{salt}:{caseid}:{order}".encode()).hexdigest()[:12].upper()


def cohort(vdb:VitalDB):
    d=vdb.cases.copy()
    d["age"]=pd.to_numeric(d.get("age"),errors="coerce"); d["asa"]=pd.to_numeric(d.get("asa"),errors="coerce")
    d=d.loc[(d.age>=18)&d.asa.between(1,4)&d.opstart.notna()&d.aneend.notna()].copy()
    ids=set(d.caseid.astype(int))
    for tr in TRACKS: ids &= set(vdb.trks.loc[vdb.trks.tname.eq(tr),"caseid"].astype(int))
    return d.loc[d.caseid.astype(int).isin(ids)].sort_values("caseid")


def process_case(vdb,row,out:Path,salt:str,cfg:Config=CFG):
    cid=int(row.caseid); result={"caseid":cid,"status":"processing"}
    try:
        bis=vdb.numeric(cid,"BIS/BIS"); sqi=vdb.numeric(cid,"BIS/SQI"); sr=vdb.numeric(cid,"BIS/SR"); ce=vdb.numeric(cid,"Orchestra/PPF20_CE")
        tci=first_time(ce,lambda x:x>0); anchor=first_time(bis,lambda x:x<=60,after=tci)
        result.update(tci_start_sec=tci,first_bis60_sec=anchor)
        if not np.isfinite(anchor): result.update(status="no_first_bis60"); return result,[],[]
        incision=finite_num(row.opstart); aneend=finite_num(row.aneend)
        if not np.isfinite(incision) or incision-anchor<cfg.window_sec: result.update(status="no_preincision_room"); return result,[],[]
        raw1=vdb.waveform(cid,"BIS/EEG1_WAV"); raw2=vdb.waveform(cid,"BIS/EEG2_WAV"); f1=filter_wave(raw1,cfg); f2=filter_wave(raw2,cfg)
        candidates=technical_candidates(raw1,raw2,f1,f2,sqi,anchor,incision,cfg)
        if not candidates: result.update(status="no_technical_candidate"); return result,[],[]
        reviews=[]; keys=[]
        for order,start in enumerate(candidates,1):
            rid=review_id(cid,order,salt); end=start+cfg.window_sec
            spectra=spectra_for_window(raw1,raw2,f1,f2,start,cfg); metrics=outcome_metrics(sr,sqi,anchor,start,end,aneend,cfg)
            image=f"images/{rid}.png"; make_review_image(out/"blinded_review"/image,rid,raw1,raw2,f1,f2,sqi,start,cfg)
            reviews.append({"review_id":rid,"candidate_order":order,"image_file":image,"decision":"","artifact_present":"","suppression_present":"","transition_present":"","both_channels_interpretable":"","comment":""})
            keys.append({"review_id":rid,"caseid":cid,"candidate_order":order,"candidate_start_sec":start,"candidate_end_sec":end,**spectra,**metrics})
        first=keys[0]
        keep=["candidate_start_sec","candidate_end_sec"]+[f"{band}_db_{channel}" for band in ["delta","theta","alpha","beta"] for channel in ["eeg1","eeg2","two_channel"]]+["pre_sr10","post_outcome_start_sec","post_outcome_end_sec","post_valid_sec","post_sr10","post_sr10_event_count","post_max_sr","post_mean_sr","post_sr_auc_percent_min","post_classifiable"]
        result.update({k:first[k] for k in keep}); result.update(status="visual_review_pending",review_id=first["review_id"],visual_candidate_n=len(keys))
        for c in ["age","sex","height","weight","bmi","asa","emop","department","optype","anestart","aneend","opstart","opend","preop_cr","preop_alb","preop_hb","icu_days","death_inhosp"]:
            if hasattr(row,c): result[c]=getattr(row,c)
        return result,reviews,keys
    except Exception as exc:
        result.update(status="processing_error",error=f"{type(exc).__name__}: {exc}")
        return result,[],[]


def write_csv(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(path,index=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--cache",required=True); ap.add_argument("--shard-index",type=int,default=0); ap.add_argument("--shard-count",type=int,default=1); ap.add_argument("--limit",type=int); ap.add_argument("--resume",action="store_true"); ap.add_argument("--salt",default="VitalDB-alpha-SR-blind-v1"); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); vdb=VitalDB(Path(args.cache)); cases=cohort(vdb); cases=cases.iloc[args.shard_index::args.shard_count]
    if args.limit: cases=cases.head(args.limit)
    checkpoint=out/"checkpoint.csv"; done=set()
    rows=[]
    if args.resume and checkpoint.exists(): rows=pd.read_csv(checkpoint).to_dict("records"); done={int(r["caseid"]) for r in rows}
    reviews=[]; keys=[]
    for n,row in enumerate(cases.itertuples(index=False),1):
        cid=int(row.caseid)
        if cid in done: continue
        r,rv,k=process_case(vdb,row,out,args.salt,CFG); rows.append(r); reviews.extend(rv); keys.extend(k)
        if n%5==0: write_csv(rows,checkpoint); print(f"shard {args.shard_index}: {n}/{len(cases)}",flush=True)
    write_csv(rows,out/"cohort.csv"); write_csv(rows,checkpoint); write_csv(reviews,out/"blinded_review"/"reviewer_form.csv"); write_csv(keys,out/"restricted_key.csv")
    (out/"run_manifest.json").write_text(json.dumps({"config":asdict(CFG),"shard_index":args.shard_index,"shard_count":args.shard_count,"cases":len(cases),"source":"Fresh VitalDB public API","visual_review_required":True},indent=2),encoding="utf-8")


if __name__=="__main__": main()
