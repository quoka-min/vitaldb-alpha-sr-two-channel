#!/usr/bin/env python3
"""Create clearly labeled synthetic burst-suppression training records."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, spectrogram


FS = 128
DURATION = 63


def colored_background(rng: np.random.Generator, n: int) -> np.ndarray:
    t = np.arange(n) / FS
    x = (9*np.sin(2*np.pi*10*t+rng.uniform(0, 6.28)) +
         4*np.sin(2*np.pi*6*t+rng.uniform(0, 6.28)) +
         3*np.sin(2*np.pi*18*t+rng.uniform(0, 6.28)) + rng.normal(0, 5, n))
    return x


def simulate(target_percent: int, seed: int, pattern: str = "mixed") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed); n = FS * DURATION
    x1 = colored_background(rng, n); x2 = .82*x1 + .45*colored_background(rng, n)
    suppressed = np.zeros(n, dtype=bool); total = int(np.ceil(n * target_percent / 100))
    if pattern == "single_long" and total:
        start = (n-total)//2; suppressed[start:start+total] = True
    elif pattern == "regular_short" and total:
        block = max(int(.6*FS), min(int(1.2*FS), total//5)); placed = 0
        for start in np.linspace(FS, n-block-FS, max(1, int(np.ceil(total/block))), dtype=int):
            take=min(block,total-placed); suppressed[start:start+take]=True; placed += take
            if placed>=total: break
    elif pattern == "clustered" and total:
        first=total//3; second=total-first
        suppressed[int(12*FS):int(12*FS)+first]=True
        start2=min(n-second-FS,int(38*FS)); suppressed[start2:start2+second]=True
    remaining = total; attempts = 0
    remaining = total-int(suppressed.sum())
    while remaining > 0 and attempts < 1000:
        attempts += 1
        length = min(remaining, int(rng.uniform(.6, 3.0)*FS))
        start = int(rng.uniform(1*FS, max(1*FS+1, n-length-1*FS)))
        if suppressed[max(0,start-FS//2):min(n,start+length+FS//2)].any(): continue
        suppressed[start:start+length] = True; remaining -= length
    # If fragmented placement could not reach the exact target, fill a final block.
    missing = total - int(suppressed.sum())
    if missing > 0: suppressed[n-missing-FS:n-FS] = True
    x1[suppressed] = rng.normal(0, .55, suppressed.sum())
    x2[suppressed] = rng.normal(0, .60, suppressed.sum())
    return x1, x2, suppressed


def render(path: Path, example_id: str, x1: np.ndarray, x2: np.ndarray,
           suppressed: np.ndarray, answer: bool) -> None:
    t = np.arange(len(x1))/FS - DURATION
    sos = butter(4, [.5, 30], btype="bandpass", fs=FS, output="sos")
    f1 = sosfiltfilt(sos, x1); f2 = sosfiltfilt(sos, x2)
    fig, ax = plt.subplots(5,1,figsize=(12,9),sharex=True,gridspec_kw={"height_ratios":[1.1,1.1,1.5,1.5,.7]})
    ax[0].plot(t,x1,lw=.3,color="#17365D"); ax[0].set_ylabel("EEG1 raw\nµV")
    ax[1].plot(t,x2,lw=.3,color="#2F75B5"); ax[1].set_ylabel("EEG2 raw\nµV")
    for j,x in enumerate([f1,f2],start=2):
        f,tt,p=spectrogram(x,fs=FS,nperseg=FS*2,noverlap=FS,scaling="density"); m=(f>=.5)&(f<=30)
        ax[j].pcolormesh(tt-DURATION,f[m],10*np.log10(p[m]+1e-12),shading="auto",cmap="viridis",vmin=-20,vmax=30)
        ax[j].set_ylim(.5,30); ax[j].set_ylabel(f"EEG{j-1}\nHz")
    ax[4].plot(t,np.full_like(t,95),color="#548235",lw=1); ax[4].axhline(80,color="#C65911",ls="--",lw=.8)
    ax[4].set_ylim(0,105); ax[4].set_ylabel("SQI"); ax[4].set_xlabel("Seconds before sampled time")
    for x in ax: x.grid(alpha=.15)
    title=f"SYNTHETIC TRAINING {example_id} | 63-second two-channel record"
    if answer:
        pct=100*suppressed.mean()
        threshold_label = "≥20%" if pct >= 20 else "≥10% and <20%" if pct >= 10 else "<10%"
        title += f" | ANSWER: suppression {pct:.1f}% ({threshold_label})"
    fig.suptitle(title,fontsize=11.5,color="#8B0000" if answer else "black")
    fig.text(.01,.005,"Synthetic educational signal—not a patient record. Judge physiologic suppression and whether suppression fraction is ≥10%.",fontsize=8.5)
    fig.tight_layout(rect=[0,.035,1,.965]); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=120); plt.close(fig)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); a=ap.parse_args(); out=Path(a.out)
    targets=[0,3,5,8,10,12,15,20,30,50]; rows=[]
    for i,target in enumerate(targets,1):
        eid=f"SYN-SR-{i:02d}"; x1,x2,supp=simulate(target,20260825+i,"mixed")
        actual=100*supp.mean(); render(out/"SELF_TEST_IMAGES"/f"{eid}.png",eid,x1,x2,supp,False)
        render(out/"ANNOTATED_ANSWERS"/f"{eid}_ANSWER.png",eid,x1,x2,supp,True)
        rows.append({"example_id":eid,"target_suppression_percent":target,"actual_suppression_percent":actual,
                     "pattern":"mixed","reference_raw_eeg_suppression_fraction_ge10":int(actual>=10),
                     "teaching_category":"no/very low" if actual<5 else "borderline <10" if actual<10 else "SR10-<20" if actual<20 else "SR20 or greater"})
    supplemental=[(12,"single_long"),(12,"regular_short"),(15,"clustered"),(18,"regular_short"),
                  (22,"single_long"),(25,"clustered"),(30,"regular_short"),(35,"mixed"),(40,"clustered"),(55,"mixed")]
    for j,(target,pattern) in enumerate(supplemental,1):
        eid=f"SYN-DIVERSE-{j:02d}"; x1,x2,supp=simulate(target,20260950+j,pattern); actual=100*supp.mean()
        render(out/"DIVERSE_SR10_SR20"/"SELF_TEST_IMAGES"/f"{eid}.png",eid,x1,x2,supp,False)
        render(out/"DIVERSE_SR10_SR20"/"ANNOTATED_ANSWERS"/f"{eid}_ANSWER.png",eid,x1,x2,supp,True)
        rows.append({"example_id":eid,"target_suppression_percent":target,"actual_suppression_percent":actual,
                     "pattern":pattern,"reference_raw_eeg_suppression_fraction_ge10":int(actual>=10),
                     "teaching_category":"SR10-<20" if actual<20 else "SR20 or greater"})
    pd.DataFrame(rows).to_csv(out/"SYNTHETIC_TRAINING_ANSWER_KEY.csv",index=False)
    (out/"README.md").write_text(
        "# Synthetic SR training set\n\nThe core set contains 10 graded examples (0, 3, 5, 8, 10, 12, 15, 20, 30, 50%). "
        "DIVERSE_SR10_SR20 contains 10 additional examples with long, short-repetitive, clustered, and mixed suppression patterns above 10% and 20%.\n"
        "Review SELF_TEST_IMAGES first using the same 63-second rule as the validation set, then compare with ANNOTATED_ANSWERS and the answer key.\n"
        "They demonstrate waveform recognition only and do not validate the manufacturer algorithm.\n",encoding="utf-8")


if __name__=="__main__": main()
