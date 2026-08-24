#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(d: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    ref = d.raw_eeg_suppression_fraction_ge10.astype(str).str.upper().map({"Y":1,"YES":1,"1":1,"N":0,"NO":0,"0":0})
    test = pd.to_numeric(d.manufacturer_segment_sr_ge10, errors="coerce")
    x = pd.DataFrame({"ref": ref, "test": test}).dropna().astype(int)
    tp = int(((x.ref==1)&(x.test==1)).sum()); fn = int(((x.ref==1)&(x.test==0)).sum())
    fp = int(((x.ref==0)&(x.test==1)).sum()); tn = int(((x.ref==0)&(x.test==0)).sum())
    div = lambda n, z: n/z if z else np.nan
    return {"analysis_set":label,"n":len(x),"TP":tp,"FN":fn,"FP":fp,"TN":tn,
            "sensitivity":div(tp,tp+fn),"specificity":div(tn,tn+fp),
            "ppv":div(tp,tp+fp),"npv":div(tn,tn+fn),"accuracy":div(tp+tn,len(x))}


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--reviews",required=True); ap.add_argument("--key",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    reviews=pd.read_csv(a.reviews,dtype=str); key=pd.read_csv(a.key)
    d=reviews.merge(key,on="validation_id",how="inner")
    usable=d.loc[d.both_channels_interpretable.astype(str).str.upper().isin(["Y","YES","1"])]
    rows=[metrics(usable.loc[usable.validation_set_x.eq("representative_random")],"primary_representative_random"),
          metrics(usable.loc[usable.validation_set_x.eq("boundary_5_to_15")],"boundary_sensitivity_analysis")]
    pd.DataFrame(rows).to_csv(out/"diagnostic_accuracy.csv",index=False)
    d.to_csv(out/"adjudication_merged_restricted.csv",index=False)


if __name__=="__main__": main()
