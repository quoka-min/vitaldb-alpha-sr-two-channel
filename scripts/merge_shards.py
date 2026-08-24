#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def concat(root:Path,name:str):
    files=list(root.glob(f"**/{name}")); frames=[]
    for f in files:
        try:
            d=pd.read_csv(f)
            if len(d): frames.append(d)
        except pd.errors.EmptyDataError: pass
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
    root=Path(a.input); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    cohort=concat(root,"cohort.csv").drop_duplicates("caseid",keep="last").sort_values("caseid")
    reviews=concat(root,"reviewer_form.csv").drop_duplicates("review_id")
    keys=concat(root,"restricted_key.csv").drop_duplicates("review_id")
    cohort.to_csv(out/"cohort_fresh_vitaldb_api.csv",index=False)
    reviews.to_csv(out/"reviewer_form_blinded.csv",index=False)
    keys.to_csv(out/"restricted_key_do_not_share_with_reviewers.csv",index=False)
    # Copy blinded images into one directory.
    image_out=out/"blinded_review_images"; image_out.mkdir(exist_ok=True)
    for f in root.glob("**/images/*.png"):
        target=image_out/f.name
        if not target.exists(): target.write_bytes(f.read_bytes())
    summary={"cohort_n":len(cohort),"status_counts":cohort.status.value_counts(dropna=False).to_dict() if len(cohort) else {},"review_rows":len(reviews),"key_rows":len(keys),"image_n":len(list(image_out.glob('*.png'))),"source":"Fresh VitalDB public API; no legacy derived EEG variables used"}
    (out/"merge_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__": main()

