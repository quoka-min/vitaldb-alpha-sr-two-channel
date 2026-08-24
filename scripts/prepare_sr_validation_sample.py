#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def blinded_id(caseid: int, set_name: str, salt: str) -> str:
    token = hashlib.sha256(f"{salt}:{set_name}:{caseid}".encode()).hexdigest()[:12].upper()
    return f"VSR-{token}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--main-n", type=int, default=300)
    ap.add_argument("--boundary-n", type=int, default=100)
    ap.add_argument("--salt", default="VitalDB-SR-validation-v1")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.cohort)
    eligible = d.loc[d.status.eq("visual_review_pending") & d.post_classifiable.eq(1)].copy()
    required = ["caseid", "post_outcome_start_sec", "post_outcome_end_sec", "post_max_sr", "post_sr10"]
    eligible = eligible.dropna(subset=required).drop_duplicates("caseid")
    rng = np.random.default_rng(a.seed)
    if len(eligible) < a.main_n:
        raise RuntimeError(f"Only {len(eligible)} eligible cases for main sample")
    main_idx = rng.choice(eligible.index.to_numpy(), size=a.main_n, replace=False)
    main = eligible.loc[main_idx].copy(); main["validation_set"] = "representative_random"
    boundary_pool = eligible.loc[eligible.post_max_sr.between(5, 15, inclusive="both") & ~eligible.caseid.isin(main.caseid)]
    if len(boundary_pool) < a.boundary_n:
        raise RuntimeError(f"Only {len(boundary_pool)} nonoverlapping boundary cases")
    boundary_idx = rng.choice(boundary_pool.index.to_numpy(), size=a.boundary_n, replace=False)
    boundary = boundary_pool.loc[boundary_idx].copy(); boundary["validation_set"] = "boundary_5_to_15"
    sample = pd.concat([main, boundary], ignore_index=True)
    sample["validation_id"] = [blinded_id(int(c), s, a.salt) for c, s in zip(sample.caseid, sample.validation_set)]
    sample["random_seed"] = a.seed
    keep = ["validation_id", "validation_set", "caseid", "post_outcome_start_sec", "post_outcome_end_sec",
            "post_max_sr", "post_mean_sr", "post_sr10", "post_valid_sec", "pre_sr10", "random_seed"]
    sample[keep].sort_values(["validation_set", "validation_id"]).to_csv(out / "restricted_sample_manifest.csv", index=False)
    blinded = sample[["validation_id", "validation_set"]].copy()
    blinded["image_file"] = "images/" + blinded.validation_id + ".png"
    blinded["both_channels_interpretable"] = ""
    blinded["artifact_present"] = ""
    blinded["physiologic_suppression_present"] = ""
    blinded["raw_eeg_suppression_fraction_ge10"] = ""
    blinded["suppression_fraction_category"] = ""
    blinded["reviewer_confidence"] = ""
    blinded["comment"] = ""
    blinded.sort_values(["validation_set", "validation_id"]).to_csv(out / "reviewer_form_blinded.csv", index=False)
    counts = sample.groupby(["validation_set", "post_sr10"]).size().rename("n").reset_index()
    counts.to_csv(out / "sample_counts_restricted.csv", index=False)


if __name__ == "__main__":
    main()
