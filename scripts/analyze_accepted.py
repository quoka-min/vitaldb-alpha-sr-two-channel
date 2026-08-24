#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.cohort)
    reviews = pd.read_csv(args.reviews, dtype={"review_id": str})
    key = pd.read_csv(args.key, dtype={"review_id": str})
    accepted = reviews.loc[reviews["decision"].str.upper().eq("ACCEPT"), ["review_id"]]
    data = key.merge(accepted, on="review_id", how="inner").merge(cohort, on="caseid", how="left")
    data = data.dropna(subset=["alpha_db_two_channel", "post_sr10"])

    data["alpha_z"] = (data["alpha_db_two_channel"] - data["alpha_db_two_channel"].mean()) / data["alpha_db_two_channel"].std(ddof=0)
    data["lower_alpha_z"] = -data["alpha_z"]
    data["alpha_quartile"] = pd.qcut(data["alpha_db_two_channel"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])

    quart = data.groupby("alpha_quartile", observed=True).agg(
        n=("caseid", "size"),
        alpha_median_db=("alpha_db_two_channel", "median"),
        pre_sr10_n=("pre_sr10", "sum"),
        post_sr10_n=("post_sr10", "sum"),
        valid_followup_sec=("post_valid_sec", "sum"),
    ).reset_index()
    quart["post_sr10_percent"] = 100 * quart["post_sr10_n"] / quart["n"]
    quart["post_sr10_rate_per_hour"] = 3600 * quart["post_sr10_n"] / quart["valid_followup_sec"]
    quart.to_csv(out / "alpha_quartile_sr_summary.csv", index=False)

    covars = [c for c in ["age", "bmi", "C(sex)", "emop", "C(asa)"] if c.replace("C(", "").replace(")", "") in data.columns]
    formula = "post_sr10 ~ lower_alpha_z" + (" + " + " + ".join(covars) if covars else "")
    model = smf.glm(formula, data=data, family=sm.families.Binomial()).fit(cov_type="HC1")
    ci = model.conf_int().loc["lower_alpha_z"]
    summary = pd.DataFrame([{
        "analysis": "all accepted: subsequent occurrence or recurrence",
        "n": len(data),
        "events": int(data.post_sr10.sum()),
        "OR_per_1SD_lower_alpha": float(np.exp(model.params["lower_alpha_z"])),
        "CI_low": float(np.exp(ci.iloc[0])),
        "CI_high": float(np.exp(ci.iloc[1])),
        "p": float(model.pvalues["lower_alpha_z"]),
    }])

    incident = data.loc[data["pre_sr10"].fillna(0).eq(0)].copy()
    if len(incident) and incident["post_sr10"].nunique() > 1:
        im = smf.glm(formula, data=incident, family=sm.families.Binomial()).fit(cov_type="HC1")
        ici = im.conf_int().loc["lower_alpha_z"]
        summary.loc[len(summary)] = {
            "analysis": "incident-SR sensitivity: no pre-window SR",
            "n": len(incident), "events": int(incident.post_sr10.sum()),
            "OR_per_1SD_lower_alpha": float(np.exp(im.params["lower_alpha_z"])),
            "CI_low": float(np.exp(ici.iloc[0])), "CI_high": float(np.exp(ici.iloc[1])),
            "p": float(im.pvalues["lower_alpha_z"]),
        }
    summary.to_csv(out / "continuous_alpha_models.csv", index=False)
    data.to_csv(out / "accepted_analysis_dataset.csv", index=False)


if __name__ == "__main__":
    main()

