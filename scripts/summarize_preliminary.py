#!/usr/bin/env python3
"""Summarize technically eligible windows before manual blinded adjudication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr


def qstats(s: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return {"n": int(len(x)), "median": float(x.median()), "q1": float(x.quantile(.25)),
            "q3": float(x.quantile(.75)), "mean": float(x.mean()), "sd": float(x.std())}


def fit_logit(data: pd.DataFrame, adjusted: bool) -> dict[str, float | int | str]:
    d = data.copy()
    sd = d["alpha_db_two_channel"].std(ddof=0)
    d["lower_alpha_z"] = -(d["alpha_db_two_channel"] - d["alpha_db_two_channel"].mean()) / sd
    terms = ["lower_alpha_z"]
    if adjusted:
        for col, term in [("age", "age"), ("sex", "C(sex)"), ("bmi", "bmi"),
                          ("asa", "C(asa)"), ("emop", "emop")]:
            if col in d.columns and d[col].notna().sum() > 0:
                terms.append(term)
    formula = "post_sr10 ~ " + " + ".join(terms)
    model = smf.glm(formula, data=d, family=sm.families.Binomial()).fit(cov_type="HC1")
    ci = model.conf_int().loc["lower_alpha_z"]
    return {"model": "adjusted" if adjusted else "unadjusted", "n": int(model.nobs),
            "events": int(d.loc[model.model.data.row_labels, "post_sr10"].sum()),
            "or_per_1sd_lower_alpha": float(np.exp(model.params["lower_alpha_z"])),
            "ci_low": float(np.exp(ci.iloc[0])), "ci_high": float(np.exp(ci.iloc[1])),
            "p": float(model.pvalues["lower_alpha_z"]), "formula": formula}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(args.cohort)
    tech = cohort.loc[cohort.status.eq("visual_review_pending")].copy()
    classifiable = tech.loc[pd.to_numeric(tech.post_classifiable, errors="coerce").eq(1)].copy()
    for c in ["post_sr10", "pre_sr10"]:
        classifiable[c] = pd.to_numeric(classifiable[c], errors="coerce")

    status = cohort.status.value_counts(dropna=False).rename_axis("status").reset_index(name="n")
    status["percent"] = 100 * status.n / len(cohort)
    status.to_csv(out / "cohort_flow.csv", index=False)

    band_rows = []
    for band in ["delta", "theta", "alpha", "beta"]:
        for channel in ["eeg1", "eeg2", "two_channel"]:
            col = f"{band}_db_{channel}"
            if col in tech:
                band_rows.append({"band": band, "channel": channel, **qstats(tech[col])})
    pd.DataFrame(band_rows).to_csv(out / "band_power_summary.csv", index=False)

    quart = pd.DataFrame()
    if len(classifiable) >= 4:
        classifiable["alpha_quartile"] = pd.qcut(classifiable.alpha_db_two_channel, 4,
                                                  labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
        quart = classifiable.groupby("alpha_quartile", observed=True).agg(
            n=("caseid", "size"), alpha_median_db=("alpha_db_two_channel", "median"),
            alpha_q1_db=("alpha_db_two_channel", lambda x: x.quantile(.25)),
            alpha_q3_db=("alpha_db_two_channel", lambda x: x.quantile(.75)),
            pre_sr_n=("pre_sr10", "sum"), post_sr_n=("post_sr10", "sum"),
            valid_followup_sec=("post_valid_sec", "sum"), post_mean_sr=("post_mean_sr", "mean"),
            post_sr_auc_percent_min=("post_sr_auc_percent_min", "mean")).reset_index()
        quart["post_sr_percent"] = 100 * quart.post_sr_n / quart.n
        quart["event_rate_per_valid_hour"] = 3600 * quart.post_sr_n / quart.valid_followup_sec
        quart.to_csv(out / "alpha_quartile_sr_summary.csv", index=False)

    models = []
    if len(classifiable) and classifiable.post_sr10.nunique() > 1:
        for adjusted in [False, True]:
            try: models.append(fit_logit(classifiable, adjusted))
            except Exception as exc: models.append({"model": "adjusted" if adjusted else "unadjusted", "error": str(exc)})
    pd.DataFrame(models).to_csv(out / "preliminary_logistic_models.csv", index=False)

    incident = classifiable.loc[classifiable.pre_sr10.eq(0)].copy()
    incident_models = []
    if len(incident) and incident.post_sr10.nunique() > 1:
        for adjusted in [False, True]:
            try: incident_models.append(fit_logit(incident, adjusted))
            except Exception as exc: incident_models.append({"model": "adjusted" if adjusted else "unadjusted", "error": str(exc)})
    pd.DataFrame(incident_models).to_csv(out / "incident_sr_sensitivity_models.csv", index=False)

    rho = pval = np.nan
    corr = classifiable[["alpha_db_two_channel", "post_mean_sr"]].dropna()
    if len(corr) >= 3: rho, pval = spearmanr(corr.iloc[:, 0], corr.iloc[:, 1])
    summary = {
        "all_candidate_cases": int(len(cohort)), "technical_windows_pending_visual_review": int(len(tech)),
        "post_sr_classifiable": int(len(classifiable)), "post_sr_events": int(classifiable.post_sr10.sum()) if len(classifiable) else 0,
        "post_sr_percent": float(100 * classifiable.post_sr10.mean()) if len(classifiable) else None,
        "pre_window_sr_n": int(classifiable.pre_sr10.sum()) if len(classifiable) else 0,
        "incident_analysis_n": int(len(incident)), "incident_post_sr_events": int(incident.post_sr10.sum()) if len(incident) else 0,
        "spearman_alpha_vs_post_mean_sr": float(rho), "spearman_p": float(pval),
        "manual_blinded_review_complete": False,
        "interpretation_boundary": "Exploratory automated-screening result only; not the final prespecified cohort."
    }
    (out / "preliminary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tech.to_csv(out / "technical_candidates_pending_review.csv", index=False)

    lines = ["# Preliminary VitalDB EEG result", "", "Automated technical screening only; manual blinded EEG review is not complete.", "",
             f"- Screened cohort: {len(cohort):,}", f"- Technically eligible windows pending review: {len(tech):,}",
             f"- Classifiable subsequent SR outcome: {len(classifiable):,}",
             f"- Subsequent SR>=10: {int(classifiable.post_sr10.sum()) if len(classifiable) else 0:,} ({100*classifiable.post_sr10.mean():.1f}% if classifiable)",
             f"- Incident-analysis population (no pre-window SR): {len(incident):,}", "",
             "Do not report these estimates as the primary result until blinded review decisions are merged."]
    (out / "README_PRELIMINARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
