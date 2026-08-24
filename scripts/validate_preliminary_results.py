#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.outliers_influence import variance_inflation_factor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--quartiles", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.data)
    q = pd.read_csv(a.quartiles)
    m = pd.read_csv(a.models)
    eligible = d.loc[d.post_classifiable.eq(1)].copy()
    complete = eligible.dropna(subset=["alpha_db_two_channel", "post_sr10"])

    # Convert the standardized adjusted estimate to an interpretable 1-dB scale.
    alpha_sd = float(complete.alpha_db_two_channel.std(ddof=0))
    adjusted = m.loc[m.model.eq("adjusted")].iloc[0]
    or_1db = float(np.exp(np.log(adjusted.or_per_1sd_lower_alpha) / alpha_sd))
    lo_1db = float(np.exp(np.log(adjusted.ci_low) / alpha_sd))
    hi_1db = float(np.exp(np.log(adjusted.ci_high) / alpha_sd))

    rates = []
    for r in q.itertuples(index=False):
        lo, hi = proportion_confint(int(r.post_sr_n), int(r.n), method="wilson")
        rates.append({"quartile": r.alpha_quartile, "n": int(r.n), "events": int(r.post_sr_n),
                      "risk": float(r.post_sr_n / r.n), "ci_low": float(lo), "ci_high": float(hi)})
    pd.DataFrame(rates).to_csv(out / "quartile_risk_with_95ci.csv", index=False)

    low, high = rates[0], rates[-1]
    channel = d.dropna(subset=["alpha_db_eeg1", "alpha_db_eeg2"])
    diff = channel.alpha_db_eeg1 - channel.alpha_db_eeg2
    pr = pearsonr(channel.alpha_db_eeg1, channel.alpha_db_eeg2)
    sr = spearmanr(channel.alpha_db_eeg1, channel.alpha_db_eeg2)

    results = {
        "alpha_sd_db": alpha_sd,
        "adjusted_or_per_1db_lower_alpha": or_1db,
        "adjusted_or_per_1db_ci_low": lo_1db,
        "adjusted_or_per_1db_ci_high": hi_1db,
        "q1_vs_q4_risk_ratio": float(low["risk"] / high["risk"]),
        "q1_minus_q4_risk_difference_percentage_points": float(100 * (low["risk"] - high["risk"])),
        "channel_alpha_difference_eeg1_minus_eeg2_median_db": float(diff.median()),
        "channel_alpha_difference_q1_db": float(diff.quantile(.25)),
        "channel_alpha_difference_q3_db": float(diff.quantile(.75)),
        "channel_alpha_pearson_r": float(pr.statistic),
        "channel_alpha_spearman_rho": float(sr.statistic),
        "technical_pending_n": int(len(d)),
        "classifiable_n": int(len(eligible)),
        "complete_alpha_outcome_n": int(len(complete)),
        "missing_alpha_among_classifiable_n": int(len(eligible) - len(complete)),
    }
    (out / "validation_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    sensitivity = []
    for band in ["delta", "theta", "alpha", "beta"]:
        for channel_name in (["eeg1", "eeg2", "two_channel"] if band == "alpha" else ["two_channel"]):
            col = f"{band}_db_{channel_name}"
            md = eligible.dropna(subset=[col, "post_sr10", "age", "sex", "bmi", "asa", "emop"]).copy()
            sd = md[col].std(ddof=0)
            md["lower_power_z"] = -(md[col] - md[col].mean()) / sd
            model = smf.glm("post_sr10 ~ lower_power_z + age + C(sex) + bmi + C(asa) + emop",
                            data=md, family=sm.families.Binomial()).fit(cov_type="HC1")
            ci = model.conf_int().loc["lower_power_z"]
            sensitivity.append({"band": band, "channel": channel_name, "n": int(model.nobs),
                                "or_per_1sd_lower_power": float(np.exp(model.params.lower_power_z)),
                                "ci_low": float(np.exp(ci.iloc[0])), "ci_high": float(np.exp(ci.iloc[1])),
                                "p": float(model.pvalues.lower_power_z)})
    pd.DataFrame(sensitivity).to_csv(out / "band_channel_sensitivity_models.csv", index=False)

    band_cols = [f"{b}_db_two_channel" for b in ["delta", "theta", "alpha", "beta"]]
    multi = eligible.dropna(subset=band_cols + ["post_sr10", "age", "sex", "bmi", "asa", "emop"]).copy()
    zcols = []
    for band, col in zip(["delta", "theta", "alpha", "beta"], band_cols):
        z = f"lower_{band}_z"; zcols.append(z)
        multi[z] = -(multi[col] - multi[col].mean()) / multi[col].std(ddof=0)
    corr_matrix = multi[band_cols].corr(method="spearman")
    corr_matrix.to_csv(out / "band_power_spearman_correlations.csv")
    x_vif = sm.add_constant(multi[zcols])
    pd.DataFrame({"term": x_vif.columns,
                  "vif": [variance_inflation_factor(x_vif.to_numpy(), i) for i in range(x_vif.shape[1])]}) \
        .to_csv(out / "multiband_vif.csv", index=False)
    multimodel = smf.glm("post_sr10 ~ " + " + ".join(zcols) + " + age + C(sex) + bmi + C(asa) + emop",
                         data=multi, family=sm.families.Binomial()).fit(cov_type="HC1")
    multi_rows = []
    for z in zcols:
        ci = multimodel.conf_int().loc[z]
        multi_rows.append({"term": z, "or": float(np.exp(multimodel.params[z])),
                           "ci_low": float(np.exp(ci.iloc[0])), "ci_high": float(np.exp(ci.iloc[1])),
                           "p": float(multimodel.pvalues[z])})
    pd.DataFrame(multi_rows).to_csv(out / "simultaneous_multiband_model.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(len(rates)); risks = np.array([r["risk"] for r in rates]) * 100
    lo = risks - np.array([r["ci_low"] for r in rates]) * 100
    hi = np.array([r["ci_high"] for r in rates]) * 100 - risks
    ax.errorbar(x, risks, yerr=[lo, hi], fmt="o-", capsize=4, color="#17365D", lw=2)
    ax.set_xticks(x, [r["quartile"] for r in rates])
    ax.set_ylabel("Subsequent SR≥10 (%)")
    ax.set_xlabel("Two-channel alpha-power quartile")
    ax.set_ylim(0, max(risks + hi) * 1.15)
    ax.grid(axis="y", alpha=.25)
    ax.set_title("Automated technical candidates; manual blinded review pending")
    fig.tight_layout(); fig.savefig(out / "alpha_quartile_sr_risk.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
