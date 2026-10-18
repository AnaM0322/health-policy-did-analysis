"""
Difference-in-Differences analysis of a regional health-coverage expansion
==========================================================================

Question: Did expanding public health-insurance coverage in a subset of regions
in 2016 increase per-capita use of outpatient care (visits per 1,000 people)?

Motivation: Universal health coverage (UHC) reforms are often rolled out
region by region, which creates quasi-experimental variation. The key
identifying assumption is parallel trends: absent the expansion, treated and
comparison regions would have followed the same utilization trajectory.

Data note: the panel below is SIMULATED so the script runs anywhere. To use real
data, replace `load_data()` with a loader returning columns: region, year,
treated_group (1 if region got the expansion), visits_per_1000.

Pipeline
--------
1. Simulate a balanced region-year panel of outpatient utilization (replace with real data in `load_data`)
2. Two-way fixed effects (TWFE) DiD with region-clustered standard errors
3. Event-study specification to test for pre-trends and show dynamic effects
4. Robustness: placebo treatment date + randomization inference
5. Export a regression table and a figure

Run:  python did_policy_analysis.py
Requires: numpy, pandas, statsmodels, matplotlib
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SEED = 42
TREAT_YEAR = 2016
TRUE_EFFECT = 4.0          # true post-expansion effect, used only to validate the estimator
OUT_DIR = Path("output")


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
def load_data(n_regions: int = 60, years=range(2010, 2023), seed: int = SEED) -> pd.DataFrame:
    """Simulate a balanced panel. Swap this function for pd.read_csv(...) on real data."""
    rng = np.random.default_rng(seed)
    regions = np.arange(n_regions)
    treated_regions = rng.choice(regions, size=n_regions // 3, replace=False)

    df = pd.DataFrame([(r, y) for r in regions for y in years], columns=["region", "year"])
    df["treated_group"] = df["region"].isin(treated_regions).astype(int)
    df["post"] = (df["year"] >= TREAT_YEAR).astype(int)
    df["treat_post"] = df["treated_group"] * df["post"]

    region_fe = rng.normal(50, 10, n_regions)[df["region"]]          # baseline differences
    year_fe = 0.8 * (df["year"] - df["year"].min())                  # common upward trend
    noise = rng.normal(0, 3, len(df))
    df["visits_per_1000"] = region_fe + year_fe + TRUE_EFFECT * df["treat_post"] + noise
    df["rel_year"] = df["year"] - TREAT_YEAR
    return df


# --------------------------------------------------------------------------- #
# 2. Main DiD (TWFE)
# --------------------------------------------------------------------------- #
def run_twfe(df: pd.DataFrame, treat_var: str = "treat_post"):
    """TWFE regression with region and year fixed effects, SEs clustered by region."""
    model = smf.ols(f"visits_per_1000 ~ {treat_var} + C(region) + C(year)", data=df)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df["region"]})


# --------------------------------------------------------------------------- #
# 3. Event study
# --------------------------------------------------------------------------- #
def run_event_study(df: pd.DataFrame, ref_period: int = -1) -> pd.DataFrame:
    """Estimate effects by year relative to treatment, omitting `ref_period`."""
    d = df.copy()
    rel_years = sorted(d["rel_year"].unique())
    terms = []
    for k in rel_years:
        if k == ref_period:
            continue
        name = f"ev_m{abs(k)}" if k < 0 else f"ev_p{k}"
        d[name] = ((d["rel_year"] == k) & (d["treated_group"] == 1)).astype(int)
        terms.append((k, name))

    formula = "visits_per_1000 ~ " + " + ".join(n for _, n in terms) + " + C(region) + C(year)"
    res = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["region"]})

    rows = []
    for k, name in terms:
        lo, hi = res.conf_int().loc[name]
        rows.append({"rel_year": k, "coef": res.params[name], "ci_low": lo, "ci_high": hi})
    rows.append({"rel_year": ref_period, "coef": 0.0, "ci_low": 0.0, "ci_high": 0.0})
    return pd.DataFrame(rows).sort_values("rel_year").reset_index(drop=True)


def joint_pretrend_test(df: pd.DataFrame, ref_period: int = -1) -> float:
    """Wald test that all pre-period event-study coefficients equal zero. Returns p-value."""
    d = df.copy()
    pre = [k for k in sorted(d["rel_year"].unique()) if k < 0 and k != ref_period]
    names = []
    for k in pre:
        n = f"pre_m{abs(k)}"
        d[n] = ((d["rel_year"] == k) & (d["treated_group"] == 1)).astype(int)
        names.append(n)
    post = d["rel_year"] >= 0
    d["post_treated"] = (post & (d["treated_group"] == 1)).astype(int)
    formula = ("visits_per_1000 ~ " + " + ".join(names) + " + post_treated + C(region) + C(year)")
    res = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["region"]})
    wald = res.wald_test(", ".join(f"{n} = 0" for n in names), scalar=True)
    return float(wald.pvalue)


# --------------------------------------------------------------------------- #
# 4. Robustness
# --------------------------------------------------------------------------- #
def placebo_test(df: pd.DataFrame, fake_year: int = 2013):
    """Pre-period only, pretend treatment started earlier. Effect should be ~0."""
    d = df[df["year"] < TREAT_YEAR].copy()
    d["placebo"] = ((d["year"] >= fake_year) & (d["treated_group"] == 1)).astype(int)
    return run_twfe(d, "placebo")


def randomization_inference(df: pd.DataFrame, n_perm: int = 1000, seed: int = SEED) -> tuple:
    """
    Permute which regions are 'treated' and recompute the DiD estimate.
    With a balanced panel and common timing, TWFE equals the classic 2x2 DiD of means.
    """
    rng = np.random.default_rng(seed)
    wide = df.pivot(index="region", columns="year", values="visits_per_1000")
    post_cols = [c for c in wide.columns if c >= TREAT_YEAR]
    pre_cols = [c for c in wide.columns if c < TREAT_YEAR]
    delta = wide[post_cols].mean(axis=1) - wide[pre_cols].mean(axis=1)   # per-region change

    treated = df.groupby("region")["treated_group"].first().reindex(wide.index).values.astype(bool)
    observed = delta[treated].mean() - delta[~treated].mean()

    n_treated = treated.sum()
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(delta))
        fake = np.zeros(len(delta), dtype=bool)
        fake[idx[:n_treated]] = True
        perm_stats[i] = delta[fake].mean() - delta[~fake].mean()

    p_value = (np.sum(np.abs(perm_stats) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, p_value, perm_stats


# --------------------------------------------------------------------------- #
# 5. Output
# --------------------------------------------------------------------------- #
def plot_event_study(es: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(es["rel_year"], es["coef"],
                yerr=[es["coef"] - es["ci_low"], es["ci_high"] - es["coef"]],
                fmt="o", color="#1f4e79", ecolor="#7fa6c9", capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="firebrick", ls="--", lw=1, label="Coverage expansion")
    ax.set_xlabel("Years relative to coverage expansion")
    ax.set_ylabel("Effect on outpatient visits per 1,000 people")
    ax.set_title("Event study: dynamic treatment effects (95% CI, clustered by region)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_table(main, placebo, ri_obs, ri_p, pretrend_p) -> pd.DataFrame:
    def row(label, coef, se, p):
        return {"Specification": label, "Estimate": f"{coef:.3f}",
                "Std. error": f"{se:.3f}" if se is not None else "-", "p-value": f"{p:.3f}"}
    return pd.DataFrame([
        row("TWFE DiD (main)", main.params["treat_post"], main.bse["treat_post"], main.pvalues["treat_post"]),
        row("Placebo (fake 2013 start)", placebo.params["placebo"], placebo.bse["placebo"], placebo.pvalues["placebo"]),
        row("Randomization inference", ri_obs, None, ri_p),
        row("Joint pre-trend test (Wald)", np.nan, None, pretrend_p),
    ])


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    print(f"Panel: {df['region'].nunique()} regions x {df['year'].nunique()} years "
          f"({df['treated_group'].sum() // df['year'].nunique()} treated regions)\n")

    main_res = run_twfe(df)
    es = run_event_study(df)
    pretrend_p = joint_pretrend_test(df)
    placebo_res = placebo_test(df)
    ri_obs, ri_p, _ = randomization_inference(df)

    table = build_table(main_res, placebo_res, ri_obs, ri_p, pretrend_p)
    print(table.to_string(index=False))
    print(f"\nTrue simulated effect: {TRUE_EFFECT}  |  Estimated: {main_res.params['treat_post']:.3f}")

    table.to_csv(OUT_DIR / "results_table.csv", index=False)
    plot_event_study(es, OUT_DIR / "event_study.png")
    print(f"\nSaved results to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
