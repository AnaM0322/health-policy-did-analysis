# Evaluating a Health Insurance Expansion with Difference-in-Differences

**A reproducible Python demonstration of quasi-experimental policy evaluation using simulated regional health-service utilization data.**

## Overview

Expanding public health-insurance coverage may improve access to healthcare, but estimating its impact requires separating policy effects from broader trends. This project demonstrates a **Difference-in-Differences (DiD)** approach to a hypothetical health-coverage expansion introduced in some regions in **2016**.

**Research question:** Did a regional expansion of public health-insurance coverage increase outpatient visits per 1,000 people relative to regions without the expansion?

> **Important:** The data are **simulated**, not observations from any country or actual reform. The results are methodological illustrations, **not evidence of a real policy's impact**.

## Study design and data

The script creates a balanced panel of **60 hypothetical regions** observed annually from **2010 through 2022**. One-third of regions are assigned to a treatment group, with coverage expansion starting in 2016. The outcome is outpatient visits per 1,000 people. The simulation includes region-specific baseline levels, a common time trend, random noise, and a deliberately specified **true treatment effect of +4 visits per 1,000 people** in treated regions after 2016.

This known effect provides a benchmark for assessing whether the estimation procedure recovers the simulated effect.

## Methods

### 1. Two-way fixed-effects DiD

The main regression is

$$
Y_{rt} = \alpha_r + \lambda_t + \beta (Treated_r \times Post_t) + \varepsilon_{rt},
$$

where $Y_{rt}$ is outpatient visits per 1,000 people in region $r$ and year $t$; $\alpha_r$ are region fixed effects; $\lambda_t$ are year fixed effects; and $\beta$ estimates the average difference-in-differences effect. Standard errors are clustered at the **region** level.

### 2. Event study and pre-treatment trends

The script estimates treatment-group-by-relative-year interactions, using **2015** (the year before expansion) as the reference period. It plots annual coefficient estimates with 95% confidence intervals and conducts a joint Wald test of pre-treatment interaction coefficients.

The central identification assumption is **parallel trends**: without the policy, treated and comparison regions would have evolved similarly. An insignificant pre-trend test does **not** prove this assumption.

### 3. Robustness demonstrations

- **Placebo intervention date:** Uses only pre-2016 observations and pretends treatment began in **2013**.
- **Randomization inference:** Randomly reassigns treatment status across regions (keeping the number treated fixed) and compares the observed DiD statistic with **1,000** placebo assignments.

These checks illustrate diagnostics; their interpretation depends on the design and assumptions of an actual study.

## Repository files

```text
.
├── did_policy_analysis.py    # Simulation, estimation, robustness checks, exports
├── README.md                 # Project documentation
└── output/                   # Generated when the script is run
    ├── results_table.csv     # Estimates, standard errors and p-values
    └── event_study.png       # Event-study estimates and 95% confidence intervals
```

The `output/` folder is generated on execution and need not be uploaded initially.

## Installation and execution

Requires **Python 3.9+** and the libraries below.

```bash
python -m pip install numpy pandas statsmodels matplotlib
python did_policy_analysis.py
```

On systems where `python` refers to Python 2, use `python3` instead.

The script prints a results table to the terminal and saves the CSV and PNG files to `output/`.

## Using real data

To adapt the analysis to actual policy data, replace the simulated `load_data()` function with a real dataset containing at least:

| Column | Meaning |
|---|---|
| `region` | Consistent region identifier |
| `year` | Calendar year |
| `treated_group` | 1 if the region was included in the expansion; otherwise 0 |
| `visits_per_1000` | Annual outpatient visits per 1,000 residents |

The remaining code also expects the derived variables `post`, `treat_post`, and `rel_year`, based on the intervention year. Check missing observations, policy timing, outcome definition, and comparison-group suitability before estimation. The existing randomization-inference implementation assumes a **balanced panel and a common treatment year**; it must be revisited for other designs.

For an empirical study, substantiate treatment assignment and timing; assess anticipation, spillovers and concurrent policies; and consider alternative DiD estimators if policy adoption is staggered across regions.

## Limitations

- **Synthetic data:** No real-world causal or policy conclusion can be drawn.
- **Simplified treatment assignment:** Treatment is randomly allocated in the simulation, unlike most real policy rollouts.
- **Simplified dynamics:** The simulated effect is constant after treatment, with a shared underlying time trend.
- **Identification:** Parallel trends is a substantive assumption; event-study diagnostics cannot establish it by themselves.
- **External validity:** Results do not represent any named region, country, insurer or healthcare system.

## Skills demonstrated

**Applied econometrics · Health-policy evaluation · Python · Panel-data analysis · Fixed effects · Clustered inference · Event-study visualization · Reproducible research**

## Project status

Methodological demonstration using synthetic data; suitable as a starting point for an empirical analysis once validated real-world data and policy details are available.
