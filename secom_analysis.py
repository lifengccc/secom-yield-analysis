import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA = sys.argv[1] if len(sys.argv) > 1 else "secom.data"
LABELS = sys.argv[2] if len(sys.argv) > 2 else "secom_labels.data"

OUT = "secom_output"

os.makedirs(OUT, exist_ok=True)

SEED = 42


# ============================================================
# 1. LOAD DATA
# ============================================================

X = pd.read_csv(
    DATA,
    sep=r"\s+",
    header=None,
    na_values=["NaN"]
)

# Give each sensor a readable name
X.columns = [
    f"S{i+1:03d}"
    for i in range(X.shape[1])
]


# Load pass/fail labels and timestamps
_rows = []

with open(LABELS, encoding="utf-8") as fh:

    for line in fh:

        line = line.strip()

        if not line:
            continue

        label_txt, stamp_txt = line.split(None, 1)

        _rows.append(
            (
                int(label_txt),
                stamp_txt.strip().strip('"')
            )
        )


lab = pd.DataFrame(
    _rows,
    columns=["label", "stamp"]
)


lab["timestamp"] = pd.to_datetime(
    lab["stamp"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)


if lab["timestamp"].isna().all():

    sys.exit(
        "Could not parse timestamps in the labels file; "
        "check its format."
    )


# SECOM labels:
# -1 = PASS
#  1 = FAIL
#
# Convert to:
# 0 = PASS
# 1 = FAIL

y = (
    lab["label"] == 1
).astype(int).values


ts = lab["timestamp"]


n_runs, n_sensors = X.shape

n_fail = int(y.sum())

yield_pct = 100 * (
    1 - n_fail / n_runs
)


print(
    f"Loaded {n_runs} runs, "
    f"{n_sensors} sensors, "
    f"{n_fail} fails, "
    f"yield {yield_pct:.1f}%"
)


# ============================================================
# 2. CLEAN DATA
# ============================================================

# Calculate the percentage of missing values
# for every sensor.

miss_rate = X.isna().mean()


# Remove sensors where more than 40%
# of the measurements are missing.

MISS_CUTOFF = 0.40


drop_missing = miss_rate[
    miss_rate > MISS_CUTOFF
].index


X1 = X.drop(
    columns=drop_missing
)


# Remove sensors that have no variation.
#
# A constant sensor cannot help distinguish
# passing runs from failing runs.

const_cols = X1.columns[
    X1.nunique(dropna=True) <= 1
]


X1 = X1.drop(
    columns=const_cols
)


n_kept = X1.shape[1]


print(
    f"Dropped {len(drop_missing)} sensors "
    f"with >{int(MISS_CUTOFF * 100)}% missing, "
    f"{len(const_cols)} constant sensors; "
    f"{n_kept} remain"
)


# ============================================================
# FIGURE 1 — MISSING DATA OVERVIEW
# ============================================================

fig, ax = plt.subplots(
    1,
    2,
    figsize=(11, 4)
)


# Missing data histogram

ax[0].hist(
    miss_rate * 100,
    bins=30,
    color="#4C72B0"
)


ax[0].axvline(
    MISS_CUTOFF * 100,
    color="crimson",
    linestyle="--",
    label=f"{int(MISS_CUTOFF * 100)}% cutoff"
)


ax[0].set_xlabel(
    "% missing per sensor"
)

ax[0].set_ylabel(
    "Number of sensors"
)

ax[0].set_title(
    "Missing data by sensor"
)

ax[0].legend()


# Sensors retained vs removed

ax[1].bar(
    [
        "Kept",
        "Dropped (missing)",
        "Dropped (constant)"
    ],
    [
        n_kept,
        len(drop_missing),
        len(const_cols)
    ],
    color=[
        "#55A868",
        "#C44E52",
        "#8172B2"
    ]
)


ax[1].set_title(
    "Sensors after cleaning"
)

ax[1].set_ylabel(
    "Count"
)


plt.tight_layout()


plt.savefig(
    f"{OUT}/fig1_missing_data.png",
    dpi=150
)


plt.close()


# ============================================================
# 3. RANK SENSORS BY ASSOCIATION WITH FAILURE
# ============================================================

# Mann-Whitney U test:
#
# Compare sensor measurements from PASS runs
# against measurements from FAIL runs.
#
# This is a non-parametric statistical test,
# so it does not require assuming the sensor
# measurements are normally distributed.
#
# Because hundreds of sensors are tested,
# Benjamini-Hochberg correction is later used
# to reduce false discoveries.


rows = []


for c in X1.columns:

    # Passing runs
    a = X1.loc[
        y == 0,
        c
    ].dropna()

    # Failing runs
    b = X1.loc[
        y == 1,
        c
    ].dropna()


    # Skip sensors without enough observations

    if len(a) < 10 or len(b) < 10:
        continue


    # Mann-Whitney U test

    u, p = stats.mannwhitneyu(
        a,
        b,
        alternative="two-sided"
    )


    # AUC-style effect size.
    #
    # 0 means little separation.
    # Values closer to 1 indicate
    # stronger separation.

    auc = u / (
        len(a) * len(b)
    )


    effect = abs(
        auc - 0.5
    ) * 2


    # Point-biserial correlation

    r, _ = stats.pointbiserialr(
        y[X1[c].notna()],
        X1[c].dropna()
    )


    rows.append(
        (
            c,
            p,
            effect,
            r,
            a.median(),
            b.median(),
            len(a),
            len(b)
        )
    )


# Store results

res = pd.DataFrame(
    rows,
    columns=[
        "sensor",
        "p_value",
        "effect_size",
        "point_biserial_r",
        "median_pass",
        "median_fail",
        "n_pass",
        "n_fail"
    ]
)


# ============================================================
# BENJAMINI-HOCHBERG MULTIPLE-TESTING CORRECTION
# ============================================================

m = len(res)

res["p_adj_BH"] = np.nan


srt = res.sort_values(
    "p_value"
).reset_index()


bh = (
    srt["p_value"]
    * m
    / (np.arange(m) + 1)
).values


bh = np.minimum.accumulate(
    bh[::-1]
)[::-1]


res.loc[
    srt["index"],
    "p_adj_BH"
] = np.clip(
    bh,
    0,
    1
)


# Rank sensors by statistical significance

res = res.sort_values(
    "p_value"
).reset_index(
    drop=True
)


# Save all sensor results

res.to_csv(
    f"{OUT}/ranked_features.csv",
    index=False
)


# Count statistically significant sensors

n_sig = int(
    (
        res["p_adj_BH"] < 0.05
    ).sum()
)


# Select top six sensors for visualization

top = res.head(6)


print(
    f"{n_sig} sensors significantly associated "
    f"with failure after BH correction (q<0.05)"
)


# ============================================================
# FIGURE 2 — TOP SENSOR VARIABLES
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 7)
)


for ax, sensor in zip(
    axes.ravel(),
    top["sensor"]
):

    data = [
        X1.loc[
            y == 0,
            sensor
        ].dropna(),

        X1.loc[
            y == 1,
            sensor
        ].dropna()
    ]


    ax.boxplot(
        data,
        tick_labels=[
            "Pass",
            "Fail"
        ],
        showfliers=False
    )


    sensor_result = res.loc[
        res["sensor"] == sensor
    ].iloc[0]


    ax.set_title(
        f"{sensor} "
        f"(q={sensor_result['p_adj_BH']:.3g})"
    )


plt.suptitle(
    "Top 6 sensors separating pass vs fail runs"
)


plt.tight_layout()


plt.savefig(
    f"{OUT}/fig2_top_features.png",
    dpi=150
)


plt.close()


# ============================================================
# 4. YIELD OVER TIME — P-CHART
# ============================================================

# Combine timestamps and failure labels.

tdf = pd.DataFrame(
    {
        "ts": ts,
        "fail": y
    }
).dropna().sort_values(
    "ts"
)


# Group production runs by week.

tdf["week"] = (
    tdf["ts"]
    .dt
    .to_period("W")
    .dt
    .start_time
)


wk = (
    tdf
    .groupby("week")
    .agg(
        n=("fail", "size"),
        f=("fail", "sum")
    )
    .reset_index()
)


# Ignore weeks with fewer than 10 runs
# because extremely small samples can produce
# unstable failure-rate estimates.

wk = wk[
    wk["n"] >= 10
]


# Overall average failure rate

pbar = (
    wk["f"].sum()
    / wk["n"].sum()
)


# Weekly failure rate

wk["p"] = (
    wk["f"]
    / wk["n"]
)


# ============================================================
# 3-SIGMA CONTROL LIMITS
# ============================================================

wk["ucl"] = (
    pbar
    + 3
    * np.sqrt(
        pbar
        * (1 - pbar)
        / wk["n"]
    )
)


wk["lcl"] = np.clip(
    pbar
    - 3
    * np.sqrt(
        pbar
        * (1 - pbar)
        / wk["n"]
    ),
    0,
    None
)


# Identify weeks outside control limits.

out_ctrl = wk[
    (wk["p"] > wk["ucl"])
    |
    (wk["p"] < wk["lcl"])
]


# ============================================================
# FIGURE 3 — WEEKLY FAILURE RATE P-CHART
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 4)
)


ax.plot(
    wk["week"],
    wk["p"] * 100,
    "o-",
    color="#4C72B0",
    label="Weekly fail rate"
)


ax.axhline(
    pbar * 100,
    color="green",
    label=f"Mean {pbar * 100:.1f}%"
)


ax.plot(
    wk["week"],
    wk["ucl"] * 100,
    "r--",
    label="UCL (3σ)"
)


ax.plot(
    wk["week"],
    wk["lcl"] * 100,
    "r--",
    label="LCL (3σ)"
)


ax.scatter(
    out_ctrl["week"],
    out_ctrl["p"] * 100,
    color="crimson",
    zorder=5,
    s=70,
    label="Out of control"
)


ax.set_ylabel(
    "Fail rate (%)"
)


ax.set_title(
    "Weekly fail rate p-chart"
)


ax.legend(
    fontsize=8,
    ncol=5,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.1
    )
)


plt.tight_layout()


plt.savefig(
    f"{OUT}/fig3_yield_trend.png",
    dpi=150
)


plt.close()


print(
    f"{len(out_ctrl)} of {len(wk)} weeks "
    f"outside 3-sigma control limits"
)


# ============================================================
# 5. MACHINE LEARNING MODEL CHECK
# ============================================================

# Use stratified 5-fold cross-validation.
#
# Stratification helps maintain a similar
# pass/fail ratio in each fold.
#
# Imputation is performed inside the pipeline
# so information from test folds does not leak
# into training.
#
# Balanced class weights are used because
# failures are much less common than passes.


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)


models = {

    "Logistic regression":

        make_pipeline(

            SimpleImputer(
                strategy="median"
            ),

            StandardScaler(),

            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=0.05
            )
        ),


    "Random forest":

        make_pipeline(

            SimpleImputer(
                strategy="median"
            ),

            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                min_samples_leaf=3,
                random_state=SEED,
                n_jobs=-1
            )
        )
}


aucs = {}


# ============================================================
# FIGURE 4 — CROSS-VALIDATED ROC CURVES
# ============================================================

fig, ax = plt.subplots(
    figsize=(5.5, 5)
)


for name, model in models.items():

    # Generate predictions only on held-out folds.

    probabilities = cross_val_predict(
        model,
        X1,
        y,
        cv=cv,
        method="predict_proba"
    )[:, 1]


    # Calculate ROC-AUC.

    aucs[name] = roc_auc_score(
        y,
        probabilities
    )


    # Generate ROC curve.

    fpr, tpr, _ = roc_curve(
        y,
        probabilities
    )


    ax.plot(
        fpr,
        tpr,
        label=(
            f"{name} "
            f"(AUC {aucs[name]:.2f})"
        )
    )


# Random-classifier baseline

ax.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random (0.50)"
)


ax.set_xlabel(
    "False positive rate"
)


ax.set_ylabel(
    "True positive rate"
)


ax.set_title(
    "5-fold cross-validated ROC"
)


ax.legend(
    fontsize=8
)


plt.tight_layout()


plt.savefig(
    f"{OUT}/fig4_model_check.png",
    dpi=150
)


plt.close()


# Find the best-performing model.

best_name = max(
    aucs,
    key=aucs.get
)


print(
    {
        name: round(score, 3)
        for name, score in aucs.items()
    }
)


# ============================================================
# 6. CREATE FINDINGS SUMMARY
# ============================================================

top3 = ", ".join(
    res["sensor"].head(3)
)


summary = f"""SECOM YIELD ANALYSIS: KEY FINDINGS
==================================

Runs analysed:                {n_runs}
Sensor signals:               {n_sensors}
Failed runs:                  {n_fail}
Yield:                        {yield_pct:.1f}%
Fail rate:                    {100 * n_fail / n_runs:.1f}%

Dropped (> {int(MISS_CUTOFF * 100)}% missing):     {len(drop_missing)}
Dropped (constant):           {len(const_cols)}
Sensors analysed:             {n_kept}

Significant sensors (q<0.05): {n_sig}
Top 3 sensors by p-value:     {top3}

Weeks outside control limits: {len(out_ctrl)} of {len(wk)}

Cross-validated model performance:
{chr(10).join(f'- {name}: ROC-AUC {score:.2f}' for name, score in aucs.items())}
"""


# Save findings summary.

with open(
    f"{OUT}/findings_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(summary)


# Print findings in Terminal.

print(
    "\n" + summary
)


print(
    f"Files written to ./{OUT}/"
)