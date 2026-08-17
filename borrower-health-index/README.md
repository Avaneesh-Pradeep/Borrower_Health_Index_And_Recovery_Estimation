# Borrower Health Index

**A behavioural credit scoring and early-warning system for consumer lending.**

This project investigates whether a commercial credit bureau score can be reconstructed from repayment behaviour alone, and extends the result into a monthly monitoring system that forecasts borrower deterioration and quantifies expected recovery.

Dataset: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).

---

## Objective

The Home Credit dataset includes `EXT_SOURCE_1/2/3` — proprietary external credit scores derived from credit bureau records covering a borrower's complete history across all lenders. These are among the strongest predictors available in the dataset.

A bureau score draws on a borrower's entire credit footprint. A lender's own installment records contain far less breadth, but substantially more granularity: exact payment timing, partial settlements, and prioritisation between competing obligations.

This project quantifies the trade-off:

> **How much of a commercial bureau score can be reconstructed from repayment behaviour alone?**

### Result

| Signal | Test AUC | 95% CI |
|---|---|---|
| **Behavioural pipeline** | **0.730** | [0.681, 0.779] |
| Home Credit external bureau score | 0.722 | [0.672, 0.772] |

The behavioural score achieves parity with the commercial bureau score. Repayment patterns observable to a single lender carry equivalent default signal to a score assembled from a borrower's full credit history.

---

## Why behavioural data is sufficient

Bureau scores are comprehensive but coarsely sampled. Transaction records are narrow but densely sampled, and contain three classes of signal unavailable at bureau level.

**Payment fragmentation.** The installment data records multiple payment rows against a single due installment — a €400 obligation settled as €150, €150, and €100 across several days. The bureau records this as paid in full and on time. The transaction record reveals that the borrower could not settle it in one transaction, which is a liquidity constraint that materialises well before formal delinquency.

**Timing volatility as distinct from lateness.** Two borrowers may both average five days late. One is consistently five days late, reflecting a stable payday cycle. The other alternates between ten days early and fifteen days late, reflecting irregular income. The rolling mean is identical; the risk profile is not. `PAY_WINDOW_STD_3M` captures dispersion separately from central tendency.

**Selective prioritisation.** Borrowers holding multiple obligations do not default uniformly. Under liquidity pressure they triage, protecting some obligations while allowing others to lapse. A Gaussian mixture model infers which obligation is being prioritised, and `SELECTIVE_FAILURE` quantifies the divergence in payment performance between the highest- and lowest-priority obligations in a given month. This behaviour precedes outright default.

---

## Methodological framing

Conventional credit modelling treats default as a binary classification problem: given a snapshot of a borrower, predict whether default occurs.

This project adopts a different framing drawn from industrial reliability engineering. Physical assets are not modelled as functional or failed; they are assigned a continuously measured condition, and Remaining Useful Life models forecast the degradation trajectory so that intervention can be scheduled prior to failure.

| Industrial asset monitoring | Consumer credit |
|---|---|
| Measured asset condition | Health Score H(t) |
| Degradation rate | Forecast ΔH(t+1) |
| Remaining Useful Life | Time to default |
| Maintenance scheduling | Collections prioritisation |
| Cost of unplanned failure | Unrecovered balance |

The resulting system maintains a health score that updates monthly, forecasts its trajectory, issues alerts with quantified lead time, and translates the forecast into an expected recovery figure.

---

## Architecture

```
Payment history (installments_payments.csv)
        │
        ├─ Gibbs sampler ──────────► baseline financial stress θᵢ
        │                            (early window, then frozen)
        │
        ├─ Gaussian mixture ───────► payment priority
        │                            (obligation-level prioritisation)
        ▼
Logistic scorecard ────────────────► Health Score H(t) ∈ [0,1]
        │
        ├─ XGBoost ────────────────► forecast ΔH(t+1)
        │                            (logit space, fully lagged features)
        │
        ├─ Recursive simulation ───► trajectory, alerts, monitoring tiers
        │
        ├─ Cox proportional hazards► survival validation (C-index 0.798)
        │
        └─ Isotonic regression ────► expected recovery, next installment
```

### Baseline stress estimation

Each borrower is assigned a latent financial stress parameter θᵢ, estimated by Gibbs sampling under the model:

```
payment_ratio_ij = 1 − θᵢ·(1 − pᵢⱼ) + ε,    ε ~ N(0, σ²)
```

A borrower under stress underpays, except on obligations they treat as high priority. Both the borrower-level stress θᵢ and the obligation-level priority pᵢⱼ are estimated jointly; the full conditionals are linear-Gaussian, so each Gibbs update reduces to a closed-form truncated-normal draw.

θᵢ is estimated exclusively from each borrower's initial observation window and then frozen, ensuring no row is scored using a parameter informed by its own future.

### Health score

The health score is a logistic scorecard predicting probability of good standing, consistent with standard credit scorecard construction. Regularisation strength is selected by cross-validation, which is material here: under weak regularisation, four correlated delinquency features assume large opposing coefficients and `MAX_DPD` acquires a positive sign. Stronger L2 shrinkage restores coefficient coherence.

Temporal smoothing uses a trailing median window (t−2, t−1, t), avoiding the forward information transfer inherent in a centered window.

### Trajectory forecasting

The forecast target is Δlogit(H) rather than ΔH. Since H is bounded on [0,1], an additive step carries different meaning depending on position within the interval. Operating in logit space renders the step size scale-invariant, and the inverse sigmoid transform enforces the [0,1] bound by construction rather than by post-hoc clipping.

Defaulter observations are upweighted by inverse frequency to prevent the majority class from dominating the loss.

### Recursive simulation

Forward simulation feeds each prediction into the subsequent step, with the following controls governing multi-step behaviour:

- **Velocity clipping** to the empirical training range
- **Momentum blending** under time-decay weighting
- **Feature capping** at the 99th training percentile, constraining the simulation to the model's observed input space
- **Phase scaling**, applying lenient adjustment during the initial observation period and stricter adjustment once behaviour is established
- **Persistence gating**, requiring a directional shift to repeat across consecutive periods before amplification is applied

The simulation produces trajectory forecasts, alert timing with quantified lead time, and monitoring tier assignment — outputs that a point-in-time score does not provide.

### Expected recovery

An isotonic regression maps forecasted health to expected payment ratio under a monotonicity constraint, ensuring higher forecast health cannot imply lower expected payment. Multiplied by the outstanding installment amount, this yields a per-account expected shortfall, enabling collections prioritisation prior to the payment date.

---

## Results

### Detection performance

```
Detection rate (TPR):   64.5%     design target ≥ 60%
False positive rate:    30.1%     design target < 30%
Precision:              16.2%     base rate 8.2%
Lift:                   1.96×
Test AUC:               0.730
```

Flagged accounts default at approximately twice the population rate. The operating threshold was selected by sweeping the ROC curve on training data for the constrained operating point, then applied without modification to the test set.

### Signal comparison

| Signal | Train AUC | Test AUC |
|---|---|---|
| Raw health score (behavioural) | 0.7134 | 0.7333 |
| Trajectory + external combined | 0.7119 | 0.7302 |
| Simulated min health (behavioural) | 0.7136 | 0.7300 |
| **Home Credit external bureau score** | 0.6975 | **0.7220** |
| Direct XGBoost (out-of-fold) | 0.6614 | 0.7073 |
| MAX_DPD alone | 0.5135 | 0.5178 |

At a standard error of approximately 0.025, the behavioural signals and the bureau score are statistically indistinguishable — establishing parity from a substantially narrower information set.

The staged pipeline also outperforms a direct XGBoost classifier trained on identical features (0.730 vs 0.707), indicating that decomposing the problem into scoring, forecasting, and simulation stages generalises more effectively than a single unconstrained model.

### Survival validation

```
Cox proportional hazards, C-index: 0.798
```

Specified as one observation per borrower, with durations derived from observed day-spans. Confirms the health score as a strong time-to-default predictor.

---

## Data integrity audit

The initial pipeline reported R² = 0.985. A systematic audit identified seven defects, four constituting information leakage. Corrected, the model reports R² = 0.32.

**Leakage defects**

**1. Latent stress estimated across full borrower history.** An observation at month 5 carried a feature informed by months 6 onward, transmitting future information through an apparently static variable.

**2. Mixture models fit prior to train/test partition.** Cluster boundaries were informed by test-set borrowers, compromising holdout integrity.

**3. Acceleration feature reconstructing its own target.** The feature was constructed as `RAW_CHANGE[t] − VELOCITY_LAG1[t]` using the current period. The prediction target was an exponentially-weighted mean of that same `RAW_CHANGE[t]`, weighting it at approximately 67%, yielding:

```
target[t] ≈ 0.67 · ACCELERATION[t] + VELOCITY_LAG1[t]
```

The model was solving an algebraic identity rather than learning a relationship. These two features accounted for 55% of total feature importance.

**4. Contemporaneous features predicting same-period change.** Features describing month t were the same inputs used to construct the health score at month t, rendering the prediction circular. This also created a mismatch between training conditions and simulation conditions, where only lagged information is available.

Correction of defects 1–4 reduced R² from 0.985 to 0.32, with top-feature importance falling from 55% to 6.5%.

**Methodological defects**

**5. Cox model misspecification.** Fit at one observation per borrower-month, treating repeated correlated measurements of the same subject as independent. Respecified to one observation per borrower.

**6. Positional rather than temporal lagging.** Row-position shifts treated multi-month gaps as single-month transitions. A borrower observed at months 3, 4, 7, 8 resolved month 7's prior observation to month 4. Correction required two iterations: an explicit gap check discarded 84% of observations, since floor-based bucketing against a global grid misclassifies consecutive payments under real ~30.44-day schedules. Anchoring bucket boundaries to each borrower's first installment reduced loss to 56% and increased the effective test sample from 633 to 1,673 borrowers.

**7. In-sample scoring of the baseline classifier.** The classifier was fit and scored on the same partition, reporting AUC 0.998 and winning signal selection before failing on test. This defect did not inflate a reported metric; it silently corrupted a model selection decision. Corrected with out-of-fold scoring.

---

## Reproduction

```bash
pip install -r requirements.txt
```

Place `installments_payments.csv` and `application_train.csv` at the path specified in `DATA_DIR` (default `/content` for Google Colab) and execute the notebook sequentially. The loader raises on missing data rather than substituting synthetic values.

---

## Repository contents

```
notebooks/borrower_health_index.ipynb    Full pipeline with execution outputs
REPORT.pdf                               Technical report
report/report.tex                        Report source (LaTeX)
report/figures/                          Figures, extracted from notebook outputs
requirements.txt                         Dependencies
LICENSE                                  MIT
```
