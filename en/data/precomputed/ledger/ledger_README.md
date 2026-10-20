# Model ledger: definitions

One source of truth for every model result of the book's running examples. Every number for the
same model is computed once, here, on the same data, split, predictors and metrics; chapters read
it with `scripts/ledger_helpers.R` (`ledger()`, `ledger_table()`, `ledger_tonight()`,
`ledger_patient()`, `ledger_lena()`).

Produced by `scripts/precompute/ledger.R`, which starts `scripts/precompute/ledger.py` (scikit-learn,
XGBoost, PyTorch, pymc-bart) in the background and computes all metrics in R from the predictions:

```
LEDGER_PYTHON=<venv-with-pymc-bart>/bin/python Rscript scripts/precompute/ledger.R
Rscript scripts/precompute/ledger.R --skip-py      # R part only, reuse parts/py_preds.csv
```

About 3 minutes on 2 CPUs. The pymc-bart environment is described at the top of `ledger.py`
(pymc 6.3.2, pymc-bart 0.13.1, `--system-site-packages`). Deterministic: fixed seeds everywhere;
glmnet, rpart and every CV use the shared folds.

## Splits

| Example | File | Train / test | CV |
|---|---|---|---|
| Ward 7 | `data/splits/ward7_split.csv` | 1 000 / 250 | 5 folds of 200 |
| ONCO-1 | `data/splits/onco1_split.csv` | 800 / 200 (chemotherapy arm: 412 / 84) | 5 folds of 160 |
| HomeBeat | `data/splits/homebeat_split.csv` (new) | 400 / 100 | 5 folds of 80 |

`homebeat_split.csv` is written by `scripts/simulate/make_homebeat_split.py`. It stores the
permutation `np.random.default_rng(2026).permutation(500)` that Chapters 8 and 19 use (column
`perm_order`; training = first 400), so their numbers do not move.

## Predictor sets

- **Ward 7 P10** (canonical, regression): site, age, sofa, lactate, map_min, baseline_creat,
  protocol_adherence, bun, crp, albumin (site as 3 dummy columns).
- **Ward 7 P5** (canonical, AKI): sofa, lactate, map_min, baseline_creat, protocol_adherence.
- **Ward 7 P2** (teaching set, Chapter 3): sofa, baseline_creat.
- **ONCO-1, response %** (all patients): treatment, dose_intensity, patient_age, tumor_grade,
  performance_score + the 47 non-leaky genes (53 columns). The "with leaky genes 08/14/19" row is
  the leakage warning only.
- **ONCO-1, high response** (chemotherapy arm only): dose, age, grade, performance + 47 genes
  ("all 51 predictors"); the 7-predictor model uses dose, grade, performance, age, genes 01/05/12.
  Genes are standardised with the mean and SD of the **training** rows. (Chapter 5 used the whole
  arm; this changes no published number, because unpenalised logistic regression is invariant to
  rescaling and glmnet re-standardises internally.)
- **HomeBeat**: resting_hr ~ age + frailty + baseline_inflammation (the Chapter 8 task).
  stress_score (Chapters 9 and 19) is a separate task. Its Chapter 19 TSTR numbers are copied
  as published (rows with `role = separate task`).

## Files

| File | Content |
|---|---|
| `ward7_reg.csv`, `ward7_aki.csv`, `onco1_reg.csv`, `onco1_class.csv`, `homebeat.csv` | one row per model |
| `ward7_tonight.csv` | Aoife, Tomás, Priya: Δ creatinine (`example = ward7_reg`) and P(AKI) (`ward7_aki`) from every model |
| `onco1_patients.csv` | the two named ONCO-1 patients, their real data and every model's prediction |
| `homebeat_lena.csv` | Lena's predicted mean resting heart rate from every model (+ the Chapter 8 all-data value as reference) |
| `versions.json` | package versions, seed, patient mapping, run time |
| `parts/` | intermediate predictions (`py_preds.csv`: example, model_id, kind, id, pred) |

### Columns of the model tables

| Column | Meaning |
|---|---|
| `order` | position in the ladder, in the order the chapters introduce the models |
| `chapter` | chapter that introduces the model |
| `model_id` | stable key used by the helpers (`ledger("ward7_reg", "ols", "test_MAE")`) |
| `model` | exact name to print, e.g. "Ridge logistic regression (C = 1, scikit-learn; …)" |
| `predictor_set` | see above |
| `role` | `canonical` (ladder), `reference` (no-model baseline), `teaching` (P2), `variant` (an old chapter implementation kept so the chapter can label it), `warning` (leakage), `second predictor set` (BART → AKI, P10), `separate task` (HomeBeat stress score) |
| `n_train`, `n_test` | rows used to fit and to test |
| `cv_MAE` / `cv_logloss` | 5-fold CV on the shared folds (see below); empty for BART and copied rows |
| `test_MAE`, `test_RMSE`, `test_R2` | regression on the test set. R² = 1 − SSE / Σ(y − ȳ_train)², i.e. against the **training** mean (can be < 0) |
| `test_AUC` | rank AUC (ties count ½) |
| `test_Brier` | mean (p − y)² |
| `cal_slope` | slope b of logit P(y = 1) = a + b·logit(p) (p clipped to [1e-6, 1 − 1e-6]); empty if p is constant or the fit does not converge |
| `cal_intercept` | calibration-in-the-large: a in logit P(y = 1) = a + logit(p) (offset), by maximum likelihood. Positive = the model under-predicts |
| `tuning` | the chosen tuning value(s) and how they were chosen |
| `software` | package and version |

**CV definition.** Out-of-fold predictions on the 5 shared folds. Each model's tuning value (λ, k,
cp, number of rounds or epochs) is held at the value chosen on the full training set, and the model
is refitted on 4 folds and predicts the 5th. For tuned models this is slightly optimistic, which is
the usual cost of not nesting. Log-loss clips p to [1e-15, 1 − 1e-15]: pure tree leaves (p = 0 or 1)
give large log-losses, which is correct.

**Tuning (all on the shared folds).** glmnet: λ min and λ 1-SE by `cv.glmnet(foldid = …)`. The
elastic net α is chosen from 0, 0.1, …, 1 by the lowest CV MSE, as in Chapter 4. For Ward 7 the
chosen α is 1, so the "elastic net" rows equal the lasso rows. rpart: the CV inside rpart uses
`xval = folds`, followed by the 1-SE rule. kNN: k from 1–150 by CV MAE. XGBoost: the number of
rounds minimises the mean validation curve over the 5 folds (MAE or log-loss), and the model is
then refitted on all training rows. This replaces Chapter 6's early stopping on a random 20 %.
MLPs: the number of epochs is chosen the same way.

## Named patients

**Tonight on Ward 7.** Aoife, Tomás and Priya come from `data/ward7_tonight.csv`. Every model is
fitted on the 1 000 training patients. Rows with `example = ward7_reg` give Δ creatinine in mg/dL,
and rows with `ward7_aki` give P(AKI). The MI row predicts tonight's patients, who have complete
data, with the average of the 20 imputed-data OLS fits.

**ONCO-1 #0042 and #0917 (one identity book-wide).** These are two real trial patients in
`onco1.csv`. They teach the lesson "similar profile, different response, uncertain prediction".

The pair was chosen by these criteria, applied by an exhaustive search over all chemotherapy-arm
**test** patients (84):

1. **Close clinical profiles.** Same tumour grade, ages within 12 years, dose within 0.12 and
   performance score within 1.
2. **Different outcomes.** One patient has `high_response = 1` and the other 0, with clearly
   different shrinkage.
3. **Moderate, different probabilities from the headline classifier.** The headline classifier is
   the **lasso logistic regression (λ min)** of Chapter 5 (`onco1_class` / `lasso_logit`). Target
   bands were about 0.55–0.9 for one patient and 0.15–0.5 for the other. The 7-predictor logistic
   was not used as the headline because it is too confident for most matched pairs (0.78 / 0.02
   for this pair).
4. **Regression predictions clearly different but not perfect.**

The ids that literally contain 042 / 917 (P1042, P6042, P3917) were rejected. All three are
untreated training patients with near-zero response. The earlier pick (P2217 / P4623) was also
dropped: every model gave it P ≈ 1.00 / 0.00, so it showed no uncertainty.

Chosen pair: both in the test set, chemotherapy arm, grade G1, the same dose (1.07) and the same
performance score (2).

| Label | patient_id | Age | Dose | Grade | Performance | Response % | High response | Lasso logistic P | Lasso (λ min) predicted response % |
|---|---|---|---|---|---|---|---|---|---|
| #0042 | P5440 | 37 | 1.072 | G1 | 2 | 36.6 | 1 | 0.75 | 32.1 |
| #0917 | P9541 | 46 | 1.067 | G1 | 2 | 20.4 | 0 | 0.18 | 26.3 |

The other models agree on the direction but not on the size:

- Elastic net: 0.83 / 0.21.
- Ridge: 0.81 / 0.37.
- Random forest: 0.90 / 0.42.
- XGBoost: 0.80 / 0.22.
- Network head: 0.92 / 0.12.
- 7-predictor logistic: 0.78 / 0.02.
- Regression models predict 30–36 % vs 22–30 %. Every model underestimates #0042's response and
  overestimates #0917's.

All predictions are in `onco1_patients.csv`. Chapters 4, 5, 7, 10 and 13 present them with this
identity: two trial patients with the same clinical profile and different responses.

**Lena (HomeBeat).** Profile: age 72, frailty 0.55, baseline inflammation 4.1 (Chapter 8). Every
ledger model is fitted on the 400 training participants. Chapter 8's inference model on all 500
participants (85.2 bpm) is listed as a reference row.

## Implementations reused (values that should not move do not move)

- Chapter 3 baselines, P2 line and kNN P2: `ch03_workflow.R` code, identical values.
- Chapter 4 ONCO-1 OLS / ridge / lasso / elastic net: `ch04_glmnet.R` code, identical values.
- Chapter 5 logistic models (Ward 7 and ONCO-1): `ch05_precompute.R` code, identical values.
- Chapter 6 rpart trees: chunk `ward7-prune`. sklearn trees, forests, bagging and the ONCO-1
  C = 1 logistic come from `ch06_ensembles.py`. All give identical values. XGBoost changes slightly
  because its rounds are now chosen by CV.
- Chapter 7 multi-output network: the `ch07_pneumonia_cnn.py` architecture and loop, refitted on the
  shared split. The CNN is not retrained.
- Chapter 8 Bayesian logistic (`ch08_stan.R` call) and BART (`ch08_bart.py` code). Both are
  identical; the BART predictions match `ward7_bart_test.csv` to the 4th decimal.
- Chapter 9 Lab imputation, now with `site` (P10).
- Chapter 12 forest = Chapter 6 forest (0.1528).
