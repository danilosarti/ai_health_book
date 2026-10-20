"""Chapter 8 — BART (Bayesian additive regression trees), precomputed, with honest test-set
comparisons against random forests, XGBoost and linear regression.

Usage (from the book folder):  python3 scripts/precompute/ch08_bart.py
Needs: numpy, pandas, scipy, scikit-learn, xgboost, pymc, pymc-bart (pip install pymc-bart).
Writes small CSV/JSON files to data/precomputed/ch08/. Runtime: a few minutes on one core.

The v1 text fitted BART with the R package dbarts; the same model can be fitted there with
dbarts::bart(x.train, y.train, x.test, ntree = 50). pymc-bart is used here so that the whole
precomputation runs in Python. MCMC is kept small: 2 chains, 1 000 warm-up + 1 000 draws.
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pymc as pm, pymc_bart as pmb

SEED = 2026
OUT = "data/precomputed/ch08"
os.makedirs(OUT, exist_ok=True)


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mae(y, p): return float(np.mean(np.abs(y - p)))


def fit_bart(Xtr, ytr, Xnew_list, m=50, seed=SEED):
    """Fit BART on (Xtr, ytr); return posterior draws of f(x) and of a new y for each X in Xnew_list."""
    with pm.Model() as model:
        Xd = pm.Data("X", Xtr)
        mu = pmb.BART("mu", Xd, ytr, m=m)
        sigma = pm.HalfNormal("sigma", float(np.std(ytr)))
        pm.Normal("y", mu, sigma, observed=ytr, shape=mu.shape)
        idata = pm.sample(draws=1000, tune=1000, chains=2, cores=1, random_seed=seed,
                          progressbar=False)
        res = []
        for Xn in Xnew_list:
            pm.set_data({"X": Xn})
            pp = pm.sample_posterior_predictive(idata, var_names=["mu", "y"], sample_vars=["mu", "y"], random_seed=seed,
                                                progressbar=False)
            f = pp.posterior_predictive["mu"].values.reshape(-1, Xn.shape[0])   # draws x patients
            yn = pp.posterior_predictive["y"].values.reshape(-1, Xn.shape[0])
            res.append((f, yn))
    sig = idata.posterior["sigma"].values
    return res, sig


def ols_interval(Xtr, ytr, Xte, level=0.95):
    A = np.column_stack([np.ones(len(Xtr)), Xtr]); B = np.column_stack([np.ones(len(Xte)), Xte])
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    res = ytr - A @ beta; dof = len(ytr) - A.shape[1]; s2 = res @ res / dof
    XtXi = np.linalg.inv(A.T @ A)
    se = np.sqrt(s2 * (1 + np.einsum("ij,jk,ik->i", B, XtXi, B)))
    q = stats.t.ppf(0.5 + level / 2, dof)
    pred = B @ beta
    return pred, pred - q * se, pred + q * se


rows = []

# ---- 1. HomeBeat: resting heart rate from three baseline characteristics -------------
hb = pd.read_csv("data/homebeat.csv")
feat_hb = ["age", "frailty", "baseline_inflammation"]
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(hb)); tr = np.zeros(len(hb), bool); tr[idx[:400]] = True   # 400 / 100
X = hb[feat_hb].to_numpy(float); y = hb["resting_hr"].to_numpy(float)
Xtr, Xte, ytr, yte = X[tr], X[~tr], y[tr], y[~tr]

lena = np.array([[72, 0.55, 4.1]])                     # Lena's profile, predicted with the test set
(res_te, res_tr), sig_hb = fit_bart(Xtr, ytr, [np.vstack([Xte, lena]), Xtr])
f_te, y_te = res_te[0][:, :-1], res_te[1][:, :-1]
f_l, y_l = res_te[0][:, -1], res_te[1][:, -1]
f_tr, _ = res_tr
p_bart = f_te.mean(axis=0)              # posterior mean for each test participant (mean over DRAWS)
lo, hi = np.quantile(y_te, [0.025, 0.975], axis=0)
rows.append(dict(data="homebeat", model="BART (50 trees)", train_RMSE=rmse(ytr, f_tr.mean(axis=0)),
                 test_RMSE=rmse(yte, p_bart), test_MAE=mae(yte, p_bart),
                 coverage95=float(np.mean((yte >= lo) & (yte <= hi))), width95=float(np.mean(hi - lo))))
pl, l_lo, l_hi = ols_interval(Xtr, ytr, Xte)
pl_tr, _, _ = ols_interval(Xtr, ytr, Xtr)
rows.append(dict(data="homebeat", model="Linear regression", train_RMSE=rmse(ytr, pl_tr),
                 test_RMSE=rmse(yte, pl), test_MAE=mae(yte, pl),
                 coverage95=float(np.mean((yte >= l_lo) & (yte <= l_hi))), width95=float(np.mean(l_hi - l_lo))))
rf = RandomForestRegressor(n_estimators=500, max_features=1 / 3, random_state=SEED, n_jobs=1).fit(Xtr, ytr)
rows.append(dict(data="homebeat", model="Random forest (500 trees)", train_RMSE=rmse(ytr, rf.predict(Xtr)),
                 test_RMSE=rmse(yte, rf.predict(Xte)), test_MAE=mae(yte, rf.predict(Xte))))
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED, n_jobs=1).fit(Xtr, ytr)
rows.append(dict(data="homebeat", model="XGBoost (300 rounds)", train_RMSE=rmse(ytr, xgb.predict(Xtr)),
                 test_RMSE=rmse(yte, xgb.predict(Xte)), test_MAE=mae(yte, xgb.predict(Xte))))
rows.append(dict(data="homebeat", model="Average (no model)", train_RMSE=rmse(ytr, ytr.mean()),
                 test_RMSE=rmse(yte, ytr.mean()), test_MAE=mae(yte, ytr.mean())))

# The v1 mistake, reproduced on purpose: averaging over PATIENTS instead of over draws
v1_bug = dict(rmse_wrong_average=rmse(yte, f_te.mean(axis=1)[: len(yte)]),
              rmse_right_average=rmse(yte, p_bart))

# Lena's profile under BART
lena_bart = dict(mean_hr=float(f_l.mean()), mean_lo=float(np.quantile(f_l, 0.025)),
                 mean_hi=float(np.quantile(f_l, 0.975)), new_lo=float(np.quantile(y_l, 0.025)),
                 new_hi=float(np.quantile(y_l, 0.975)), p_new_above_88=float(np.mean(y_l > 88)))

# ---- 2. Ward 7: change in creatinine, shared split, predictors as in Chapter 6 -------------
icu = pd.read_csv("data/icu_aki.csv")
sp7 = pd.read_csv("data/splits/ward7_split.csv").set_index("patient_id").loc[icu["patient_id"]].reset_index()
tr7 = (sp7["set"] == "train").to_numpy()
feat = ["site", "age", "sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence", "bun", "crp", "albumin"]
tonight = pd.read_csv("data/ward7_tonight.csv")
allX = pd.get_dummies(pd.concat([icu[feat], tonight[feat]]), columns=["site"], drop_first=True, dtype=float)
X7 = allX.iloc[: len(icu)].to_numpy(float); Xt7 = allX.iloc[len(icu):].to_numpy(float)
y7 = icu["delta_creatinine_48h"].to_numpy(float)
(r_te7, r_tr7), sig7 = fit_bart(X7[tr7], y7[tr7], [np.vstack([X7[~tr7], Xt7]), X7[tr7]])
nte = int((~tr7).sum())
f7, yn7 = r_te7[0][:, :nte], r_te7[1][:, :nte]
r_tn7 = (r_te7[0][:, nte:], r_te7[1][:, nte:])
p7 = f7.mean(axis=0); lo7, hi7 = np.quantile(yn7, [0.025, 0.975], axis=0)
yte7 = y7[~tr7]
rows.append(dict(data="ward7", model="BART (50 trees)", train_RMSE=rmse(y7[tr7], r_tr7[0].mean(axis=0)),
                 test_RMSE=rmse(yte7, p7), test_MAE=mae(yte7, p7),
                 coverage95=float(np.mean((yte7 >= lo7) & (yte7 <= hi7))), width95=float(np.mean(hi7 - lo7))))
pl7, a7, b7 = ols_interval(X7[tr7], y7[tr7], X7[~tr7])
rows.append(dict(data="ward7", model="Linear regression", test_RMSE=rmse(yte7, pl7), test_MAE=mae(yte7, pl7),
                 coverage95=float(np.mean((yte7 >= a7) & (yte7 <= b7))), width95=float(np.mean(b7 - a7))))
# AKI on the test set judged from the BART predictive distribution: P(change >= 0.3)
p_aki_test = (yn7 >= 0.3).mean(axis=0)
pd.DataFrame({"patient_id": icu["patient_id"][~tr7].to_numpy(), "delta": yte7, "f_mean": p7.round(4),
              "lo95": lo7.round(4), "hi95": hi7.round(4), "p_aki": p_aki_test.round(4)}).to_csv(
    os.path.join(OUT, "ward7_bart_test.csv"), index=False)
f_tn, y_tn = r_tn7
tn = pd.DataFrame({"name": tonight["name"], "f_mean": f_tn.mean(axis=0), "f_lo": np.quantile(f_tn, 0.025, axis=0),
                   "f_hi": np.quantile(f_tn, 0.975, axis=0), "y_lo": np.quantile(y_tn, 0.025, axis=0),
                   "y_hi": np.quantile(y_tn, 0.975, axis=0), "p_aki": (y_tn >= 0.3).mean(axis=0)}).round(4)
tn.to_csv(os.path.join(OUT, "ward7_bart_tonight.csv"), index=False, encoding="utf-8")

pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "bart_comparison.csv"), index=False)
import pymc_bart, pymc, sklearn, xgboost
json.dump({"homebeat_split": "rng = np.random.default_rng(2026); first 400 of a permutation = training",
           "seed": SEED, "lena_bart": lena_bart, "v1_bug": v1_bug,
           "sigma_homebeat": float(sig_hb.mean()), "sigma_ward7": float(sig7.mean()),
           "versions": {"pymc": pymc.__version__, "pymc_bart": pymc_bart.__version__,
                        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
                        "numpy": np.__version__}},
          open(os.path.join(OUT, "bart_summary.json"), "w"), indent=1)
print(pd.DataFrame(rows).round(3).to_string())
print(tn.to_string())
print(lena_bart, v1_bug)

# The same extra numbers as a two-column CSV, easy to read with base R
extras = {**{"lena_" + k: v for k, v in lena_bart.items()}, **v1_bug,
          "sigma_homebeat": float(sig_hb.mean()), "sigma_ward7": float(sig7.mean())}
pd.DataFrame({"key": list(extras), "value": [round(v, 4) for v in extras.values()]}).to_csv(
    os.path.join(OUT, "bart_extras.csv"), index=False)
