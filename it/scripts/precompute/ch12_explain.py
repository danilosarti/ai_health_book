"""Chapter 12 - precomputed explanations (permutation importance, PDP/ICE, H statistics,
Shapley values, bootstrap uncertainty of explanations).

Everything heavy in Chapter 12 is computed here, once, and saved as small CSV/JSON files in
data/precomputed/ch12/. The chapter reads these files; nothing here runs in the browser.

Data and splits
* Ward 7 (data/icu_aki.csv) with the book's shared split data/splits/ward7_split.csv. The random
  forest is the one of Chapter 6 (same features, settings and seed as
  scripts/precompute/ch06_ensembles.py), refitted here so that it can be probed.
* Lung follow-up data (data/lung_sim_complete.csv) with data/splits/lung_split.csv
  (made by scripts/simulate/make_ch12_data.R).
* The small oncology study (data/med_small_sim.csv), split 75/25 here with seed 2026.

Usage (from the book folder):  python3 scripts/precompute/ch12_explain.py
Requires numpy, pandas, scikit-learn, xgboost. Single-threaded; about three minutes.
"""
import itertools
import json
import math
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor

SEED = 2026
OUT = "data/precomputed/ch12/"
import os
os.makedirs(OUT, exist_ok=True)
mae = lambda y, p: float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
rmse = lambda y, p: float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def fit_xgb(Xtr, ytr, rounds=2000, eta=0.05, depth=3):
    """XGBoost with early stopping on 20% of the TRAINING data (as in Chapter 6)."""
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(ytr))
    nval = len(ytr) // 5
    va, fi = idx[:nval], idx[nval:]
    m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=eta, max_depth=depth, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.0, early_stopping_rounds=50,
                         random_state=SEED, n_jobs=1)
    m.fit(Xtr.iloc[fi], ytr[fi], eval_set=[(Xtr.iloc[va], ytr[va])], verbose=False)
    return m


def permutation_importance(predict, X, y, groups, R, rng, metric=mae):
    """Increase in error when a feature (or a group of columns) is shuffled.
    Importance = mean over R shuffles of error(permuted) - error(original): positive = the model relies on it."""
    base = metric(y, predict(X))
    out = {}
    for name, cols in groups.items():
        errs = []
        for _ in range(R):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            errs.append(metric(y, predict(Xp)))
        out[name] = errs
    return base, out


def pd_at_points(predict, X, cols, points):
    """Partial dependence of `predict` on the columns `cols`, evaluated at each row of `points`
    (averaging over the rows of X). Returns one value per row of points."""
    n = len(X)
    big = pd.concat([X] * len(points), ignore_index=True)
    vals = np.repeat(points[cols].to_numpy(), n, axis=0)
    big[cols] = vals
    return predict(big).reshape(len(points), n).mean(axis=1)


def h_statistics(predict, X, features):
    """Friedman-Popescu pairwise H (normalised) and the unnormalised version (output scale),
    computed at the observed points of X with centred partial dependence functions."""
    pd1 = {}
    for f in features:
        v = pd_at_points(predict, X, [f], X)
        pd1[f] = v - v.mean()
    res = []
    for a, b in itertools.combinations(features, 2):
        v = pd_at_points(predict, X, [a, b], X)
        v = v - v.mean()
        num = np.sum((v - pd1[a] - pd1[b]) ** 2)
        res.append({"var1": a, "var2": b, "H": float(np.sqrt(num / np.sum(v ** 2))),
                    "H_raw": float(np.sqrt(num / len(X)))})
    return pd.DataFrame(res)


# =============================================================== 1. Ward 7
icu = pd.read_csv("data/icu_aki.csv")
tonight = pd.read_csv("data/ward7_tonight.csv")
sp7 = pd.read_csv("data/splits/ward7_split.csv")
assert (sp7["patient_id"].to_numpy() == icu["patient_id"].to_numpy()).all()
tr7 = (sp7["set"] == "train").to_numpy()
feat = ["site", "age", "sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence",
        "bun", "crp", "albumin"]
both = pd.get_dummies(pd.concat([icu[feat], tonight[feat]], ignore_index=True),
                      drop_first=True, dtype=float)
X = both.iloc[:len(icu)].reset_index(drop=True)
Xn = both.iloc[len(icu):].reset_index(drop=True)
y = icu["delta_creatinine_48h"].to_numpy()
Xtr, Xte, ytr, yte = X[tr7].reset_index(drop=True), X[~tr7].reset_index(drop=True), y[tr7], y[~tr7]
groups7 = {f: [c for c in X.columns if c == f or c.startswith(f + "_")] for f in feat}

rf7 = RandomForestRegressor(n_estimators=500, max_features=1/3, oob_score=True, random_state=SEED,
                            n_jobs=1).fit(Xtr, ytr)
lin7 = LinearRegression().fit(Xtr, ytr)
print("Ward 7 forest test MAE", round(mae(yte, rf7.predict(Xte)), 4), "(Chapter 6: 0.1528)")

rng = np.random.default_rng(SEED)
perm7 = {"features": feat, "R": 30, "models": {}}
for label, model in [("Random forest", rf7), ("Linear regression", lin7)]:
    base, errs = permutation_importance(model.predict, Xte, yte, groups7, 30, rng)
    perm7["models"][label] = {"base_MAE": round(base, 4),
                              "perm_MAE": {k: [round(v, 4) for v in vals] for k, vals in errs.items()}}
json.dump(perm7, open(OUT + "ward7_permutation.json", "w"))

# --- exact Shapley values for tonight's patients (10 players; site = one player) ---------
bg = Xtr.iloc[np.random.default_rng(SEED).choice(len(Xtr), 200, replace=False)].reset_index(drop=True)
players = feat
P = len(players)
masks = list(itertools.product([0, 1], repeat=P))
weights = {s: math.factorial(s) * math.factorial(P - s - 1) / math.factorial(P) for s in range(P)}


def coalition_values(predict, xrow):
    """v(S) = average prediction when the players in S take the patient's values and the
    others take the values of each background patient (interventional Shapley values)."""
    nb = len(bg)
    big = pd.concat([bg] * len(masks), ignore_index=True)
    for k, m in enumerate(masks):
        for j, on in enumerate(m):
            if on:
                cols = groups7[players[j]]
                big.loc[k * nb:(k + 1) * nb - 1, cols] = xrow[cols].to_numpy()
    preds = predict(big).reshape(len(masks), nb).mean(axis=1)
    return {m: preds[k] for k, m in enumerate(masks)}


rows = []
for i, name in enumerate(tonight["name"]):
    xrow = Xn.iloc[i]
    for label, model in [("Random forest", rf7), ("Linear regression", lin7)]:
        v = coalition_values(model.predict, xrow)
        phi = np.zeros(P)
        for m, val in v.items():
            for j in range(P):
                if m[j] == 0:
                    m2 = list(m); m2[j] = 1
                    phi[j] += weights[sum(m)] * (v[tuple(m2)] - val)
        base, full = v[tuple([0] * P)], v[tuple([1] * P)]
        assert abs(phi.sum() - (full - base)) < 1e-8          # efficiency: contributions add up
        for j, p in enumerate(players):
            val = tonight.loc[i, p]
            rows.append({"name": name, "model": label, "feature": p, "value": val,
                         "phi": round(float(phi[j]), 5), "baseline": round(float(base), 5),
                         "prediction": round(float(full), 5)})
shap7 = pd.DataFrame(rows)
shap7.to_csv(OUT + "ward7_shapley.csv", index=False)
# check: for the linear model, Shapley value = coefficient x (value - background mean)
lin_coef = dict(zip(X.columns, lin7.coef_))
a = shap7[(shap7.name == "Aoife Byrne") & (shap7.model == "Linear regression") & (shap7.feature == "sofa")].phi.iloc[0]
print("linear Shapley check (sofa, Aoife):", a, round(lin_coef["sofa"] * (Xn.loc[0, "sofa"] - bg["sofa"].mean()), 5))

# =============================================================== 2. Lung follow-up data
lung = pd.read_csv("data/lung_sim_complete.csv")
spl = pd.read_csv("data/splits/lung_split.csv")
assert (spl["patient_id"].to_numpy() == lung["patient_id"].to_numpy()).all()
trl = (spl["set"] == "train").to_numpy()
lung["male"] = (lung["sex"] == "male").astype(float)
lf = ["age", "male", "ph.ecog", "ph.karno", "wt.loss"]
XL, yL = lung[lf].astype(float), lung["time"].to_numpy(dtype=float)
XLtr, XLte = XL[trl].reset_index(drop=True), XL[~trl].reset_index(drop=True)
yLtr, yLte = yL[trl], yL[~trl]
cv5 = KFold(5, shuffle=True, random_state=SEED)

lmods = {}
lmods["Linear regression"] = LinearRegression().fit(XLtr, yLtr)
lmods["Lasso"] = make_pipeline(StandardScaler(), LassoCV(cv=cv5, random_state=SEED)).fit(XLtr, yLtr)
g = GridSearchCV(DecisionTreeRegressor(random_state=SEED, min_samples_leaf=5),
                 {"max_depth": list(range(1, 9))}, cv=cv5, scoring="neg_mean_absolute_error").fit(XLtr, yLtr)
lmods[f"Tree (depth {g.best_params_['max_depth']}, chosen by CV)"] = g.best_estimator_
lmods["Random forest"] = RandomForestRegressor(n_estimators=500, max_features=0.5, min_samples_leaf=3,
                                               random_state=SEED, n_jobs=1).fit(XLtr, yLtr)
lmods["XGBoost"] = fit_xgb(XLtr, yLtr)
lmods["Neural network (2 hidden layers)"] = TransformedTargetRegressor(
    regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16, 8), alpha=1e-3, max_iter=5000,
                                                           random_state=SEED)),
    transformer=StandardScaler()).fit(XLtr, yLtr)
bench = pd.DataFrame([{"model": k, "test_RMSE": round(rmse(yLte, m.predict(XLte)), 2),
                       "test_MAE": round(mae(yLte, m.predict(XLte)), 2)} for k, m in lmods.items()])
bench = pd.concat([pd.DataFrame([{"model": "Average (no model)", "test_RMSE": round(rmse(yLte, np.full(len(yLte), yLtr.mean())), 2),
                                  "test_MAE": round(mae(yLte, np.full(len(yLte), yLtr.mean())), 2)}]), bench])
bench.to_csv(OUT + "lung_benchmark.csv", index=False)
print(bench)

probe = {"Linear regression": lmods["Linear regression"], "Random forest": lmods["Random forest"],
         "XGBoost": lmods["XGBoost"]}
rng = np.random.default_rng(SEED)
prow = []
for label, m in probe.items():
    base, errs = permutation_importance(m.predict, XLte, yLte, {f: [f] for f in lf}, 30, rng)
    for f, e in errs.items():
        d = np.array(e) - base
        prow.append({"model": label, "feature": f, "importance": round(float(d.mean()), 2),
                     "sd": round(float(d.std(ddof=1)), 2), "base_MAE": round(base, 2)})
pd.DataFrame(prow).to_csv(OUT + "lung_permutation.csv", index=False)

# --- ICE curves (all 171 training patients) ---
grids = {"ph.ecog": [0, 1, 2, 3], "ph.karno": list(range(40, 101, 5)),
         "wt.loss": [round(v, 2) for v in np.linspace(0.8, 25, 20)], "age": list(range(40, 83, 3))}
labels = {"ph.ecog": "ECOG performance status (0 = fully active, 3 = mostly in bed)",
          "ph.karno": "Karnofsky score (100 = normal, 40 = disabled)",
          "wt.loss": "Weight loss in the previous six months (kg)", "age": "Age (years)"}
ice = {"features": {f: {"grid": grids[f], "label": labels[f], "observed": XLtr[f].round(1).tolist()} for f in grids},
       "patients": lung.loc[trl, "patient_id"].tolist(), "ice": {}}
for label, m in probe.items():
    ice["ice"][label] = {}
    for f, gv in grids.items():
        big = pd.concat([XLtr] * len(gv), ignore_index=True)
        big[f] = np.repeat(gv, len(XLtr))
        pr = m.predict(big).reshape(len(gv), len(XLtr)).T          # patients x grid
        ice["ice"][label][f] = np.round(pr, 1).tolist()
json.dump(ice, open(OUT + "lung_ice.json", "w"), separators=(",", ":"))

# --- interaction strength (H) for the forest and XGBoost ---
hrows = []
for label in ["Random forest", "XGBoost"]:
    h = h_statistics(probe[label].predict, XLtr, lf)
    h["model"] = label
    hrows.append(h)
H = pd.concat(hrows)
H["H"] = H["H"].round(3); H["H_raw"] = H["H_raw"].round(2)
H.to_csv(OUT + "lung_h.csv", index=False)
print(H.sort_values("H", ascending=False).head(6))

# --- bootstrap uncertainty of the explanations (forest refitted on 25 bootstrap samples) ---
B = 25
rng = np.random.default_rng(SEED)
sub = XLtr.iloc[rng.choice(len(XLtr), 80, replace=False)].reset_index(drop=True)
brow = []
for b in range(B):
    idx = rng.integers(0, len(XLtr), len(XLtr))
    m = RandomForestRegressor(n_estimators=200, max_features=0.5, min_samples_leaf=3,
                              random_state=SEED + b, n_jobs=1).fit(XLtr.iloc[idx], yLtr[idx])
    base, errs = permutation_importance(m.predict, XLte, yLte, {f: [f] for f in lf}, 5, rng)
    for f, e in errs.items():
        brow.append({"b": b, "var1": f, "var2": f, "value": float(np.mean(e) - base)})
    h = h_statistics(m.predict, sub, lf)
    for _, r in h.iterrows():
        brow.append({"b": b, "var1": r.var1, "var2": r.var2, "value": r.H_raw})
bt = pd.DataFrame(brow).groupby(["var1", "var2"])["value"].agg(["mean", "std"]).reset_index()
bt["cv"] = bt["std"] / bt["mean"].abs()
bt.round(3).to_csv(OUT + "lung_vivi_bootstrap.csv", index=False)

# =============================================================== 3. Small oncology study
med = pd.read_csv("data/med_small_sim.csv")
med["male"] = (med["sex"] == "male").astype(float)
med["stage_num"] = med["stage"].map({"I": 1, "II": 2, "III": 3}).astype(float)
med["new_therapy"] = (med["treatment"] == "new_therapy").astype(float)
mf = ["age", "male", "bmi", "stage_num", "baseline_mm", "crp", "egfr", "new_therapy", "adherence",
      "biomarker"] + [f"gene_{j:03d}" for j in range(1, 21)]
rngm = np.random.default_rng(SEED)
te_idx = rngm.choice(len(med), 200, replace=False)
tem = np.zeros(len(med), bool); tem[te_idx] = True
XM, yM = med[mf], med["tumor_shrinkage_pct"].to_numpy()
XMtr, XMte = XM[~tem].reset_index(drop=True), XM[tem].reset_index(drop=True)
xg_m = fit_xgb(XMtr, yM[~tem])
lin_m = LinearRegression().fit(XMtr, yM[~tem])


def true_mean(d, treat):
    """Mean tumour shrinkage under the simulation's formula (scripts/simulate/make_ch12_data.R)."""
    ec = (d["egfr"] - 90) / 20
    eta = (0.8 * treat + 1.1 * treat * d["biomarker"] + 0.5 * d["adherence"] - 0.25 * (d["age"] - 60) / 10
           - 0.35 * d["male"] - 0.30 * (d["bmi"] - 27) / 5 - 0.60 * (d["stage_num"] - 1)
           - 0.55 * np.log1p(d["crp"]) - 0.22 * np.sqrt(d["baseline_mm"]) + 0.35 * ec - 0.15 * ec ** 2)
    return 100 / (1 + np.exp(-(eta - 0.5)))


def effect(m, d):
    d1, d0 = d.copy(), d.copy()
    d1["new_therapy"] = 1.0; d0["new_therapy"] = 0.0
    return m.predict(d1) - m.predict(d0)


eff = pd.DataFrame({"patient_id": med.loc[tem, "patient_id"].to_numpy(),
                    "biomarker": XMte["biomarker"].round(3),
                    "effect_xgb": np.round(effect(xg_m, XMte), 2),
                    "effect_linear": np.round(effect(lin_m, XMte), 2),
                    "effect_true": np.round(true_mean(XMte, 1) - true_mean(XMte, 0), 2)})
eff.to_csv(OUT + "med_treatment_effect.csv", index=False)
rng = np.random.default_rng(SEED)
mrow = []
for label, m in [("XGBoost", xg_m), ("Linear regression", lin_m)]:
    base, errs = permutation_importance(m.predict, XMte, yM[tem], {f: [f] for f in mf}, 20, rng)
    for f, e in errs.items():
        mrow.append({"model": label, "feature": f, "importance": round(float(np.mean(e) - base), 3)})
pd.DataFrame(mrow).to_csv(OUT + "med_permutation.csv", index=False)
hm = h_statistics(xg_m.predict, XMtr.iloc[:150], ["new_therapy", "biomarker", "stage_num", "crp", "age"])
hm.round(3).to_csv(OUT + "med_h.csv", index=False)
summ = {"test_MAE_xgb": round(mae(yM[tem], xg_m.predict(XMte)), 2),
        "test_MAE_linear": round(mae(yM[tem], lin_m.predict(XMte)), 2)}

json.dump({"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
           "xgboost": xgb.__version__, "seed": SEED, "lung_tree_depth": int(g.best_params_["max_depth"]),
           "ward7_rf_test_MAE": round(mae(yte, rf7.predict(Xte)), 4), "shap_background_n": len(bg),
           **summ}, open(OUT + "versions.json", "w"), indent=1)
print(summ)
print(hm)
print("done")
