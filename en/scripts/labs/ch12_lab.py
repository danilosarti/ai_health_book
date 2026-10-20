# Chapter 12 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 0. Packages and data -----------------------------------------------------
import itertools, math
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
lung = pd.read_csv("data/lung_sim_complete.csv")
lung["male"] = (lung["sex"] == "male").astype(float)
is_tr = pd.read_csv("data/splits/lung_split.csv")["set"].eq("train").to_numpy()
ltr, lte = lung[is_tr].reset_index(drop=True), lung[~is_tr].reset_index(drop=True)
feats = ["age", "male", "ph.ecog", "ph.karno", "wt.loss"]
mae = lambda y, p: float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))

# 1. Two models of days of follow-up (remember: censoring is ignored) ------
lin = LinearRegression().fit(ltr[feats], ltr["time"])
rf = RandomForestRegressor(n_estimators=300, random_state=2026).fit(ltr[feats], ltr["time"])
print({"linear": round(mae(lte["time"], lin.predict(lte[feats])), 1),
       "forest": round(mae(lte["time"], rf.predict(lte[feats])), 1)})

# 2. Permutation importance for any model ----------------------------------
def perm_importance(predict_fun, X, y, features, n_shuffles=10, seed=2026):
    rng = np.random.default_rng(seed)
    base = mae(y, predict_fun(X))
    out = {}
    for f in features:
        errs = []
        for _ in range(n_shuffles):
            Xp = X.copy(); Xp[f] = rng.permutation(Xp[f].to_numpy())
            errs.append(mae(y, predict_fun(Xp)))
        out[f] = np.mean(errs) - base                  # shuffled minus original
    return pd.Series(out).round(1)
print(perm_importance(lin.predict, lte[feats], lte["time"], feats))
print(perm_importance(rf.predict, lte[feats], lte["time"], feats))

# 3. ICE curves and partial dependence ---------------------------------------
def ice_curves(predict_fun, X, feature, grid):
    return np.column_stack([predict_fun(X.assign(**{feature: g})) for g in grid])  # patients x grid
grid = np.arange(1, 26)
ice = ice_curves(rf.predict, ltr[feats], "wt.loss", grid)
c_ice = ice - ice[:, [0]]
print("PDP change from 1 to 25 kg:", round(float(c_ice[:, -1].mean()), 1),
      "| spread across patients (SD):", round(float(c_ice[:, -1].std()), 1))

# 4. Friedman's H for one pair (at 60 sampled training patients) -------------
def pd_at(predict_fun, X, cols, points):
    return np.array([predict_fun(X.assign(**dict(zip(cols, row)))).mean()
                     for row in points[cols].to_numpy()])
def h_stat(predict_fun, X, a, b):
    fa = pd_at(predict_fun, X, [a], X); fa -= fa.mean()
    fb = pd_at(predict_fun, X, [b], X); fb -= fb.mean()
    fab = pd_at(predict_fun, X, [a, b], X); fab -= fab.mean()
    return {"H": round(float(np.sqrt(np.sum((fab - fa - fb) ** 2) / np.sum(fab ** 2))), 3),
            "H_days": round(float(np.sqrt(np.mean((fab - fa - fb) ** 2))), 2)}
sub = ltr[feats].sample(60, random_state=2026).reset_index(drop=True)
print(h_stat(rf.predict, sub, "ph.ecog", "wt.loss"), h_stat(rf.predict, sub, "age", "male"))

# 5. Exact Shapley values for Ward 7 patients: the OLS model of Chapter 4 -----
#    (P10: the ten predictors of Chapters 4 and 6; site counts as one player)
from sklearn.ensemble import RandomForestRegressor
icu = pd.read_csv("data/icu_aki.csv")
is_tr7 = pd.read_csv("data/splits/ward7_split.csv")["set"].eq("train").to_numpy()
train, test = icu[is_tr7].reset_index(drop=True), icu[~is_tr7].reset_index(drop=True)
tonight = pd.read_csv("data/ward7_tonight.csv")
f10 = ["site", "age", "sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence",
       "bun", "crp", "albumin"]
sites = sorted(train["site"].unique())
def design(d):
    X = d[f10].copy(); X["site"] = pd.Categorical(X["site"], categories=sites)
    return pd.get_dummies(X, columns=["site"], drop_first=True, dtype=float)
ols7 = LinearRegression().fit(design(train), train["delta_creatinine_48h"])
print("OLS:", round(mae(test["delta_creatinine_48h"], ols7.predict(design(test))), 4))   # 0.1413, the ledger value (Chapter 4)
def shapley_exact(predict_fun, x, background, features):
    p = len(features); masks = list(itertools.product([0, 1], repeat=p)); v = {}
    for m in masks:
        d = background[features].copy()
        for j, on in enumerate(m):
            if on: d[features[j]] = x[features[j]]
        v[m] = predict_fun(d).mean()
    phi = np.zeros(p)
    for m in masks:
        s = sum(m)
        for j in range(p):
            if m[j] == 0:
                m2 = list(m); m2[j] = 1
                phi[j] += math.factorial(s) * math.factorial(p - s - 1) / math.factorial(p) * (v[tuple(m2)] - v[m])
    return pd.Series([v[masks[0]], *phi, v[masks[-1]]], index=["baseline", *features, "prediction"])
background = train.sample(100, random_state=2026)
# each column: baseline + contributions = prediction
sh = pd.DataFrame({tonight.loc[i, "name"]: shapley_exact(lambda d: ols7.predict(design(d)), tonight.loc[i], background, f10)
                   for i in range(3)})
print(sh.round(3))

# 6. The random forest of Chapter 6 (500 trees, scikit-learn) --------------------
rf7 = RandomForestRegressor(n_estimators=500, max_features=1/3, random_state=2026, n_jobs=1)
rf7.fit(design(train), train["delta_creatinine_48h"])
print("forest:", round(mae(test["delta_creatinine_48h"], rf7.predict(design(test))), 4))   # the same forest as Chapter 6 and the ledger: test MAE 0.1528
#    Its exact Shapley values need about a million predictions per patient, so they
#    so they were computed in advance by scripts/precompute/ch12_explain.py: read them.
shf = pd.read_csv("data/precomputed/ch12/ward7_shapley.csv")
shf = shf[shf["model"] == "Random forest"]
print(shf.pivot(index="feature", columns="name", values="phi").round(3))
# forest predictions 0.29 / -0.09 / 0.83, as in the ledger
print(shf[["name", "baseline", "prediction"]].drop_duplicates())

# 7. vivid and bartMan are R packages with no Python version. The closest equivalent of a
#    VIVID matrix: permutation importance on the diagonal (scikit-learn, test patients) and the
#    unnormalised H of step 4 off the diagonal. For BART in Python there is PyMC-BART, but the
#    VSUP display would have to be drawn by hand: for bartMan, run the R code of the Lab.
from sklearn.inspection import permutation_importance
pi = permutation_importance(rf, lte[feats], lte["time"], scoring="neg_mean_absolute_error",
                            n_repeats=10, random_state=2026)
vivid_py = pd.DataFrame(0.0, index=feats, columns=feats)
for j, f in enumerate(feats):
    vivid_py.loc[f, f] = pi.importances_mean[j]          # increase in test MAE (days)
for a, b in itertools.combinations(feats, 2):              # H in days, at the 60 patients of step 4
    vivid_py.loc[a, b] = vivid_py.loc[b, a] = h_stat(rf.predict, sub, a, b)["H_days"]
print(vivid_py.round(1))
