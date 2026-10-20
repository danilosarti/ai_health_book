"""Chapter 6 - precomputed results for trees and ensembles (forests and XGBoost).

Everything heavy in Chapter 6 is computed here, once, and saved as small CSV files in
data/precomputed/ch06/. The chapter reads these files; nothing here runs in the browser.

Splits: the book's shared splits from Chapter 3, data/splits/ward7_split.csv and
data/splits/onco1_split.csv (patient_id, set = train/test, fold 1-5 for training rows).
Ward 7: 1 000 training / 250 test patients; ONCO-1: 800 training / 200 test patients.
Cross-validation inside the training set uses the shared folds.

Usage (from the book folder):  python3 scripts/precompute/ch06_ensembles.py
Requires numpy, pandas, scikit-learn, xgboost (single-threaded, about one minute).
"""
import json
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 2026
OUT = "data/precomputed/ch06/"
mae = lambda y, p: float(np.mean(np.abs(y - p)))
rmse = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))


def design(df, cols):
    """Numeric design matrix with dummy columns for categorical variables."""
    return pd.get_dummies(df[cols], drop_first=True, dtype=float)


def fit_xgb(Xtr, ytr, objective, eval_metric, eta=0.05, rounds=2000):
    """XGBoost with early stopping on a validation set carved out of the TRAINING data only."""
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(ytr))
    nval = len(ytr) // 5
    va, fi = idx[:nval], idx[nval:]
    m = xgb.XGBRegressor if objective.startswith("reg") else xgb.XGBClassifier
    model = m(n_estimators=rounds, learning_rate=eta, max_depth=3, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=1.0, objective=objective, eval_metric=eval_metric,
              early_stopping_rounds=50, random_state=SEED, n_jobs=1)
    model.fit(Xtr.iloc[fi], ytr[fi], eval_set=[(Xtr.iloc[va], ytr[va])], verbose=False)
    return model


# ------------------------------------------------------------------ Ward 7 (regression)
icu = pd.read_csv("data/icu_aki.csv")
sp7 = pd.read_csv("data/splits/ward7_split.csv")
assert (sp7["patient_id"].to_numpy() == icu["patient_id"].to_numpy()).all()
tr7 = (sp7["set"] == "train").to_numpy()
folds7 = PredefinedSplit(sp7.loc[tr7, "fold"].astype(int).to_numpy() - 1)
feat = ["site", "age", "sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence",
        "bun", "crp", "albumin"]
X = design(icu, feat)
y = icu["delta_creatinine_48h"].to_numpy()
Xtr, Xte, ytr, yte = X[tr7], X[~tr7], y[tr7], y[~tr7]

rows = []
def add(data, model, ptr, pte):
    rows.append({"data": data, "model": model, "train_MAE": mae(ytr, ptr), "test_MAE": mae(yte, pte),
                 "train_RMSE": rmse(ytr, ptr), "test_RMSE": rmse(yte, pte)})

add("ward7", "Average (no model)", np.full(len(ytr), ytr.mean()), np.full(len(yte), ytr.mean()))
lin = LinearRegression().fit(Xtr, ytr)
add("ward7", "Linear regression", lin.predict(Xtr), lin.predict(Xte))
deep = DecisionTreeRegressor(min_samples_leaf=1, random_state=SEED).fit(Xtr, ytr)
add("ward7", "Fully grown tree", deep.predict(Xtr), deep.predict(Xte))
grid = GridSearchCV(DecisionTreeRegressor(random_state=SEED), {"max_depth": list(range(1, 11))},
                    cv=folds7, scoring="neg_mean_absolute_error").fit(Xtr, ytr)
tree_depth = int(grid.best_params_["max_depth"])
add("ward7", f"Tree, depth {tree_depth} (chosen by 5-fold CV)", grid.predict(Xtr), grid.predict(Xte))
ytr7 = ytr.copy()
rf = RandomForestRegressor(n_estimators=500, max_features=1/3, oob_score=True, random_state=SEED,
                           n_jobs=1).fit(Xtr, ytr)
add("ward7", "Random forest (500 trees)", rf.predict(Xtr), rf.predict(Xte))
bxgb = fit_xgb(Xtr, ytr, "reg:squarederror", "mae")
add("ward7", f"XGBoost ({bxgb.best_iteration + 1} rounds, early stopping)", bxgb.predict(Xtr), bxgb.predict(Xte))

# OOB error as trees are added (one forest, grown incrementally with warm_start)
oob = []
rfw = RandomForestRegressor(n_estimators=20, max_features=1/3, oob_score=True, warm_start=True,
                            random_state=SEED, n_jobs=1)
bag = RandomForestRegressor(n_estimators=20, max_features=None, oob_score=True, warm_start=True,
                            random_state=SEED, n_jobs=1)
single_tree_test = mae(yte, DecisionTreeRegressor(random_state=SEED).fit(Xtr, ytr).predict(Xte))
for B in list(range(20, 100, 10)) + list(range(100, 501, 25)):
    rfw.set_params(n_estimators=B).fit(Xtr, ytr)
    bag.set_params(n_estimators=B).fit(Xtr, ytr)
    oob.append({"trees": B,
                "rf_oob_MAE": mae(ytr, rfw.oob_prediction_), "rf_test_MAE": mae(yte, rfw.predict(Xte)),
                "bag_oob_MAE": mae(ytr, bag.oob_prediction_), "bag_test_MAE": mae(yte, bag.predict(Xte))})
pd.DataFrame(oob).round(4).to_csv(OUT + "ward7_oob_curve.csv", index=False)

# Boosting curves: training and test MAE after each round, small vs large learning rate, no early stopping
curves = []
for eta in [0.05, 0.3]:
    m = xgb.XGBRegressor(n_estimators=600, learning_rate=eta, max_depth=3, objective="reg:squarederror",
                         eval_metric="mae", random_state=SEED, n_jobs=1)
    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xte, yte)], verbose=False)
    ev = m.evals_result()
    for r in range(600):
        curves.append({"eta": eta, "round": r + 1, "train_MAE": ev["validation_0"]["mae"][r],
                       "test_MAE": ev["validation_1"]["mae"][r]})
pd.DataFrame(curves).round(4).to_csv(OUT + "ward7_boosting_curve.csv", index=False)

imp = pd.DataFrame({"feature": X.columns, "rf_impurity": rf.feature_importances_,
                    "xgb_gain_share": pd.Series(bxgb.get_booster().get_score(importance_type="gain"))
                    .reindex(X.columns).fillna(0).to_numpy()})
imp["xgb_gain_share"] /= imp["xgb_gain_share"].sum()
imp.sort_values("rf_impurity", ascending=False).round(4).to_csv(OUT + "ward7_importance.csv", index=False)

# ------------------------------------------------------------------ ONCO-1
onco = pd.read_csv("data/onco1.csv")
spO = pd.read_csv("data/splits/onco1_split.csv")
assert (spO["patient_id"].to_numpy() == onco["patient_id"].to_numpy()).all()
is_tr = (spO["set"] == "train").to_numpy()
genes = [c for c in onco.columns if c.startswith("gene_")]
clin = ["treatment", "dose_intensity", "patient_age", "tumor_grade", "performance_score"]
leaky = ["gene_08", "gene_14", "gene_19"]
variants = {"all features": clin + genes,
            "without genes 08, 14, 19": clin + [g for g in genes if g not in leaky]}

# Baseline: the treatment arm alone (chemo = 1), no model at all
te = onco[~is_tr]
rows.append({"data": "onco1_class", "variant": "treatment arm only", "model": "Treatment arm alone",
             "test_AUC": roc_auc_score(te["high_response"], (te["treatment"] == "chemo").astype(int))})

def class_models():
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "Tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=SEED),
        "Random forest (500 trees)": RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=1),
    }

# Two populations: everybody, and only the patients who received chemotherapy
for pop in ["all patients", "chemotherapy arm"]:
    keep = np.ones(len(onco), bool) if pop == "all patients" else (onco["treatment"] == "chemo").to_numpy()
    for vname, cols in variants.items():
        if pop == "chemotherapy arm":
            cols = [c for c in cols if c != "treatment"]
        X = design(onco, cols)
        Xtr, Xte = X[keep & is_tr], X[keep & ~is_tr]
        tag = "onco1_class" if pop == "all patients" else "onco1_class_chemo"
        yc = onco["high_response"].to_numpy()
        ytr_c, yte_c = yc[keep & is_tr], yc[keep & ~is_tr]
        models = class_models()
        for name, m in models.items():
            m.fit(Xtr, ytr_c)
            rows.append({"data": tag, "variant": vname, "model": name, "n_train": len(ytr_c), "n_test": len(yte_c),
                         "train_AUC": roc_auc_score(ytr_c, m.predict_proba(Xtr)[:, 1]),
                         "test_AUC": roc_auc_score(yte_c, m.predict_proba(Xte)[:, 1])})
        m = fit_xgb(Xtr, ytr_c, "binary:logistic", "logloss")
        rows.append({"data": tag, "variant": vname, "model": "XGBoost", "n_train": len(ytr_c), "n_test": len(yte_c),
                     "train_AUC": roc_auc_score(ytr_c, m.predict_proba(Xtr)[:, 1]),
                     "test_AUC": roc_auc_score(yte_c, m.predict_proba(Xte)[:, 1])})
        if pop == "all patients" and vname == "all features":
            rfc = models["Random forest (500 trees)"]
            top = pd.DataFrame({"feature": X.columns, "importance": rfc.feature_importances_})
            top.sort_values("importance", ascending=False).head(10).round(4).to_csv(
                OUT + "onco1_rf_importance_all.csv", index=False)
        if pop == "all patients":
            yr = onco["response_percent"].to_numpy()
            ytr, yte = yr[is_tr], yr[~is_tr]
            for name, m in {"Linear regression": LinearRegression(),
                            "Random forest (500 trees)": RandomForestRegressor(
                                n_estimators=500, max_features=1/3, random_state=SEED, n_jobs=1)}.items():
                m.fit(Xtr, ytr)
                rows.append({"data": "onco1_reg", "variant": vname, "model": name,
                             "train_MAE": mae(ytr, m.predict(Xtr)), "test_MAE": mae(yte, m.predict(Xte))})
            m = fit_xgb(Xtr, ytr, "reg:squarederror", "mae")
            rows.append({"data": "onco1_reg", "variant": vname, "model": "XGBoost",
                         "train_MAE": mae(ytr, m.predict(Xtr)), "test_MAE": mae(yte, m.predict(Xte))})

pd.DataFrame(rows).round(4).to_csv(OUT + "benchmark.csv", index=False)

# correlation of every gene with the outcome (training rows only)
tr = onco[is_tr]
cor = pd.DataFrame({"gene": genes, "r_response": [tr[g].corr(tr["response_percent"]) for g in genes]})
cor.round(3).to_csv(OUT + "onco1_gene_correlations.csv", index=False)

json.dump({"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
           "xgboost": xgb.__version__, "seed": SEED, "ward7_tree_depth": tree_depth,
           "ward7_rf_oob_MAE": mae(ytr7, rf.oob_prediction_)},
          open(OUT + "versions.json", "w"), indent=1)
print(pd.DataFrame(rows).round(3).to_string())
