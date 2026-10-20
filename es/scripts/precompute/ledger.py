"""Model ledger, Python part: the sklearn / XGBoost / PyTorch / pymc-bart models.

The ledger is the book's single source of truth for every model result of the running examples
(Ward 7, ONCO-1, HomeBeat). It is driven by scripts/precompute/ledger.R, which calls this script
first and then computes EVERY metric in one place from the predictions written here.

This script fits models only and writes predictions (no metrics) to
    data/precomputed/ledger/parts/py_preds.csv   example, model_id, kind, id, pred
        kind = test     prediction for a test row (id = patient / participant id)
               oof      out-of-fold prediction for a training row (5 shared folds; the model's
                        tuning value is held fixed at the one chosen on the full training set)
               tonight  Aoife / Tomas / Priya (Ward 7)
               patient  ONCO-1 patients #0042 / #0917 (see ledger.R for the mapping)
               lena     Lena (HomeBeat)
    data/precomputed/ledger/parts/py_meta.csv    example, model_id, tuning, software

Implementations are those of the chapters, so values that should not move do not move:
  * random forests, bagging, sklearn trees, the ONCO-1 "logistic regression (C = 1)": exactly
    scripts/precompute/ch06_ensembles.py (same design matrices, settings and seed). The Ward 7
    forest is also the forest of Chapter 12 (ch12_explain.py).
  * XGBoost: Chapter 6 settings (eta 0.05, depth 3, subsample 0.8, colsample 0.8, lambda 1), BUT the
    number of rounds is now chosen by 5-fold CV on the shared folds (mean validation curve), then
    the model is refitted on all training rows with that number of rounds. (Chapter 6 used early
    stopping on a random 20 % of the training set.)
  * BART: exactly scripts/precompute/ch08_bart.py (pymc-bart, 50 trees, 2 chains x 1 000 draws +
    1 000 tuning, seed 2026, same call order), so Chapter 8's numbers are reproduced. No CV for BART
    (five more MCMC runs are not worth it): its cv column is empty.
  * ONCO-1 multi-output network: the architecture and training loop of
    scripts/precompute/ch07_pneumonia_cnn.py (32-16, dropout 0.2, Adam 1e-3, weight decay 1e-4),
    refitted on the shared split (800 / 200); the number of epochs is chosen by 5-fold CV on the
    shared folds instead of an inner random 15 % split. The CNN is not touched.
  * MLPs for Ward 7 (Chapter 7, new): sklearn MLP 32-16, ReLU, Adam, L2 alpha 1e-3, batch 32,
    standardised on training rows, seed 2026; number of epochs chosen by 5-fold CV.

Usage (from the book folder), with the pymc-bart environment documented below:
    <venv>/bin/python scripts/precompute/ledger.py            (about 2.5 minutes)
    <venv>/bin/python scripts/precompute/ledger.py --no-bart  (skips BART; keeps old BART rows)
pymc-bart environment (pymc 6.3.2, pymc-bart 0.13.1 as in data/precomputed/ch08/bart_summary.json):
    python3 -m venv --system-site-packages ~/venv-bart
    ~/venv-bart/bin/pip install pymc==6.3.2 pymc-bart==0.13.1
(--system-site-packages keeps the system numpy / pandas / scikit-learn 1.9.1 / xgboost 3.4.1 / torch.)
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = 2026
PARTS = "data/precomputed/ledger/parts"
os.makedirs(PARTS, exist_ok=True)
DO_BART = "--no-bart" not in sys.argv
NJ = 2                      # forests: results do not depend on n_jobs in scikit-learn

preds, meta = [], []


def put(example, model_id, kind, ids, p):
    p = np.asarray(p, float).ravel()
    ids = list(ids)
    assert len(ids) == len(p), (example, model_id, kind)
    preds.append(pd.DataFrame({"example": example, "model_id": model_id, "kind": kind,
                               "id": ids, "pred": p}))


def note(example, model_id, tuning, software):
    meta.append({"example": example, "model_id": model_id, "tuning": tuning, "software": software})


def design(df, cols):
    """Numeric design matrix with dummy columns for categorical variables (as in Chapter 6)."""
    return pd.get_dummies(df[cols], drop_first=True, dtype=float)


def oof_fit(make, X, y, fold, proba=False):
    """Out-of-fold predictions on the training rows with a fixed model specification."""
    out = np.empty(len(y))
    for v in range(1, 6):
        a, b = fold != v, fold == v
        m = make().fit(X[a], y[a])
        out[b] = m.predict_proba(X[b])[:, 1] if proba else m.predict(X[b])
    return out


# ------------------------------------------------------------------ XGBoost, rounds by CV
def xgb_cv(X, y, fold, task, max_rounds=1000, eta=0.05):
    """Choose the number of rounds by 5-fold CV on the shared folds (mean validation curve),
    then refit on all training rows. Returns (final model, best rounds, out-of-fold predictions)."""
    cls = xgb.XGBRegressor if task == "reg" else xgb.XGBClassifier
    obj, met = ("reg:squarederror", "mae") if task == "reg" else ("binary:logistic", "logloss")
    par = dict(learning_rate=eta, max_depth=3, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
               objective=obj, eval_metric=met, random_state=SEED, n_jobs=1)
    curves, fits = [], []
    for v in range(1, 6):
        a, b = fold != v, fold == v
        m = cls(n_estimators=max_rounds, **par).fit(X[a], y[a], eval_set=[(X[b], y[b])], verbose=False)
        curves.append(m.evals_result()["validation_0"][met]); fits.append((m, b))
    curve = np.mean(np.array(curves), axis=0)
    best = int(np.argmin(curve)) + 1
    oof = np.empty(len(y))
    for m, b in fits:
        oof[b] = (m.predict(X[b], iteration_range=(0, best)) if task == "reg"
                  else m.predict_proba(X[b], iteration_range=(0, best))[:, 1])
    final = cls(n_estimators=best, **par).fit(X, y, verbose=False)
    return final, best, oof


# ------------------------------------------------------------------ sklearn MLP, epochs by CV
def mlp_cv(X, y, fold, task, max_epochs=200):
    """sklearn MLP (32-16) trained epoch by epoch with partial_fit; the number of epochs is the one
    with the lowest mean validation loss over the 5 shared folds (MAE for regression, log-loss for
    classification). Scaling is fitted on the rows used for training."""
    def new():
        kw = dict(hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3, batch_size=32,
                  random_state=SEED)
        return MLPRegressor(**kw) if task == "reg" else MLPClassifier(**kw)

    def step(m, Z, t):
        if task == "reg":
            m.partial_fit(Z, t)
        else:
            m.partial_fit(Z, t, classes=np.array([0, 1]))

    def pred(m, Z):
        return m.predict(Z) if task == "reg" else m.predict_proba(Z)[:, 1]

    X = np.asarray(X, float)
    losses, vpreds = np.zeros((5, max_epochs)), {}
    for v in range(1, 6):
        a, b = fold != v, fold == v
        sc = StandardScaler().fit(X[a]); Za, Zb = sc.transform(X[a]), sc.transform(X[b])
        m = new(); P = np.zeros((max_epochs, b.sum()))
        for e in range(max_epochs):
            step(m, Za, y[a]); p = pred(m, Zb); P[e] = p
            if task == "reg":
                losses[v - 1, e] = np.mean(np.abs(y[b] - p))
            else:
                q = np.clip(p, 1e-12, 1 - 1e-12)
                losses[v - 1, e] = -np.mean(y[b] * np.log(q) + (1 - y[b]) * np.log(1 - q))
        vpreds[v] = (b, P)
    best = int(np.argmin(losses.mean(axis=0))) + 1
    oof = np.empty(len(y))
    for v, (b, P) in vpreds.items():
        oof[b] = P[best - 1]
    sc = StandardScaler().fit(X); m = new()
    for e in range(best):
        step(m, sc.transform(X), y)
    return (lambda Xn: pred(m, sc.transform(np.asarray(Xn, float)))), best, oof


# ------------------------------------------------------------------ BART (as in Chapter 8)
def fit_bart(Xtr, ytr, Xnew_list, m=50, seed=SEED):
    """Copied from scripts/precompute/ch08_bart.py (same calls, same order, same seed)."""
    import pymc as pm, pymc_bart as pmb
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
            f = pp.posterior_predictive["mu"].values.reshape(-1, Xn.shape[0])
            yn = pp.posterior_predictive["y"].values.reshape(-1, Xn.shape[0])
            res.append((f, yn))
    return res


# =================================================================== WARD 7
icu = pd.read_csv("data/icu_aki.csv")
tonight = pd.read_csv("data/ward7_tonight.csv")
sp7 = pd.read_csv("data/splits/ward7_split.csv")
assert (sp7["patient_id"].to_numpy() == icu["patient_id"].to_numpy()).all()
tr7 = (sp7["set"] == "train").to_numpy()
fold7 = sp7.loc[tr7, "fold"].astype(int).to_numpy()
P10 = ["site", "age", "sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence", "bun", "crp", "albumin"]
P5 = ["sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence"]
X10 = design(icu, P10)                                          # identical to Chapter 6
Xt10 = pd.get_dummies(pd.concat([icu[P10], tonight[P10]], ignore_index=True), drop_first=True,
                      dtype=float).iloc[len(icu):].reset_index(drop=True)[X10.columns]
y7 = icu["delta_creatinine_48h"].to_numpy()
a7 = (y7 >= 0.3).astype(int)
Xtr, Xte, ytr = X10[tr7], X10[~tr7], y7[tr7]
id_tr, id_te = icu["patient_id"][tr7].tolist(), icu["patient_id"][~tr7].tolist()
names = tonight["name"].tolist()


def ward7_out(ex, mid, fitted_predict, oof):
    put(ex, mid, "test", id_te, fitted_predict(Xte if ex == "ward7_reg" else X5te))
    put(ex, mid, "tonight", names, fitted_predict(Xt10 if ex == "ward7_reg" else Xt5))
    if oof is not None:
        put(ex, mid, "oof", id_tr, oof)


print("Ward 7 regression ...", flush=True)
# Chapter 6 table variants (scikit-learn trees)
mk = lambda: DecisionTreeRegressor(min_samples_leaf=1, random_state=SEED)
m = mk().fit(Xtr, ytr); ward7_out("ward7_reg", "tree_sklearn_full", m.predict, oof_fit(mk, Xtr, ytr, fold7))
note("ward7_reg", "tree_sklearn_full", "no pruning (min_samples_leaf = 1)", f"scikit-learn {sklearn.__version__}")
grid = GridSearchCV(DecisionTreeRegressor(random_state=SEED), {"max_depth": list(range(1, 11))},
                    cv=PredefinedSplit(fold7 - 1), scoring="neg_mean_absolute_error").fit(Xtr, ytr)
dep = int(grid.best_params_["max_depth"])
mk = lambda: DecisionTreeRegressor(max_depth=dep, random_state=SEED)
ward7_out("ward7_reg", "tree_sklearn_depthcv", grid.predict, oof_fit(mk, Xtr, ytr, fold7))
note("ward7_reg", "tree_sklearn_depthcv", f"max_depth = {dep} (5-fold CV on shared folds, MAE)", f"scikit-learn {sklearn.__version__}")
# bagging and random forest (Chapter 6; the forest is also Chapter 12's)
mk = lambda: RandomForestRegressor(n_estimators=500, max_features=None, random_state=SEED, n_jobs=NJ)
m = mk().fit(Xtr, ytr); ward7_out("ward7_reg", "bagging500", m.predict, oof_fit(mk, Xtr, ytr, fold7))
note("ward7_reg", "bagging500", "500 bootstrap trees, all 12 columns at each split", f"scikit-learn {sklearn.__version__}")
mk = lambda: RandomForestRegressor(n_estimators=500, max_features=1/3, random_state=SEED, n_jobs=NJ)
m = mk().fit(Xtr, ytr); ward7_out("ward7_reg", "rf500", m.predict, oof_fit(mk, Xtr, ytr, fold7))
note("ward7_reg", "rf500", "500 trees, max_features = 1/3, seed 2026", f"scikit-learn {sklearn.__version__}")
m, best, oof = xgb_cv(Xtr, ytr, fold7, "reg")
ward7_out("ward7_reg", "xgb", m.predict, oof)
note("ward7_reg", "xgb", f"{best} rounds (5-fold CV, MAE); eta 0.05, depth 3, subsample 0.8, colsample 0.8", f"xgboost {xgb.__version__}")
f, best, oof = mlp_cv(Xtr, ytr, fold7, "reg")
ward7_out("ward7_reg", "mlp", f, oof)
note("ward7_reg", "mlp", f"32-16 ReLU, alpha 1e-3, batch 32, {best} epochs (5-fold CV, MAE)", f"scikit-learn {sklearn.__version__}")

print("Ward 7 AKI ...", flush=True)
X5 = icu[P5].astype(float)
X5tr, X5te, Xt5 = X5[tr7], X5[~tr7], tonight[P5].astype(float)
atr = a7[tr7]
mk = lambda: RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=NJ)
m = mk().fit(X5tr, atr)
ward7_out("ward7_aki", "rf500", lambda Z: m.predict_proba(Z)[:, 1], oof_fit(mk, X5tr, atr, fold7, proba=True))
note("ward7_aki", "rf500", "500 trees, max_features = sqrt (default), seed 2026", f"scikit-learn {sklearn.__version__}")
m, best, oof = xgb_cv(X5tr, atr, fold7, "clf")
ward7_out("ward7_aki", "xgb", lambda Z: m.predict_proba(Z)[:, 1], oof)
note("ward7_aki", "xgb", f"{best} rounds (5-fold CV, log-loss); eta 0.05, depth 3, subsample 0.8, colsample 0.8", f"xgboost {xgb.__version__}")
f, best, oof = mlp_cv(X5tr, atr, fold7, "clf")
ward7_out("ward7_aki", "mlp", f, oof)
note("ward7_aki", "mlp", f"32-16 ReLU, alpha 1e-3, batch 32, {best} epochs (5-fold CV, log-loss)", f"scikit-learn {sklearn.__version__}")

# =================================================================== ONCO-1
print("ONCO-1 ...", flush=True)
onco = pd.read_csv("data/onco1.csv")
spO = pd.read_csv("data/splits/onco1_split.csv")
assert (spO["patient_id"].to_numpy() == onco["patient_id"].to_numpy()).all()
isO = (spO["set"] == "train").to_numpy()
foldO = spO["fold"].to_numpy()
genes = [c for c in onco.columns if c.startswith("gene_")]
leaky = ["gene_08", "gene_14", "gene_19"]
g47 = [g for g in genes if g not in leaky]
clin = ["treatment", "dose_intensity", "patient_age", "tumor_grade", "performance_score"]
PAT = {"#0042": "P5440", "#0917": "P9541"}           # mapping documented in ledger.R / README
pat_rows = np.array([onco.index[onco["patient_id"] == PAT[k]][0] for k in PAT])
chemo = (onco["treatment"] == "chemo").to_numpy()
yr = onco["response_percent"].to_numpy(float)
yb = onco["high_response"].to_numpy()
oid = onco["patient_id"].to_numpy()


def onco_out(ex, mid, predict, Xall, rows_tr, rows_te, oof):
    put(ex, mid, "test", oid[rows_te], predict(Xall.iloc[rows_te]))
    put(ex, mid, "patient", list(PAT), predict(Xall.iloc[pat_rows]))
    if oof is not None:
        put(ex, mid, "oof", oid[rows_tr], oof)


# regression, all patients, canonical predictors (Chapter 6 design)
XO = design(onco, clin + g47)
rtr, rte = np.where(isO)[0], np.where(~isO)[0]
fO = foldO[isO].astype(int)
mk = lambda: RandomForestRegressor(n_estimators=500, max_features=1/3, random_state=SEED, n_jobs=NJ)
m = mk().fit(XO.iloc[rtr], yr[rtr])
onco_out("onco1_reg", "rf500", m.predict, XO, rtr, rte, oof_fit(mk, XO.iloc[rtr], yr[rtr], fO))
note("onco1_reg", "rf500", "500 trees, max_features = 1/3, seed 2026", f"scikit-learn {sklearn.__version__}")
m, best, oof = xgb_cv(XO.iloc[rtr], yr[rtr], fO, "reg")
onco_out("onco1_reg", "xgb", m.predict, XO, rtr, rte, oof)
note("onco1_reg", "xgb", f"{best} rounds (5-fold CV, MAE); eta 0.05, depth 3, subsample 0.8, colsample 0.8", f"xgboost {xgb.__version__}")

# classification, chemotherapy arm (Chapter 6 design without treatment)
XC = design(onco, [c for c in clin if c != "treatment"] + g47)
ctr, cte = np.where(chemo & isO)[0], np.where(chemo & ~isO)[0]
fC = foldO[ctr].astype(int)
for mid, mk, tun in [
    ("ridge_logistic_C1_sklearn", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
     "L2 penalty, C = 1 (scikit-learn default), standardised on training rows; the Chapter 6 'logistic regression'"),
    ("tree_depth4", lambda: DecisionTreeClassifier(max_depth=4, random_state=SEED), "max_depth = 4 (fixed, as in Chapter 6)"),
    ("rf500", lambda: RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=NJ),
     "500 trees, max_features = sqrt (default), seed 2026")]:
    m = mk().fit(XC.iloc[ctr], yb[ctr])
    onco_out("onco1_class", mid, lambda Z, m=m: m.predict_proba(Z)[:, 1], XC, ctr, cte,
             oof_fit(mk, XC.iloc[ctr], yb[ctr], fC, proba=True))
    note("onco1_class", mid, tun, f"scikit-learn {sklearn.__version__}")
m, best, oof = xgb_cv(XC.iloc[ctr], yb[ctr], fC, "clf")
onco_out("onco1_class", "xgb", lambda Z: m.predict_proba(Z)[:, 1], XC, ctr, cte, oof)
note("onco1_class", "xgb", f"{best} rounds (5-fold CV, log-loss); eta 0.05, depth 3, subsample 0.8, colsample 0.8", f"xgboost {xgb.__version__}")

# multi-output network (Chapter 7), refitted on the shared split
print("ONCO-1 multi-output network ...", flush=True)
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
base = pd.concat([onco[["dose_intensity", "patient_age", "performance_score"]],
                  (onco["treatment"] == "chemo").astype(int).rename("chemo"),
                  pd.get_dummies(onco["tumor_grade"], prefix="grade", drop_first=True).astype(int)], axis=1)
XM = pd.concat([base, onco[g47]], axis=1).to_numpy(np.float32)       # 53 columns, as in Chapter 7
yc32, yb32 = yr.astype(np.float32), yb.astype(np.float32)


def mo_train(fit_i, epochs, val_i=None):
    """Chapter 7 training loop for a fixed number of epochs; returns predict(rows) and, if val_i is
    given, the validation loss and predictions after every epoch."""
    mu, sd = XM[fit_i].mean(0), XM[fit_i].std(0) + 1e-8              # scale on the training rows
    Xs = (XM - mu) / sd
    torch.manual_seed(SEED); np.random.seed(SEED)
    T = lambda a: torch.tensor(a)
    shared = nn.Sequential(nn.Linear(Xs.shape[1], 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 16), nn.ReLU())
    head_c, head_b = nn.Linear(16, 1), nn.Linear(16, 1)
    params = list(shared.parameters()) + list(head_c.parameters()) + list(head_b.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    sc = yc32[fit_i].std()

    def loss_fn(i):
        h = shared(T(Xs[i]))
        return (F.mse_loss(head_c(h).squeeze(1), T(yc32[i])) / sc ** 2 +
                F.binary_cross_entropy_with_logits(head_b(h).squeeze(1), T(yb32[i])))

    def predict(rows):
        shared.eval()
        with torch.no_grad():
            h = shared(T(Xs[rows]))
            return head_c(h).squeeze(1).numpy().astype(float), torch.sigmoid(head_b(h).squeeze(1)).numpy().astype(float)
    hist = []
    for ep in range(1, epochs + 1):
        shared.train()
        for j in np.array_split(np.random.default_rng(ep).permutation(fit_i), 9):
            opt.zero_grad(); l = loss_fn(j); l.backward(); opt.step()
        if val_i is not None:
            shared.eval()
            with torch.no_grad():
                vl = loss_fn(val_i).item()
            hist.append((vl,) + predict(val_i))
    return predict, hist


MAXEP = 300
H = {}
for v in range(1, 6):
    fit_i, val_i = rtr[fO != v], rtr[fO == v]
    _, hist = mo_train(fit_i, MAXEP, val_i)
    H[v] = (val_i, hist)
curve = np.mean([[h[0] for h in H[v][1]] for v in H], axis=0)
best_ep = int(np.argmin(curve)) + 1
oof_c, oof_b = pd.Series(np.nan, index=range(len(onco))), pd.Series(np.nan, index=range(len(onco)))
for v, (val_i, hist) in H.items():
    oof_c[val_i] = hist[best_ep - 1][1]; oof_b[val_i] = hist[best_ep - 1][2]
predict, _ = mo_train(rtr, best_ep)
pc_all, pb_all = predict(np.arange(len(onco)))
put("onco1_reg", "mlp_multi", "test", oid[rte], pc_all[rte])
put("onco1_reg", "mlp_multi", "patient", list(PAT), pc_all[pat_rows])
put("onco1_reg", "mlp_multi", "oof", oid[rtr], oof_c[rtr])
put("onco1_class", "mlp_multi", "test", oid[cte], pb_all[cte])
put("onco1_class", "mlp_multi", "patient", list(PAT), pb_all[pat_rows])
put("onco1_class", "mlp_multi", "oof", oid[ctr], oof_b[ctr])
tun = (f"one network, two heads (response %, high response); 53 inputs, 32-16, dropout 0.2, Adam 1e-3, "
       f"weight decay 1e-4; {best_ep} epochs (5-fold CV, combined loss); trained on all 800 training patients")
note("onco1_reg", "mlp_multi", tun, f"PyTorch {torch.__version__}")
note("onco1_class", "mlp_multi", tun + "; evaluated on the chemotherapy-arm patients", f"PyTorch {torch.__version__}")

# =================================================================== HomeBeat
print("HomeBeat ...", flush=True)
hb = pd.read_csv("data/homebeat.csv")
sph = pd.read_csv("data/splits/homebeat_split.csv")
assert (sph["participant_id"].to_numpy() == hb["participant_id"].to_numpy()).all()
trh = (sph["set"] == "train").to_numpy()
fH = sph.loc[trh, "fold"].astype(int).to_numpy()
feat_hb = ["age", "frailty", "baseline_inflammation"]
Xh = hb[feat_hb].to_numpy(float); yh = hb["resting_hr"].to_numpy(float)
lena = np.array([[72, 0.55, 4.1]])
hid = hb["participant_id"].to_numpy()


def hb_out(mid, predict, oof):
    put("homebeat", mid, "test", hid[~trh], predict(Xh[~trh]))
    put("homebeat", mid, "lena", ["Lena"], predict(lena))
    if oof is not None:
        put("homebeat", mid, "oof", hid[trh], oof)


mk = lambda: RandomForestRegressor(n_estimators=500, max_features=1/3, random_state=SEED, n_jobs=NJ)
m = mk().fit(Xh[trh], yh[trh]); hb_out("rf500", m.predict, oof_fit(mk, Xh[trh], yh[trh], fH))
note("homebeat", "rf500", "500 trees, max_features = 1/3, seed 2026 (as in Chapter 8)", f"scikit-learn {sklearn.__version__}")
mk = lambda: xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED, n_jobs=1)
m = mk().fit(Xh[trh], yh[trh]); hb_out("xgb300", m.predict, oof_fit(mk, Xh[trh], yh[trh], fH))
note("homebeat", "xgb300", "300 rounds fixed, eta 0.05, depth 3 (the Chapter 8 setting)", f"xgboost {xgb.__version__}")
m, best, oof = xgb_cv(Xh[trh], yh[trh], fH, "reg")
hb_out("xgb", m.predict, oof)
note("homebeat", "xgb", f"{best} rounds (5-fold CV, MAE); eta 0.05, depth 3, subsample 0.8, colsample 0.8", f"xgboost {xgb.__version__}")

# =================================================================== BART (Chapter 8)
bart_info = {}
if DO_BART:
    import pymc, pymc_bart
    print("BART HomeBeat ...", flush=True)
    # The same split as Chapter 8, rebuilt with the same generator so that the call is identical
    idx = np.random.default_rng(SEED).permutation(len(hb)); trb = np.zeros(len(hb), bool); trb[idx[:400]] = True
    assert (trb == trh).all()
    (res_te, res_tr) = fit_bart(Xh[trh], yh[trh], [np.vstack([Xh[~trh], lena]), Xh[trh]])
    f_te = res_te[0]
    put("homebeat", "bart", "test", hid[~trh], f_te[:, :-1].mean(axis=0))
    put("homebeat", "bart", "lena", ["Lena"], [f_te[:, -1].mean()])
    note("homebeat", "bart", "50 trees; 2 chains x 1 000 draws + 1 000 tuning; posterior mean of f(x)",
         f"pymc {pymc.__version__}, pymc-bart {pymc_bart.__version__}")
    print("BART Ward 7 ...", flush=True)
    sp7b = sp7.set_index("patient_id").loc[icu["patient_id"]].reset_index()
    tr7b = (sp7b["set"] == "train").to_numpy()
    allX = pd.get_dummies(pd.concat([icu[P10], tonight[P10]]), columns=["site"], drop_first=True, dtype=float)
    X7 = allX.iloc[: len(icu)].to_numpy(float); Xt7 = allX.iloc[len(icu):].to_numpy(float)
    (r_te7, r_tr7) = fit_bart(X7[tr7b], y7[tr7b], [np.vstack([X7[~tr7b], Xt7]), X7[tr7b]])
    nte = int((~tr7b).sum())
    f7, yn7 = r_te7[0][:, :nte], r_te7[1][:, :nte]
    put("ward7_reg", "bart", "test", id_te, f7.mean(axis=0))
    put("ward7_reg", "bart", "tonight", names, r_te7[0][:, nte:].mean(axis=0))
    put("ward7_aki", "bart_p10", "test", id_te, (yn7 >= 0.3).mean(axis=0))
    put("ward7_aki", "bart_p10", "tonight", names, (r_te7[1][:, nte:] >= 0.3).mean(axis=0))
    for ex in ["ward7_reg", "ward7_aki"]:
        note(ex, "bart" if ex == "ward7_reg" else "bart_p10",
             "50 trees; 2 chains x 1 000 draws + 1 000 tuning" +
             ("; posterior mean of f(x)" if ex == "ward7_reg" else "; P(new Δ ≥ 0.3) from the posterior predictive draws"),
             f"pymc {pymc.__version__}, pymc-bart {pymc_bart.__version__}")
elif os.path.exists(os.path.join(PARTS, "py_preds.csv")):
    old = pd.read_csv(os.path.join(PARTS, "py_preds.csv"))
    oldm = pd.read_csv(os.path.join(PARTS, "py_meta.csv"))
    keep = old["model_id"].isin(["bart", "bart_p10"])
    preds.append(old[keep]); meta.extend(oldm[oldm["model_id"].isin(["bart", "bart_p10"])].to_dict("records"))
    print("BART skipped: previous BART predictions kept")

pd.concat(preds, ignore_index=True).to_csv(os.path.join(PARTS, "py_preds.csv"), index=False)
pd.DataFrame(meta).to_csv(os.path.join(PARTS, "py_meta.csv"), index=False)
print("ledger.py done", flush=True)
