# Laboratorio del capitolo 12
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 0. Pacchetti e dati ------------------------------------------------------
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

# 1. Due modelli dei giorni di follow-up (ricorda: la censura è ignorata) ---
lin = LinearRegression().fit(ltr[feats], ltr["time"])
rf = RandomForestRegressor(n_estimators=300, random_state=2026).fit(ltr[feats], ltr["time"])
print({"linear": round(mae(lte["time"], lin.predict(lte[feats])), 1),
       "forest": round(mae(lte["time"], rf.predict(lte[feats])), 1)})

# 2. Importanza per permutazione per qualsiasi modello -----------------------
def perm_importance(predict_fun, X, y, features, n_shuffles=10, seed=2026):
    rng = np.random.default_rng(seed)
    base = mae(y, predict_fun(X))
    out = {}
    for f in features:
        errs = []
        for _ in range(n_shuffles):
            Xp = X.copy(); Xp[f] = rng.permutation(Xp[f].to_numpy())
            errs.append(mae(y, predict_fun(Xp)))
        out[f] = np.mean(errs) - base                  # rimescolato meno originale
    return pd.Series(out).round(1)
print(perm_importance(lin.predict, lte[feats], lte["time"], feats))
print(perm_importance(rf.predict, lte[feats], lte["time"], feats))

# 3. Curve ICE e dipendenza parziale -------------------------------------------
def ice_curves(predict_fun, X, feature, grid):
    return np.column_stack([predict_fun(X.assign(**{feature: g})) for g in grid])  # pazienti x griglia
grid = np.arange(1, 26)
ice = ice_curves(rf.predict, ltr[feats], "wt.loss", grid)
c_ice = ice - ice[:, [0]]
print("Variazione del PDP da 1 a 25 kg:", round(float(c_ice[:, -1].mean()), 1),
      "| dispersione tra i pazienti (DS):", round(float(c_ice[:, -1].std()), 1))

# 4. H di Friedman per una coppia (su 60 pazienti di addestramento estratti) -
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

# 5. Valori di Shapley esatti per pazienti del Reparto 7: il modello OLS del Capitolo 4
#    (P10: i dieci predittori dei Capitoli 4 e 6; il sito conta come un solo giocatore)
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
print("OLS:", round(mae(test["delta_creatinine_48h"], ols7.predict(design(test))), 4))   # 0,1413, il valore del registro dei modelli (Capitolo 4)
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
# ogni colonna: riferimento + contributi = previsione
sh = pd.DataFrame({tonight.loc[i, "name"]: shapley_exact(lambda d: ols7.predict(design(d)), tonight.loc[i], background, f10)
                   for i in range(3)})
print(sh.round(3))

# 6. La foresta casuale del Capitolo 6 (500 alberi, scikit-learn) ------------------
rf7 = RandomForestRegressor(n_estimators=500, max_features=1/3, random_state=2026, n_jobs=1)
rf7.fit(design(train), train["delta_creatinine_48h"])
print("forest:", round(mae(test["delta_creatinine_48h"], rf7.predict(design(test))), 4))   # la stessa foresta del Capitolo 6 e del registro dei modelli: MAE di test 0,1528
#    I suoi valori di Shapley esatti richiedono circa un milione di previsioni per paziente,
#    perciò sono stati calcolati in anticipo da scripts/precompute/ch12_explain.py: si leggono.
shf = pd.read_csv("data/precomputed/ch12/ward7_shapley.csv")
shf = shf[shf["model"] == "Random forest"]
print(shf.pivot(index="feature", columns="name", values="phi").round(3))
# previsioni della foresta 0,29 / -0,09 / 0,83, come nel registro dei modelli
print(shf[["name", "baseline", "prediction"]].drop_duplicates())

# 7. vivid e bartMan sono pacchetti R senza una versione Python. L'equivalente più vicino di una
#    matrice VIVID: importanza per permutazione sulla diagonale (scikit-learn, pazienti di test) e
#    l'H non normalizzato del passo 4 fuori dalla diagonale. Per BART in Python esiste PyMC-BART, ma
#    la visualizzazione VSUP andrebbe disegnata a mano: per bartMan, esegui il codice R del Laboratorio.
from sklearn.inspection import permutation_importance
pi = permutation_importance(rf, lte[feats], lte["time"], scoring="neg_mean_absolute_error",
                            n_repeats=10, random_state=2026)
vivid_py = pd.DataFrame(0.0, index=feats, columns=feats)
for j, f in enumerate(feats):
    vivid_py.loc[f, f] = pi.importances_mean[j]          # aumento del MAE di test (giorni)
for a, b in itertools.combinations(feats, 2):              # H in giorni, sui 60 pazienti del passo 4
    vivid_py.loc[a, b] = vivid_py.loc[b, a] = h_stat(rf.predict, sub, a, b)["H_days"]
print(vivid_py.round(1))
