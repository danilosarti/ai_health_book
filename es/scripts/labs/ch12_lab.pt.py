# Laboratório do capítulo 12
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 0. Pacotes e dados -------------------------------------------------------
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

# 1. Dois modelos dos dias de seguimento (lembre-se: a censura é ignorada) ---
lin = LinearRegression().fit(ltr[feats], ltr["time"])
rf = RandomForestRegressor(n_estimators=300, random_state=2026).fit(ltr[feats], ltr["time"])
print({"linear": round(mae(lte["time"], lin.predict(lte[feats])), 1),
       "forest": round(mae(lte["time"], rf.predict(lte[feats])), 1)})

# 2. Importância por permutação para qualquer modelo -------------------------
def perm_importance(predict_fun, X, y, features, n_shuffles=10, seed=2026):
    rng = np.random.default_rng(seed)
    base = mae(y, predict_fun(X))
    out = {}
    for f in features:
        errs = []
        for _ in range(n_shuffles):
            Xp = X.copy(); Xp[f] = rng.permutation(Xp[f].to_numpy())
            errs.append(mae(y, predict_fun(Xp)))
        out[f] = np.mean(errs) - base                  # embaralhado menos original
    return pd.Series(out).round(1)
print(perm_importance(lin.predict, lte[feats], lte["time"], feats))
print(perm_importance(rf.predict, lte[feats], lte["time"], feats))

# 3. Curvas ICE e dependência parcial ------------------------------------------
def ice_curves(predict_fun, X, feature, grid):
    return np.column_stack([predict_fun(X.assign(**{feature: g})) for g in grid])  # pacientes x grade
grid = np.arange(1, 26)
ice = ice_curves(rf.predict, ltr[feats], "wt.loss", grid)
c_ice = ice - ice[:, [0]]
print("Variação do PDP de 1 a 25 kg:", round(float(c_ice[:, -1].mean()), 1),
      "| dispersão entre os pacientes (DP):", round(float(c_ice[:, -1].std()), 1))

# 4. H de Friedman para um par (em 60 pacientes de treino sorteados) ----------
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

# 5. Valores de Shapley exatos para pacientes da UTI Ala 7: o modelo OLS do Capítulo 4
#    (P10: os dez preditores dos Capítulos 4 e 6; o local conta como um único jogador)
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
print("OLS:", round(mae(test["delta_creatinine_48h"], ols7.predict(design(test))), 4))   # 0,1413, o valor do registro de modelos (Capítulo 4)
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
# cada coluna: referência + contribuições = predição
sh = pd.DataFrame({tonight.loc[i, "name"]: shapley_exact(lambda d: ols7.predict(design(d)), tonight.loc[i], background, f10)
                   for i in range(3)})
print(sh.round(3))

# 6. A floresta aleatória do Capítulo 6 (500 árvores, scikit-learn) ---------------
rf7 = RandomForestRegressor(n_estimators=500, max_features=1/3, random_state=2026, n_jobs=1)
rf7.fit(design(train), train["delta_creatinine_48h"])
print("forest:", round(mae(test["delta_creatinine_48h"], rf7.predict(design(test))), 4))   # a mesma floresta do Capítulo 6 e do registro de modelos: EAM de teste 0,1528
#    Seus valores de Shapley exatos exigem cerca de um milhão de predições por paciente,
#    por isso foram calculados antes por scripts/precompute/ch12_explain.py: leia-os.
shf = pd.read_csv("data/precomputed/ch12/ward7_shapley.csv")
shf = shf[shf["model"] == "Random forest"]
print(shf.pivot(index="feature", columns="name", values="phi").round(3))
# predições da floresta 0,29 / -0,09 / 0,83, como no registro de modelos
print(shf[["name", "baseline", "prediction"]].drop_duplicates())

# 7. vivid e bartMan são pacotes de R sem versão em Python. O equivalente mais próximo de uma
#    matriz VIVID: importância por permutação na diagonal (scikit-learn, pacientes de teste) e o
#    H não normalizado do passo 4 fora da diagonal. Para o BART em Python existe o PyMC-BART, mas a
#    exibição VSUP teria de ser desenhada à mão: para o bartMan, rode o código em R do Lab.
from sklearn.inspection import permutation_importance
pi = permutation_importance(rf, lte[feats], lte["time"], scoring="neg_mean_absolute_error",
                            n_repeats=10, random_state=2026)
vivid_py = pd.DataFrame(0.0, index=feats, columns=feats)
for j, f in enumerate(feats):
    vivid_py.loc[f, f] = pi.importances_mean[j]          # aumento do MAE de teste (dias)
for a, b in itertools.combinations(feats, 2):              # H em dias, nos 60 pacientes do passo 4
    vivid_py.loc[a, b] = vivid_py.loc[b, a] = h_stat(rf.predict, sub, a, b)["H_days"]
print(vivid_py.round(1))
