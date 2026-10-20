# Laboratorio del capítulo 6
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 0. Paquetes y datos ------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit

icu = pd.read_csv("data/icu_aki.csv")
tonight = pd.read_csv("data/ward7_tonight.csv")
feat = ["site", "age", "sofa", "lactate", "map_min", "baseline_creat",
        "protocol_adherence", "bun", "crp", "albumin"]
X = pd.get_dummies(icu[feat], drop_first=True, dtype=float)
y = icu["delta_creatinine_48h"]
split = pd.read_csv("data/splits/ward7_split.csv")          # división compartida del Capítulo 3
m = split["set"].eq("train").to_numpy()
Xtr, Xte, ytr, yte = X[m], X[~m], y[m], y[~m]
folds = PredefinedSplit(split.loc[m, "fold"].astype(int).to_numpy() - 1)   # cinco particiones fijas
Xnew = pd.get_dummies(tonight[feat], dtype=float).reindex(columns=X.columns, fill_value=0.0)
mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

# 1. Elegir la profundidad de un árbol por VC con 5 particiones (compartidas) -----
grid = GridSearchCV(DecisionTreeRegressor(random_state=2026), {"max_depth": list(range(1, 9))},
                    cv=folds, scoring="neg_mean_absolute_error").fit(Xtr, ytr)
tree = grid.best_estimator_     # el árbol de scikit-learn; las tablas del libro usan el árbol 1-SE de rpart (Laboratorio en R)
print("profundidad elegida:", grid.best_params_["max_depth"])
print(export_text(DecisionTreeRegressor(max_depth=2).fit(Xtr, ytr), feature_names=list(X.columns)))

# 2. El bosque de 500 árboles de las comparaciones del libro (unos 10 s) ----------
rf = RandomForestRegressor(n_estimators=500, max_features=1/3, oob_score=True,
                           random_state=2026).fit(Xtr, ytr)
print("MAE OOB:", round(mae(ytr, rf.oob_prediction_), 3))

# 3. Comparar en los pacientes de prueba ------------------------------------------
lin = LinearRegression().fit(Xtr, ytr)
print({"average": round(mae(yte, np.full(len(yte), ytr.mean())), 3),
       "tree": round(mae(yte, tree.predict(Xte)), 3),
       "forest": round(mae(yte, rf.predict(Xte)), 3),
       "linear": round(mae(yte, lin.predict(Xte)), 3)})
# XGBoost no se ejecuta en el navegador: léelo, junto con todos los demás modelos, del ledger del libro
led = pd.read_csv("data/precomputed/ledger/ward7_reg.csv")
print(led.loc[(led["chapter"] <= 6) & (led["role"] != "variant"), ["model", "test_MAE"]].round(3).to_string(index=False))

# 4. Los pacientes de esta noche ---------------------------------------------------
print(pd.DataFrame({"name": tonight["name"], "tree": tree.predict(Xnew).round(2),
                    "forest": rf.predict(Xnew).round(2)}))

# 5. ONCO-1: cribado de fugas antes de modelar ------------------------------------
onco = pd.read_csv("data/onco1.csv")
om = pd.read_csv("data/splits/onco1_split.csv")["set"].eq("train").to_numpy()
otr, ote = onco[om], onco[~om]
genes = [c for c in onco.columns if c.startswith("gene_")]
print(otr[genes].corrwith(otr["response_percent"]).round(2).sort_values(ascending=False).head(5))

# 6. ...y ver qué hacen los genes con fuga en un árbol de regresión ---------------
drop = ["patient_id", "baseline_tumor_mm", "post_tumor_mm", "high_response", "response_percent"]
Xo = pd.get_dummies(onco.drop(columns=drop), drop_first=True, dtype=float)
clean = [c for c in Xo.columns if c not in ["gene_08", "gene_14", "gene_19"]]
yo = onco["response_percent"]
t_all = DecisionTreeRegressor(max_depth=5, random_state=0).fit(Xo[om], yo[om])
t_cln = DecisionTreeRegressor(max_depth=5, random_state=0).fit(Xo[om][clean], yo[om])
print({"with_leaky_genes": round(mae(yo[~om], t_all.predict(Xo[~om])), 2),
       "without_leaky_genes": round(mae(yo[~om], t_cln.predict(Xo[~om][clean])), 2)})
