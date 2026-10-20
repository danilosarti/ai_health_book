# Laboratorio del capítulo 10
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. Cargar la copia de Dermato-ATT para el navegador --------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist

dm = pd.read_csv("data/dermato.csv")
print(dm.shape)
X = dm.drop(columns=["patient_id", "Outcome_PASI12"])
y = dm["Outcome_PASI12"]

# 2. Distancias: brutas frente a estandarizadas --------------------------------
cols = ["PASI_baseline", "NO2", "gene_1"]
print(np.round(pdist(X[cols].iloc[:3]), 2))                 # brutas: domina el NO2
Zc = (X[cols] - X[cols].mean()) / X[cols].std()
print(np.round(pdist(Zc.iloc[:3]), 2))

# 3. Concentración de las distancias en ruido puro -----------------------------
rng = np.random.default_rng(1)
for p in [2, 10, 100, 1000]:
    d = pdist(rng.standard_normal((200, p)))
    print(p, round(d.min() / d.max(), 3))

# 4. PCA (estandarizada) -------------------------------------------------------------
Z = StandardScaler().fit_transform(X)
pca = PCA().fit(Z)
print(np.round(100 * pca.explained_variance_ratio_[:6], 1))
contrib = 100 * pca.components_ ** 2                         # filas = componentes
print(contrib[:3].sum(axis=1))                               # 100 cada uno
top = pd.Series(contrib[0], index=X.columns).sort_values(ascending=False)
print(top.head().round(2), "media:", round(100 / X.shape[1], 3))
scores = pca.transform(Z)
loading = pd.DataFrame(np.corrcoef(Z.T, scores[:, :3].T)[:-3, -3:], index=X.columns)
print(loading.loc[["gene_1", "ASV_1", "cytokine_1", "NO2"]].round(2))

# 5. ¿Predicen los componentes el desenlace? -----------------------------------------
print("R2, cinco componentes:", round(LinearRegression().fit(scores[:, :5], y).score(scores[:, :5], y), 3))
print("R2, solo el NO2:", round(LinearRegression().fit(X[["NO2"]], y).score(X[["NO2"]], y), 3))

# 6. PCA del exposoma sin estandarizar: deciden las unidades --------------------------
E = X[["NO2", "PM10", "UV_index", "temperature", "humidity"] + [f"expo_{i}" for i in range(6, 31)]]
pe = PCA().fit(E)                                            # NO estandarizada
print(round(pe.explained_variance_ratio_[0], 2), E.columns[np.abs(pe.components_[0]).argmax()])

# 7. Digital Trial: PCA y los mapas t-SNE / UMAP precalculados ------------------------
dt = pd.read_csv("data/digital_trial.csv")
truth = pd.read_csv("data/digital_trial_truth.csv")
maps = pd.read_csv("data/precomputed/ch10/digital_maps.csv")
Zd = StandardScaler().fit_transform(dt.drop(columns=["patient_id", "delta_score", "responder"]))
print(np.round(PCA().fit(Zd).explained_variance_ratio_, 2))
piece = truth["shape"] + np.where(truth["X1"] > 0, " X1+", " X1-") + np.where(truth["X2"] > 0, " X2+", " X2-")
# Cada isla del mapa t-SNE es un trozo: la posición media de cada trozo en el mapa
print(maps.assign(piece=piece).groupby("piece")[["tsne_x", "tsne_y"]].mean().round(1))
print(pd.crosstab(dt["clinical_1"] > 4, truth["shape"]))
