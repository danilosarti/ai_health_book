# Laboratório do capítulo 10
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Carregar a cópia da Dermato-ATT para o navegador ------------------------------
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

# 2. Distâncias: brutas versus padronizadas ----------------------------------------
cols = ["PASI_baseline", "NO2", "gene_1"]
print(np.round(pdist(X[cols].iloc[:3]), 2))                 # brutas: o NO2 domina
Zc = (X[cols] - X[cols].mean()) / X[cols].std()
print(np.round(pdist(Zc.iloc[:3]), 2))

# 3. Concentração das distâncias em puro ruído -------------------------------------
rng = np.random.default_rng(1)
for p in [2, 10, 100, 1000]:
    d = pdist(rng.standard_normal((200, p)))
    print(p, round(d.min() / d.max(), 3))

# 4. PCA (padronizada) --------------------------------------------------------------
Z = StandardScaler().fit_transform(X)
pca = PCA().fit(Z)
print(np.round(100 * pca.explained_variance_ratio_[:6], 1))
contrib = 100 * pca.components_ ** 2                         # linhas = componentes
print(contrib[:3].sum(axis=1))                               # 100 cada
top = pd.Series(contrib[0], index=X.columns).sort_values(ascending=False)
print(top.head().round(2), "média:", round(100 / X.shape[1], 3))
scores = pca.transform(Z)
loading = pd.DataFrame(np.corrcoef(Z.T, scores[:, :3].T)[:-3, -3:], index=X.columns)
print(loading.loc[["gene_1", "ASV_1", "cytokine_1", "NO2"]].round(2))

# 5. As componentes predizem o desfecho? -------------------------------------------
print("R2, cinco componentes:", round(LinearRegression().fit(scores[:, :5], y).score(scores[:, :5], y), 3))
print("R2, só o NO2:", round(LinearRegression().fit(X[["NO2"]], y).score(X[["NO2"]], y), 3))

# 6. PCA do exposoma sem padronização: as unidades decidem -------------------------
E = X[["NO2", "PM10", "UV_index", "temperature", "humidity"] + [f"expo_{i}" for i in range(6, 31)]]
pe = PCA().fit(E)                                            # NÃO padronizada
print(round(pe.explained_variance_ratio_[0], 2), E.columns[np.abs(pe.components_[0]).argmax()])

# 7. Digital Trial: PCA e os mapas t-SNE / UMAP pré-calculados ---------------------
dt = pd.read_csv("data/digital_trial.csv")
truth = pd.read_csv("data/digital_trial_truth.csv")
maps = pd.read_csv("data/precomputed/ch10/digital_maps.csv")
Zd = StandardScaler().fit_transform(dt.drop(columns=["patient_id", "delta_score", "responder"]))
print(np.round(PCA().fit(Zd).explained_variance_ratio_, 2))
piece = truth["shape"] + np.where(truth["X1"] > 0, " X1+", " X1-") + np.where(truth["X2"] > 0, " X2+", " X2-")
# Cada ilha do mapa t-SNE é um pedaço: a posição média de cada pedaço no mapa
print(maps.assign(piece=piece).groupby("piece")[["tsne_x", "tsne_y"]].mean().round(1))
print(pd.crosstab(dt["clinical_1"] > 4, truth["shape"]))
