# Laboratorio del capitolo 11
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Caricare e standardizzare i dati del pilota ------------------------------
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score as ari
from scipy.cluster.hierarchy import linkage, fcluster

wear = pd.read_csv("data/wearable_phenotypes.csv")
vars = ["HR_mean", "HRV_sdnn", "steps_mean", "skin_temp", "spo2", "sleep_eff"]
X = ((wear[vars] - wear[vars].mean()) / wear[vars].std()).to_numpy()
truth = wear["phenotype_true"]           # noto solo perché i dati sono simulati

# 2. k-means per K = 2..8: WSS, silhouette e accordo con la verità ------------
rows = []
for K in range(2, 9):
    km = KMeans(n_clusters=K, n_init=20, random_state=2025).fit(X)
    rows.append({"K": K, "WSS": km.inertia_, "silhouette": silhouette_score(X, km.labels_),
                 "ARI": ari(km.labels_, truth)})
print(pd.DataFrame(rows).round(3))       # la silhouette sceglie 4; la verità è 5

# 3. La soluzione a quattro cluster contro la verità ---------------------------
km4 = KMeans(n_clusters=4, n_init=20, random_state=2025).fit(X)
print(pd.crosstab(km4.labels_, truth))

# 4. DBSCAN e una mistura gaussiana con covarianza comune ('tied' = EEE di mclust)
db = DBSCAN(eps=1.3, min_samples=10).fit(X)
print("Cluster DBSCAN:", len(set(db.labels_) - {-1}), " rumore:", int((db.labels_ == -1).sum()))
bics = {G: GaussianMixture(G, covariance_type="tied", n_init=5, random_state=2025).fit(X).bic(X) for G in range(2, 9)}
G_best = min(bics, key=bics.get)         # BIC di scikit-learn: più basso è meglio
gmm = GaussianMixture(G_best, covariance_type="tied", n_init=5, random_state=2025).fit(X)
print("Componenti del GMM:", G_best, " ARI:", round(ari(gmm.predict(X), truth), 3))

# 5. Clustering gerarchico della matrice genomica -----------------------------
gen = pd.read_csv("data/genomic_subtypes.csv")
G = gen.iloc[:, 2:]
G = (G - G.mean()) / G.std()
Z = linkage(G.to_numpy(), method="ward")
print("Ward, 3 cluster, ARI:", ari(fcluster(Z, t=3, criterion="maxclust"), gen["subtype"]))

# 6. HomeBeat: cluster di un continuo -----------------------------------------
hb = pd.read_csv("data/homebeat.csv")
sig = ["resting_hr", "hrv", "skin_temp", "activity_index"]
H = ((hb[sig] - hb[sig].mean()) / hb[sig].std()).to_numpy()
k3 = KMeans(n_clusters=3, n_init=20, random_state=2025).fit(H)
print(hb.assign(cluster=k3.labels_).groupby("cluster")[["age", "event_occurred"]].mean().round(2))
