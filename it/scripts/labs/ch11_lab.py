# Chapter 11 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Load and standardise the pilot data ------------------------------------
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score as ari
from scipy.cluster.hierarchy import linkage, fcluster

wear = pd.read_csv("data/wearable_phenotypes.csv")
vars = ["HR_mean", "HRV_sdnn", "steps_mean", "skin_temp", "spo2", "sleep_eff"]
X = ((wear[vars] - wear[vars].mean()) / wear[vars].std()).to_numpy()
truth = wear["phenotype_true"]           # known only because the data are simulated

# 2. k-means for K = 2..8: WSS, silhouette and agreement with the truth -------
rows = []
for K in range(2, 9):
    km = KMeans(n_clusters=K, n_init=20, random_state=2025).fit(X)
    rows.append({"K": K, "WSS": km.inertia_, "silhouette": silhouette_score(X, km.labels_),
                 "ARI": ari(km.labels_, truth)})
print(pd.DataFrame(rows).round(3))       # silhouette chooses 4; the truth is 5

# 3. The four-cluster solution against the truth -------------------------------
km4 = KMeans(n_clusters=4, n_init=20, random_state=2025).fit(X)
print(pd.crosstab(km4.labels_, truth))

# 4. DBSCAN and a Gaussian mixture with a shared covariance ('tied' = mclust EEE)
db = DBSCAN(eps=1.3, min_samples=10).fit(X)
print("DBSCAN clusters:", len(set(db.labels_) - {-1}), " noise:", int((db.labels_ == -1).sum()))
bics = {G: GaussianMixture(G, covariance_type="tied", n_init=5, random_state=2025).fit(X).bic(X) for G in range(2, 9)}
G_best = min(bics, key=bics.get)         # scikit-learn's BIC: lower is better
gmm = GaussianMixture(G_best, covariance_type="tied", n_init=5, random_state=2025).fit(X)
print("GMM components:", G_best, " ARI:", round(ari(gmm.predict(X), truth), 3))

# 5. Hierarchical clustering of the genomic matrix ----------------------------
gen = pd.read_csv("data/genomic_subtypes.csv")
G = gen.iloc[:, 2:]
G = (G - G.mean()) / G.std()
Z = linkage(G.to_numpy(), method="ward")
print("Ward, 3 clusters, ARI:", ari(fcluster(Z, t=3, criterion="maxclust"), gen["subtype"]))

# 6. HomeBeat: clusters of a continuum ----------------------------------------
hb = pd.read_csv("data/homebeat.csv")
sig = ["resting_hr", "hrv", "skin_temp", "activity_index"]
H = ((hb[sig] - hb[sig].mean()) / hb[sig].std()).to_numpy()
k3 = KMeans(n_clusters=3, n_init=20, random_state=2025).fit(H)
print(hb.assign(cluster=k3.labels_).groupby("cluster")[["age", "event_occurred"]].mean().round(2))
