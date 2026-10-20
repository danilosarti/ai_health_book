# Chapter 11 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Load and standardise the pilot data ------------------------------------
library(cluster)
wear <- read.csv("data/wearable_phenotypes.csv")
vars <- c("HR_mean", "HRV_sdnn", "steps_mean", "skin_temp", "spo2", "sleep_eff")
X <- scale(wear[, vars])
truth <- wear$phenotype_true            # known only because the data are simulated
D <- dist(X)

ari <- function(a, b) {                  # adjusted Rand index
  tab <- table(a, b); c2 <- function(x) sum(x * (x - 1) / 2)
  sij <- c2(tab); sa <- c2(rowSums(tab)); sb <- c2(colSums(tab))
  e <- sa * sb / c2(sum(tab)); (sij - e) / ((sa + sb) / 2 - e)
}

# 2. k-means for K = 2..8: WSS, silhouette and agreement with the truth -------
set.seed(2025)
res <- t(sapply(2:8, function(K) {
  km <- kmeans(X, centers = K, nstart = 20)
  c(K = K, WSS = km$tot.withinss,
    silhouette = mean(silhouette(km$cluster, D)[, "sil_width"]),
    ARI = ari(km$cluster, truth))
}))
round(res, 3)                            # silhouette chooses 4; the truth is 5

# 3. The four-cluster solution against the truth -------------------------------
km4 <- kmeans(X, centers = 4, nstart = 20)
table(cluster = km4$cluster, phenotype = truth)
round(km4$centers, 2)                    # centroids in standard-deviation units

# 4. Hierarchical clustering of the genomic matrix ----------------------------
gen <- read.csv("data/genomic_subtypes.csv")
G   <- scale(gen[, -(1:2)])
hc  <- hclust(dist(G), method = "ward.D2")
plot(hc, labels = FALSE, main = "Ward dendrogram, 60 patients", xlab = "", sub = "")
rect.hclust(hc, k = 3, border = c("#1f6f8b", "#b35c00", "#6a8f3a"))
ari(cutree(hc, k = 3), gen$subtype)

# 5. HomeBeat: clusters of a continuum, and participants with gaps ------------
hb  <- read.csv("data/homebeat.csv")
hbm <- read.csv("data/homebeat_missing.csv")
sig <- c("resting_hr", "hrv", "skin_temp", "activity_index")
mu <- colMeans(hb[, sig]); s <- apply(hb[, sig], 2, sd)
H  <- scale(hb[, sig], center = mu, scale = s)
k3 <- kmeans(H, centers = 3, nstart = 20)
hb$cluster <- k3$cluster
aggregate(cbind(age, event_occurred) ~ cluster, data = hb, FUN = mean)
Hm  <- scale(hbm[, sig], center = mu, scale = s)
inc <- which(!complete.cases(Hm))
nearest_available <- function(x, C) { ok <- !is.na(x); which.min(colSums((t(C[, ok, drop = FALSE]) - x[ok])^2)) }
placed <- apply(Hm[inc, ], 1, nearest_available, C = k3$centers)
mean(placed == k3$cluster[inc])          # agreement with the complete-data clusters

# 6. Optional, on your computer: DBSCAN and Gaussian mixtures -----------------
if (requireNamespace("dbscan", quietly = TRUE)) {
  db <- dbscan::dbscan(X, eps = 1.3, minPts = 10)
  print(table(db$cluster))               # cluster 0 = noise
}
if (requireNamespace("mclust", quietly = TRUE)) {
  library(mclust)
  gmm <- Mclust(X, G = 1:8, verbose = FALSE)
  print(c(model = gmm$modelName, G = gmm$G))
  print(ari(gmm$classification, truth))  # the five phenotypes, recovered
}
