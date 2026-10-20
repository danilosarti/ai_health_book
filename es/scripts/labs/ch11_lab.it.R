# Laboratorio del capitolo 11
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Caricare e standardizzare i dati del pilota ------------------------------
library(cluster)
wear <- read.csv("data/wearable_phenotypes.csv")
vars <- c("HR_mean", "HRV_sdnn", "steps_mean", "skin_temp", "spo2", "sleep_eff")
X <- scale(wear[, vars])
truth <- wear$phenotype_true            # noto solo perché i dati sono simulati
D <- dist(X)

ari <- function(a, b) {                  # indice di Rand aggiustato
  tab <- table(a, b); c2 <- function(x) sum(x * (x - 1) / 2)
  sij <- c2(tab); sa <- c2(rowSums(tab)); sb <- c2(colSums(tab))
  e <- sa * sb / c2(sum(tab)); (sij - e) / ((sa + sb) / 2 - e)
}

# 2. k-means per K = 2..8: WSS, silhouette e accordo con la verità ------------
set.seed(2025)
res <- t(sapply(2:8, function(K) {
  km <- kmeans(X, centers = K, nstart = 20)
  c(K = K, WSS = km$tot.withinss,
    silhouette = mean(silhouette(km$cluster, D)[, "sil_width"]),
    ARI = ari(km$cluster, truth))
}))
round(res, 3)                            # la silhouette sceglie 4; la verità è 5

# 3. La soluzione a quattro cluster contro la verità ---------------------------
km4 <- kmeans(X, centers = 4, nstart = 20)
table(cluster = km4$cluster, phenotype = truth)
round(km4$centers, 2)                    # centroidi in unità di deviazione standard

# 4. Clustering gerarchico della matrice genomica -----------------------------
gen <- read.csv("data/genomic_subtypes.csv")
G   <- scale(gen[, -(1:2)])
hc  <- hclust(dist(G), method = "ward.D2")
plot(hc, labels = FALSE, main = "Dendrogramma di Ward, 60 pazienti", xlab = "", sub = "")
rect.hclust(hc, k = 3, border = c("#1f6f8b", "#b35c00", "#6a8f3a"))
ari(cutree(hc, k = 3), gen$subtype)

# 5. HomeBeat: cluster di un continuo, e partecipanti con lacune --------------
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
mean(placed == k3$cluster[inc])          # accordo con i cluster ottenuti dai dati completi

# 6. Facoltativo, sul tuo computer: DBSCAN e misture gaussiane ----------------
if (requireNamespace("dbscan", quietly = TRUE)) {
  db <- dbscan::dbscan(X, eps = 1.3, minPts = 10)
  print(table(db$cluster))               # cluster 0 = rumore
}
if (requireNamespace("mclust", quietly = TRUE)) {
  library(mclust)
  gmm <- Mclust(X, G = 1:8, verbose = FALSE)
  print(c(model = gmm$modelName, G = gmm$G))
  print(ari(gmm$classification, truth))  # i cinque fenotipi, recuperati
}
