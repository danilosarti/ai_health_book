# Laboratório do capítulo 11
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Carregar e padronizar os dados do piloto ---------------------------------
library(cluster)
wear <- read.csv("data/wearable_phenotypes.csv")
vars <- c("HR_mean", "HRV_sdnn", "steps_mean", "skin_temp", "spo2", "sleep_eff")
X <- scale(wear[, vars])
truth <- wear$phenotype_true            # conhecido apenas porque os dados são simulados
D <- dist(X)

ari <- function(a, b) {                  # índice de Rand ajustado
  tab <- table(a, b); c2 <- function(x) sum(x * (x - 1) / 2)
  sij <- c2(tab); sa <- c2(rowSums(tab)); sb <- c2(colSums(tab))
  e <- sa * sb / c2(sum(tab)); (sij - e) / ((sa + sb) / 2 - e)
}

# 2. k-médias para K = 2..8: WSS, silhueta e concordância com a verdade -------
set.seed(2025)
res <- t(sapply(2:8, function(K) {
  km <- kmeans(X, centers = K, nstart = 20)
  c(K = K, WSS = km$tot.withinss,
    silhouette = mean(silhouette(km$cluster, D)[, "sil_width"]),
    ARI = ari(km$cluster, truth))
}))
round(res, 3)                            # a silhueta escolhe 4; a verdade é 5

# 3. A solução com quatro clusters contra a verdade ----------------------------
km4 <- kmeans(X, centers = 4, nstart = 20)
table(cluster = km4$cluster, phenotype = truth)
round(km4$centers, 2)                    # centroides em unidades de desvio-padrão

# 4. Agrupamento hierárquico da matriz genômica -------------------------------
gen <- read.csv("data/genomic_subtypes.csv")
G   <- scale(gen[, -(1:2)])
hc  <- hclust(dist(G), method = "ward.D2")
plot(hc, labels = FALSE, main = "Dendrograma de Ward, 60 pacientes", xlab = "", sub = "")
rect.hclust(hc, k = 3, border = c("#1f6f8b", "#b35c00", "#6a8f3a"))
ari(cutree(hc, k = 3), gen$subtype)

# 5. HomeBeat: clusters de um contínuo, e participantes com lacunas -----------
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
mean(placed == k3$cluster[inc])          # concordância com os clusters dos dados completos

# 6. Opcional, no seu computador: DBSCAN e misturas gaussianas ----------------
if (requireNamespace("dbscan", quietly = TRUE)) {
  db <- dbscan::dbscan(X, eps = 1.3, minPts = 10)
  print(table(db$cluster))               # cluster 0 = ruído
}
if (requireNamespace("mclust", quietly = TRUE)) {
  library(mclust)
  gmm <- Mclust(X, G = 1:8, verbose = FALSE)
  print(c(model = gmm$modelName, G = gmm$G))
  print(ari(gmm$classification, truth))  # os cinco fenótipos, recuperados
}
