# Laboratorio del capitolo 10
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Caricare la copia di Dermato-ATT per il browser -------------------------------
dm <- read.csv("data/dermato.csv")
dim(dm)
X <- dm[, setdiff(names(dm), c("patient_id", "Outcome_PASI12"))]   # solo le variabili
y <- dm$Outcome_PASI12

# 2. Distanze: grezze contro standardizzate ----------------------------------------
three <- X[1:3, c("PASI_baseline", "NO2", "gene_1")]
round(dist(three), 2)                      # grezze: domina l'NO2
round(dist(scale(X[, c("PASI_baseline", "NO2", "gene_1")])[1:3, ]), 2)

# 3. Concentrazione delle distanze nel puro rumore ---------------------------------
set.seed(1)
sapply(c(2, 10, 100, 1000), function(p) {
  d <- dist(matrix(rnorm(200 * p), 200)); round(min(d) / max(d), 3)
})

# 4. PCA (standardizzata) -----------------------------------------------------------
pca <- prcomp(X, scale. = TRUE)
pve <- pca$sdev^2 / sum(pca$sdev^2)
round(100 * pve[1:6], 1)                   # varianza spiegata (%)
barplot(100 * pve[1:15], names.arg = 1:15, xlab = "Componente", ylab = "Varianza spiegata (%)")

# Contributi = 100 x coefficienti al quadrato; sommano a 100 per ogni componente
contrib <- 100 * pca$rotation^2
colSums(contrib[, 1:3])
round(head(sort(contrib[, 1], decreasing = TRUE)), 2)
100 / ncol(X)                              # il contributo medio

# Loading (correlazioni tra variabili e componenti)
loading <- cor(X, pca$x[, 1:3])
round(loading[c("gene_1", "ASV_1", "cytokine_1", "NO2"), ], 2)

# 5. Le componenti prevedono l'esito? ----------------------------------------------
summary(lm(y ~ pca$x[, 1:5]))$r.squared   # cinque componenti
summary(lm(y ~ X$NO2))$r.squared           # una colonna dell'esposoma

# 6. PCA non standardizzata dell'esposoma: decidono le unità -----------------------
E <- X[, c("NO2", "PM10", "UV_index", "temperature", "humidity", paste0("expo_", 6:30))]
pe <- prcomp(E)                            # NON standardizzata
round(pe$sdev[1]^2 / sum(pe$sdev^2), 2)
names(which.max(abs(pe$rotation[, 1])))

# 7. Digital Trial: PCA e le mappe t-SNE / UMAP precalcolate ------------------------
dt    <- read.csv("data/digital_trial.csv")
truth <- read.csv("data/digital_trial_truth.csv")
maps  <- read.csv("data/precomputed/ch10/digital_maps.csv")
Z <- scale(dt[, setdiff(names(dt), c("patient_id", "delta_score", "responder"))])
pz <- prcomp(Z)
round(pz$sdev^2 / sum(pz$sdev^2), 2)
piece <- interaction(truth$shape, truth$X1 > 0, truth$X2 > 0, drop = TRUE)
par(mfrow = c(1, 3))
plot(pz$x[, 1:2], col = as.integer(piece), pch = 16, main = "PCA")
plot(maps$tsne_x, maps$tsne_y, col = as.integer(piece), pch = 16, main = "t-SNE (perplessità 30)")
plot(maps$umap_x, maps$umap_y, col = as.integer(piece), pch = 16, main = "UMAP (15 vicini)")
table(clinical_1_sopra_4 = dt$clinical_1 > 4, forma = truth$shape)

# 8. Facoltativo: ricalcolare t-SNE e UMAP sul tuo computer -------------------------
if (requireNamespace("Rtsne", quietly = TRUE) && requireNamespace("uwot", quietly = TRUE)) {
  set.seed(1)
  ts <- Rtsne::Rtsne(Z, perplexity = 30, pca = FALSE, check_duplicates = FALSE)
  um <- uwot::umap(Z, n_neighbors = 15, min_dist = 0.1, seed = 1, n_threads = 1)
  par(mfrow = c(1, 2))
  plot(ts$Y, col = as.integer(piece), pch = 16, main = "il tuo t-SNE")
  plot(um, col = as.integer(piece), pch = 16, main = "la tua UMAP")
}
