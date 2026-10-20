# Chapter 10 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Load the browser copy of Dermato-ATT -------------------------------------
dm <- read.csv("data/dermato.csv")
dim(dm)
X <- dm[, setdiff(names(dm), c("patient_id", "Outcome_PASI12"))]   # features only
y <- dm$Outcome_PASI12

# 2. Distances: raw versus standardised -----------------------------------------
three <- X[1:3, c("PASI_baseline", "NO2", "gene_1")]
round(dist(three), 2)                      # raw: NO2 dominates
round(dist(scale(X[, c("PASI_baseline", "NO2", "gene_1")])[1:3, ]), 2)

# 3. Distance concentration in pure noise -----------------------------------------
set.seed(1)
sapply(c(2, 10, 100, 1000), function(p) {
  d <- dist(matrix(rnorm(200 * p), 200)); round(min(d) / max(d), 3)
})

# 4. PCA (standardised) -------------------------------------------------------------
pca <- prcomp(X, scale. = TRUE)
pve <- pca$sdev^2 / sum(pca$sdev^2)
round(100 * pve[1:6], 1)                   # variance explained (%)
barplot(100 * pve[1:15], names.arg = 1:15, xlab = "Component", ylab = "Variance explained (%)")

# Contributions = 100 x squared coefficients; they sum to 100 for each component
contrib <- 100 * pca$rotation^2
colSums(contrib[, 1:3])
round(head(sort(contrib[, 1], decreasing = TRUE)), 2)
100 / ncol(X)                              # the average contribution

# Loadings (correlations between variables and components)
loading <- cor(X, pca$x[, 1:3])
round(loading[c("gene_1", "ASV_1", "cytokine_1", "NO2"), ], 2)

# 5. Do the components predict the outcome? -----------------------------------------
summary(lm(y ~ pca$x[, 1:5]))$r.squared   # five components
summary(lm(y ~ X$NO2))$r.squared           # one exposome column

# 6. Unscaled PCA of the exposome: units decide -------------------------------------
E <- X[, c("NO2", "PM10", "UV_index", "temperature", "humidity", paste0("expo_", 6:30))]
pe <- prcomp(E)                            # NOT standardised
round(pe$sdev[1]^2 / sum(pe$sdev^2), 2)
names(which.max(abs(pe$rotation[, 1])))

# 7. Digital Trial: PCA, and the precomputed t-SNE / UMAP maps -----------------------
dt    <- read.csv("data/digital_trial.csv")
truth <- read.csv("data/digital_trial_truth.csv")
maps  <- read.csv("data/precomputed/ch10/digital_maps.csv")
Z <- scale(dt[, setdiff(names(dt), c("patient_id", "delta_score", "responder"))])
pz <- prcomp(Z)
round(pz$sdev^2 / sum(pz$sdev^2), 2)
piece <- interaction(truth$shape, truth$X1 > 0, truth$X2 > 0, drop = TRUE)
par(mfrow = c(1, 3))
plot(pz$x[, 1:2], col = as.integer(piece), pch = 16, main = "PCA")
plot(maps$tsne_x, maps$tsne_y, col = as.integer(piece), pch = 16, main = "t-SNE (perplexity 30)")
plot(maps$umap_x, maps$umap_y, col = as.integer(piece), pch = 16, main = "UMAP (15 neighbours)")
table(clinical_1_above_4 = dt$clinical_1 > 4, shape = truth$shape)

# 8. Optional: recompute t-SNE and UMAP on your computer -----------------------------
if (requireNamespace("Rtsne", quietly = TRUE) && requireNamespace("uwot", quietly = TRUE)) {
  set.seed(1)
  ts <- Rtsne::Rtsne(Z, perplexity = 30, pca = FALSE, check_duplicates = FALSE)
  um <- uwot::umap(Z, n_neighbors = 15, min_dist = 0.1, seed = 1, n_threads = 1)
  par(mfrow = c(1, 2))
  plot(ts$Y, col = as.integer(piece), pch = 16, main = "your t-SNE")
  plot(um, col = as.integer(piece), pch = 16, main = "your UMAP")
}
