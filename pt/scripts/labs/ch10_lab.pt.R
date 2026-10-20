# Laboratório do capítulo 10
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Carregar a cópia da Dermato-ATT para o navegador ------------------------------
dm <- read.csv("data/dermato.csv")
dim(dm)
X <- dm[, setdiff(names(dm), c("patient_id", "Outcome_PASI12"))]   # apenas os atributos
y <- dm$Outcome_PASI12

# 2. Distâncias: brutas versus padronizadas ----------------------------------------
three <- X[1:3, c("PASI_baseline", "NO2", "gene_1")]
round(dist(three), 2)                      # brutas: o NO2 domina
round(dist(scale(X[, c("PASI_baseline", "NO2", "gene_1")])[1:3, ]), 2)

# 3. Concentração das distâncias em puro ruído -------------------------------------
set.seed(1)
sapply(c(2, 10, 100, 1000), function(p) {
  d <- dist(matrix(rnorm(200 * p), 200)); round(min(d) / max(d), 3)
})

# 4. PCA (padronizada) --------------------------------------------------------------
pca <- prcomp(X, scale. = TRUE)
pve <- pca$sdev^2 / sum(pca$sdev^2)
round(100 * pve[1:6], 1)                   # variância explicada (%)
barplot(100 * pve[1:15], names.arg = 1:15, xlab = "Componente", ylab = "Variância explicada (%)")

# Contribuições = 100 x coeficientes ao quadrado; somam 100 em cada componente
contrib <- 100 * pca$rotation^2
colSums(contrib[, 1:3])
round(head(sort(contrib[, 1], decreasing = TRUE)), 2)
100 / ncol(X)                              # a contribuição média

# Cargas (correlações entre variáveis e componentes)
loading <- cor(X, pca$x[, 1:3])
round(loading[c("gene_1", "ASV_1", "cytokine_1", "NO2"), ], 2)

# 5. As componentes predizem o desfecho? -------------------------------------------
summary(lm(y ~ pca$x[, 1:5]))$r.squared   # cinco componentes
summary(lm(y ~ X$NO2))$r.squared           # uma coluna do exposoma

# 6. PCA do exposoma sem padronização: as unidades decidem -------------------------
E <- X[, c("NO2", "PM10", "UV_index", "temperature", "humidity", paste0("expo_", 6:30))]
pe <- prcomp(E)                            # NÃO padronizada
round(pe$sdev[1]^2 / sum(pe$sdev^2), 2)
names(which.max(abs(pe$rotation[, 1])))

# 7. Digital Trial: PCA e os mapas t-SNE / UMAP pré-calculados ---------------------
dt    <- read.csv("data/digital_trial.csv")
truth <- read.csv("data/digital_trial_truth.csv")
maps  <- read.csv("data/precomputed/ch10/digital_maps.csv")
Z <- scale(dt[, setdiff(names(dt), c("patient_id", "delta_score", "responder"))])
pz <- prcomp(Z)
round(pz$sdev^2 / sum(pz$sdev^2), 2)
piece <- interaction(truth$shape, truth$X1 > 0, truth$X2 > 0, drop = TRUE)
par(mfrow = c(1, 3))
plot(pz$x[, 1:2], col = as.integer(piece), pch = 16, main = "PCA")
plot(maps$tsne_x, maps$tsne_y, col = as.integer(piece), pch = 16, main = "t-SNE (perplexidade 30)")
plot(maps$umap_x, maps$umap_y, col = as.integer(piece), pch = 16, main = "UMAP (15 vizinhos)")
table(clinical_1_acima_de_4 = dt$clinical_1 > 4, forma = truth$shape)

# 8. Opcional: recalcular o t-SNE e a UMAP no seu computador -----------------------
if (requireNamespace("Rtsne", quietly = TRUE) && requireNamespace("uwot", quietly = TRUE)) {
  set.seed(1)
  ts <- Rtsne::Rtsne(Z, perplexity = 30, pca = FALSE, check_duplicates = FALSE)
  um <- uwot::umap(Z, n_neighbors = 15, min_dist = 0.1, seed = 1, n_threads = 1)
  par(mfrow = c(1, 2))
  plot(ts$Y, col = as.integer(piece), pch = 16, main = "o seu t-SNE")
  plot(um, col = as.integer(piece), pch = 16, main = "a sua UMAP")
}
