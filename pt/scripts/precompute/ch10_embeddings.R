# =====================================================================
# Chapter 10 — t-SNE and UMAP embeddings of the Digital Trial dataset
# (500 participants, 11 standardised features), precomputed for the
# "perplexity grid" widget and the figures. t-SNE and UMAP are too heavy for
# the browser engines, so every map shown in the chapter comes from here.
#
# Grid: t-SNE perplexity 5, 15, 30, 50, 100 and UMAP n_neighbors 5, 15, 30,
# 50, 100 (min_dist 0.1), each with two random seeds, plus PCA.
# For every map we also record two quality measures:
#   knn10    share of each participant's 10 nearest neighbours in the
#            standardised 11-dimensional space that are also among its 10
#            nearest neighbours in the map (local structure);
#   dist_cor Spearman correlation between all pairwise distances in the
#            11-dimensional space and in the map (global structure).
# Usage (from the book root): Rscript scripts/precompute/ch10_embeddings.R
# Needs: Rtsne, uwot, jsonlite. Single-threaded on purpose (reproducible);
# about one minute on two CPUs.
# =====================================================================
suppressPackageStartupMessages({ library(Rtsne); library(uwot); library(jsonlite) })
out <- "data/precomputed/ch10"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

dt    <- read.csv("data/digital_trial.csv")
truth <- read.csv("data/digital_trial_truth.csv")
feat  <- setdiff(names(dt), c("patient_id", "delta_score", "responder"))
X     <- scale(as.matrix(dt[, feat]))
n     <- nrow(X)

D_hi <- as.matrix(dist(X))
nn_of <- function(D, k = 10) t(apply(D, 1, function(r) order(r)[2:(k + 1)]))
NN_hi <- nn_of(D_hi)
lower <- lower.tri(D_hi)
quality <- function(Y) {
  D_lo <- as.matrix(dist(Y))
  NN_lo <- nn_of(D_lo)
  knn <- mean(vapply(seq_len(n), function(i) length(intersect(NN_hi[i, ], NN_lo[i, ])) / 10, 0))
  c(knn10 = knn, dist_cor = cor(D_hi[lower], D_lo[lower], method = "spearman"))
}

maps <- list()
add <- function(method, param, seed, Y) {
  q <- quality(Y)
  maps[[length(maps) + 1]] <<- list(method = method, param = param, seed = seed,
                                     x = round(Y[, 1], 2), y = round(Y[, 2], 2),
                                     knn10 = round(unname(q["knn10"]), 3),
                                     dist_cor = round(unname(q["dist_cor"]), 3))
  cat(sprintf("%-5s %4s seed %d  knn10 %.3f  dist_cor %.3f\n", method, param, seed, q["knn10"], q["dist_cor"]))
}

pc <- prcomp(X)
add("PCA", 0, 0, pc$x[, 1:2])
pve <- pc$sdev^2 / sum(pc$sdev^2)

for (perp in c(5, 15, 30, 50, 100)) for (seed in 1:2) {
  set.seed(seed)
  fit <- Rtsne(X, dims = 2, perplexity = perp, theta = 0.5, max_iter = 1000,
               pca = FALSE, check_duplicates = FALSE, num_threads = 1)
  add("tSNE", perp, seed, fit$Y)
}
for (nb in c(5, 15, 30, 50, 100)) for (seed in 1:2) {
  set.seed(seed)
  Y <- umap(X, n_neighbors = nb, min_dist = 0.1, n_components = 2, metric = "euclidean",
            n_threads = 1, n_sgd_threads = 1, seed = seed)
  add("UMAP", nb, seed, Y)
}

# "piece": the shape and the signs of X1 and X2. The step-shaped clinical
# variables (clinical_2 jumps where X2 = 0, clinical_3 where X1 = 0, clinical_1
# between the shapes) cut the two shapes into up to eight pieces.
piece <- paste0(truth$shape, ifelse(truth$X1 > 0, ", X1>0", ", X1<0"), ifelse(truth$X2 > 0, ", X2>0", ", X2<0"))
points <- data.frame(id = dt$patient_id, shape = truth$shape, piece = piece, position = truth$position,
                     delta = round(dt$delta_score, 3), clinical_1 = round(dt$clinical_1, 3),
                     wearable_4 = round(dt$wearable_4, 3))
write_json(list(features = feat, pca_pve = round(pve, 4), points = points, maps = maps),
           file.path(out, "digital_embeddings.json"), auto_unbox = TRUE, digits = NA)

# Compact table of the quality measures for the text
qt <- do.call(rbind, lapply(maps, function(m) data.frame(method = m$method, param = m$param,
                                                      seed = m$seed, knn10 = m$knn10, dist_cor = m$dist_cor)))
write.csv(qt, file.path(out, "digital_quality.csv"), row.names = FALSE)


# Two maps as a plain CSV for the Lab (base R / pandas, no JSON reader needed)
pick <- function(meth, par, sd) Filter(function(m) m$method == meth && m$param == par && m$seed == sd, maps)[[1]]
ts <- pick("tSNE", 30, 1); um <- pick("UMAP", 15, 1)
write.csv(data.frame(patient_id = dt$patient_id, tsne_x = ts$x, tsne_y = ts$y, umap_x = um$x, umap_y = um$y),
          file.path(out, "digital_maps.csv"), row.names = FALSE)
cat("ch10_embeddings.R done\n")
