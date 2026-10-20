# =====================================================================
# Chapter 11: precomputed results (clustering patients)
# Usage (from the book folder): Rscript scripts/precompute/ch11_clustering.R
# Needs: data/wearable_phenotypes.csv, data/genomic_subtypes.csv,
#        data/homebeat.csv, data/homebeat_missing.csv
#        (run scripts/simulate/make_ch11_data.R first) and the R packages
#        cluster, dbscan and mclust. Runs in well under a minute on one core.
# Writes small CSV/JSON files to data/precomputed/ch11/.
# =====================================================================
suppressMessages({ library(cluster); library(dbscan); library(mclust) })
out <- "data/precomputed/ch11"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
vars <- c("HR_mean", "HRV_sdnn", "steps_mean", "skin_temp", "spo2", "sleep_eff")

# ---- helpers: internal indices written out in base R ---------------------
centroids <- function(X, cl) t(sapply(sort(unique(cl)), function(k) colMeans(X[cl == k, , drop = FALSE])))
wss_of <- function(X, cl) sum(sapply(unique(cl), function(k) {
  Z <- X[cl == k, , drop = FALSE]; sum(sweep(Z, 2, colMeans(Z))^2) }))
davies_bouldin <- function(X, cl) {
  ks <- sort(unique(cl)); C <- centroids(X, cl)
  S <- sapply(seq_along(ks), function(j) mean(sqrt(rowSums(sweep(X[cl == ks[j], , drop = FALSE], 2, C[j, ])^2))))
  R <- outer(S, S, "+") / as.matrix(dist(C)); diag(R) <- -Inf
  mean(apply(R, 1, max))
}
calinski_harabasz <- function(X, cl) {
  n <- nrow(X); K <- length(unique(cl)); W <- wss_of(X, cl)
  tot <- sum(sweep(X, 2, colMeans(X))^2)
  ((tot - W) / (K - 1)) / (W / (n - K))
}
dunn_index <- function(Dm, cl) {
  ks <- sort(unique(cl))
  diam <- max(sapply(ks, function(k) max(Dm[cl == k, cl == k])))
  sep <- Inf
  for (a in ks) for (b in ks) if (a < b) sep <- min(sep, min(Dm[cl == a, cl == b]))
  sep / diam
}
nearest <- function(X, C) apply(X, 1, function(x) which.min(colSums((t(C) - x)^2)))

# ---- 1. Wearable phenotypes: k-means for K = 2..8 -------------------------
wear  <- read.csv("data/wearable_phenotypes.csv")
X     <- scale(as.matrix(wear[, vars]))
truth <- wear$phenotype_true
D <- dist(X); Dm <- as.matrix(D)

set.seed(2025)
fits <- lapply(2:8, function(K) kmeans(X, centers = K, nstart = 20, iter.max = 100))
idx <- data.frame(
  K          = 2:8,
  wss        = sapply(fits, `[[`, "tot.withinss"),
  silhouette = sapply(fits, function(f) mean(silhouette(f$cluster, D)[, "sil_width"])),
  db         = sapply(fits, function(f) davies_bouldin(X, f$cluster)),
  ch         = sapply(fits, function(f) calinski_harabasz(X, f$cluster)),
  dunn       = sapply(fits, function(f) dunn_index(Dm, f$cluster)),
  ari_truth  = sapply(fits, function(f) adjustedRandIndex(f$cluster, truth)))

# bootstrap stability: refit on resamples, assign everyone to the nearest
# bootstrap centroid, compare with the full-data partition (ARI)
set.seed(2025)
idx$stability <- sapply(2:8, function(K) {
  ref <- fits[[K - 1]]$cluster
  mean(replicate(20, {
    i  <- sample(nrow(X), replace = TRUE)
    kb <- kmeans(X[i, ], centers = K, nstart = 10, iter.max = 100)
    adjustedRandIndex(nearest(X, kb$centers), ref)
  }))
})
write.csv(transform(idx, wss = round(wss, 1), silhouette = round(silhouette, 3), db = round(db, 3),
                    ch = round(ch, 1), dunn = round(dunn, 3), ari_truth = round(ari_truth, 3),
                    stability = round(stability, 3)),
          file.path(out, "kmeans_indices.csv"), row.names = FALSE)

# cross-tabulations of K = 4 and K = 5 against the true phenotypes
xt <- function(cl, method) {
  tb <- as.data.frame(table(cluster = cl, phenotype = truth))
  tb$method <- method; tb }
tabs <- rbind(xt(fits[[3]]$cluster, "kmeans_K4"), xt(fits[[4]]$cluster, "kmeans_K5"))

# true-partition WSS vs k-means K = 5 WSS (why k-means cannot find the truth)
wss_truth <- wss_of(X, as.integer(factor(truth)))

# centroid profiles (standardised units) for K = 4
prof4 <- data.frame(cluster = 1:4, size = as.vector(table(fits[[3]]$cluster)),
                    round(fits[[3]]$centers, 2))
write.csv(prof4, file.path(out, "kmeans_K4_profiles.csv"), row.names = FALSE)

# within-phenotype spread of each standardised feature vs overall spread (= 1)
spread <- round(sapply(vars, function(v) mean(tapply(X[, v], truth, sd))), 3)

# ---- 2. DBSCAN and OPTICS -------------------------------------------------
# 2a. two rings plus background noise (toy data for the widget and figures)
set.seed(2026)
ring <- function(n, r, sd) { a <- runif(n, 0, 2 * pi); cbind(r * cos(a) + rnorm(n, 0, sd), r * sin(a) + rnorm(n, 0, sd)) }
R <- rbind(ring(150, 1, 0.07), ring(260, 2.2, 0.07), cbind(runif(40, -3, 3), runif(40, -3, 3)))
rings <- data.frame(x = round(R[, 1], 3), y = round(R[, 2], 3),
                    truth = rep(c("inner ring", "outer ring", "background"), c(150, 260, 40)))
rings$kmeans2 <- kmeans(R, centers = 2, nstart = 20)$cluster
db_r <- dbscan(R, eps = 0.3, minPts = 5)
rings$dbscan <- db_r$cluster
write.csv(rings, file.path(out, "rings.csv"), row.names = FALSE)
op_r <- optics(R, minPts = 5)
write.csv(data.frame(position = seq_along(op_r$order), point = op_r$order,
                     reachability = round(op_r$reachdist[op_r$order], 4),
                     truth = rings$truth[op_r$order]),
          file.path(out, "rings_optics.csv"), row.names = FALSE)

# 2b. wearable data: kNN distances (minPts = 10) and an eps grid
knn10 <- sort(kNNdist(X, k = 10))
write.csv(data.frame(rank = seq_along(knn10), knn10 = round(knn10, 4)),
          file.path(out, "wearable_knn10.csv"), row.names = FALSE)
eps_grid <- c(0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
dbg <- do.call(rbind, lapply(eps_grid, function(e) {
  d <- dbscan(X, eps = e, minPts = 10)
  data.frame(eps = e, clusters = length(setdiff(unique(d$cluster), 0)),
             noise = sum(d$cluster == 0),
             ari_truth = round(adjustedRandIndex(d$cluster, truth), 3))
}))
write.csv(dbg, file.path(out, "wearable_dbscan_eps.csv"), row.names = FALSE)
op_w <- optics(X, minPts = 10)
op_w_cl <- extractDBSCAN(op_w, eps_cl = 1.3)$cluster

# ---- 3. Gaussian mixture models ------------------------------------------
bic <- mclustBIC(X, G = 1:8, modelNames = c("EII", "VII", "EEE", "VVV"), verbose = FALSE)
bic_df <- data.frame(G = 1:8, round(unclass(bic)[, c("EII", "VII", "EEE", "VVV")], 1))
write.csv(bic_df, file.path(out, "gmm_bic.csv"), row.names = FALSE)
gmm <- Mclust(X, G = 1:8, verbose = FALSE)           # best model by BIC
gmm4 <- Mclust(X, G = 4, verbose = FALSE)
tabs <- rbind(tabs, xt(gmm$classification, "gmm_best"))
write.csv(tabs, file.path(out, "crosstabs.csv"), row.names = FALSE)
post <- gmm$z
top <- order(-gmm$uncertainty)[1:5]
amb <- data.frame(person_id = wear$person_id[top], phenotype_true = truth[top],
                  map_cluster = gmm$classification[top],
                  p_max = round(apply(post[top, ], 1, max), 3),
                  p_second = round(apply(post[top, ], 1, function(p) sort(p, TRUE)[2]), 3))
write.csv(amb, file.path(out, "gmm_most_uncertain.csv"), row.names = FALSE)

# ---- 4. Hierarchical clustering of the genomic matrix ---------------------
gen <- read.csv("data/genomic_subtypes.csv", check.names = FALSE)
G   <- scale(as.matrix(gen[, -(1:2)]))
sub <- gen$subtype
Dg  <- dist(G)
links <- c("ward.D2", "complete", "average", "single")
hcs <- lapply(links, function(m) hclust(Dg, method = m))
link_tab <- do.call(rbind, lapply(seq_along(links), function(i) data.frame(
  linkage = links[i], K = 2:5,
  ari = round(sapply(2:5, function(k) adjustedRandIndex(cutree(hcs[[i]], k), sub)), 3),
  silhouette = round(sapply(2:5, function(k) mean(silhouette(cutree(hcs[[i]], k), Dg)[, "sil_width"])), 3))))
hc_cor <- hclust(as.dist(1 - cor(t(G))), method = "average")
link_tab <- rbind(link_tab, data.frame(linkage = "correlation + average", K = 2:5,
  ari = round(sapply(2:5, function(k) adjustedRandIndex(cutree(hc_cor, k), sub)), 3),
  silhouette = NA))
write.csv(link_tab, file.path(out, "genomic_linkage.csv"), row.names = FALSE)
hw <- hcs[[1]]
dendro <- list(merge = unname(hw$merge), height = round(hw$height, 3), order = hw$order,
               labels = gen$patient_id, subtype = sub)
writeLines(jsonlite::toJSON(dendro, auto_unbox = TRUE), file.path(out, "genomic_dendrogram.json"))

# ---- 5. Back to HomeBeat: clusters of a continuum, and partial records ----
hb  <- read.csv("data/homebeat.csv")
hbm <- read.csv("data/homebeat_missing.csv")
sig <- c("resting_hr", "hrv", "skin_temp", "activity_index")
mu <- colMeans(hb[, sig]); s <- apply(hb[, sig], 2, sd)
H  <- scale(hb[, sig], center = mu, scale = s)
Dh <- dist(H)
set.seed(2025)
hb_fits <- lapply(2:6, function(K) kmeans(H, centers = K, nstart = 20))
hb_sil <- data.frame(K = 2:6, silhouette = round(sapply(hb_fits, function(f)
  mean(silhouette(f$cluster, Dh)[, "sil_width"])), 3))
write.csv(hb_sil, file.path(out, "homebeat_silhouette.csv"), row.names = FALSE)
k3 <- hb_fits[[2]]
ordk <- order(k3$centers[, "resting_hr"])            # label clusters by heart rate
cl3 <- match(k3$cluster, ordk)
hb_prof <- do.call(rbind, lapply(1:3, function(k) data.frame(
  cluster = k, n = sum(cl3 == k),
  resting_hr = round(mean(hb$resting_hr[cl3 == k]), 1), hrv = round(mean(hb$hrv[cl3 == k]), 1),
  skin_temp = round(mean(hb$skin_temp[cl3 == k]), 2), activity_index = round(mean(hb$activity_index[cl3 == k]), 1),
  age = round(mean(hb$age[cl3 == k]), 1), event_rate = round(mean(hb$event_occurred[cl3 == k]), 3))))
write.csv(hb_prof, file.path(out, "homebeat_k3_profiles.csv"), row.names = FALSE)
# how well does age alone separate events, compared with the clusters?
age3 <- cut(hb$age, quantile(hb$age, c(0, 1/3, 2/3, 1)), include.lowest = TRUE, labels = 1:3)
hb_age <- data.frame(tertile = 1:3, event_rate = round(as.vector(tapply(hb$event_occurred, age3, mean)), 3))
write.csv(hb_age, file.path(out, "homebeat_age_tertiles.csv"), row.names = FALSE)
# participants with missing watch signals: nearest centroid on the signals that arrived
C3 <- k3$centers[ordk, ]
Hm <- scale(hbm[, sig], center = mu, scale = s)
inc <- which(!complete.cases(Hm))
part <- sapply(inc, function(i) { ok <- !is.na(Hm[i, ])
  which.min(colSums((t(C3[, ok, drop = FALSE]) - Hm[i, ok])^2)) })
partial <- list(n_incomplete = length(inc), agreement = round(mean(part == cl3[inc]), 3),
                n_signals_missing_mean = round(mean(rowSums(is.na(Hm[inc, , drop = FALSE]))), 2))
# soft assignment on a continuum: a three-component GMM on the same signals
hb_gmm <- Mclust(H, G = 3, verbose = FALSE)
hb_pmax <- apply(hb_gmm$z, 1, max)
hb_soft <- list(model = hb_gmm$modelName, share_pmax_below_08 = round(mean(hb_pmax < 0.8), 3),
                share_pmax_below_06 = round(mean(hb_pmax < 0.6), 3))

# ---- 6. Summary numbers used in the text ---------------------------------
summ <- list(
  wss_truth = round(wss_truth, 1),
  wss_k5 = round(fits[[4]]$tot.withinss, 1),
  spread = as.list(spread),
  gmm_model = gmm$modelName, gmm_G = gmm$G,
  gmm_ari = round(adjustedRandIndex(gmm$classification, truth), 3),
  gmm4_model = gmm4$modelName, gmm4_ari = round(adjustedRandIndex(gmm4$classification, truth), 3),
  gmm_n_uncertain = sum(gmm$uncertainty > 0.1),
  optics_wear_clusters = length(setdiff(unique(op_w_cl), 0)), optics_wear_noise = sum(op_w_cl == 0),
  rings_dbscan_clusters = length(setdiff(unique(db_r$cluster), 0)), rings_dbscan_noise = sum(db_r$cluster == 0),
  rings_kmeans_ari = round(adjustedRandIndex(rings$kmeans2[rings$truth != "background"], rings$truth[rings$truth != "background"]), 3),
  ward_heights_top = round(rev(tail(hw$height, 4)), 1),
  homebeat_partial = partial, homebeat_soft = hb_soft,
  wear_soft_share_pmax_below_08 = round(mean(apply(post, 1, max) < 0.8), 3),
  homebeat_event_rate = round(mean(hb$event_occurred), 3))
writeLines(jsonlite::toJSON(summ, auto_unbox = TRUE, pretty = TRUE), file.path(out, "summary.json"))
writeLines(c(R.version.string, paste("cluster", packageVersion("cluster")),
             paste("dbscan", packageVersion("dbscan")), paste("mclust", packageVersion("mclust"))),
           file.path(out, "versions.txt"))
message("written to ", out)
