# =====================================================================
# Chapter 12 - VIVID (package vivid) and BART uncertainty displays (package bartMan)
#
# What this script does (lung-cancer clinic, first-edition data set):
#   1. Fits the R random forest of the chapter's Lab (randomForest, 300 trees, seed 2026)
#      on the 171 training patients of data/splits/lung_split.csv.
#   2. vivid: vivi() importance/interaction matrix, viviHeatmap(), viviNetwork(), pdpPairs().
#      (pdpZen() needs the 'zenplots' package; it is drawn only if zenplots is installed.)
#   3. Fits BART with dbarts (80 trees and 200 burn-in iterations as in the first edition, but
#      200 instead of 600 kept posterior draws, because bartMan's extractTreeData() time grows
#      faster than linearly with the number of draws), summarises the posterior predictions of the 57 test patients, and uses
#      bartMan: extractTreeData(), viviBartMatrix() (standard and VSUP, CV), viviBartPlot(),
#      and permVimp() (inclusion proportion minus that of a null model with permuted response).
#   Note: extractTreeData() re-reads the arguments of the bart() call, so they must be literal
#   values (ntree = 80L, not ntree = n_trees).
# Outputs: data/precomputed/ch12/{lung_vivid_*.csv, lung_bart_*.csv, lung_vivid_bartman.rds,
#          vivid_bartman_versions.json} and images/ch12_{vivid_heatmap,vivid_network,
#          vivid_pdppairs,bart_vsup}.png. Afterwards run
#          python3 scripts/precompute/ch12_reencode_png.py
#          to re-encode the PNGs with PIL (longest side <= 900 px, no metadata).
#
# Usage (from the book folder):  Rscript scripts/precompute/ch12_vivid_bartman.R
# Requires: randomForest, vivid (0.2.8), dbarts, bartMan (0.1.1), ggplot2, ragg.
# Single-threaded; about two to four minutes on 2 CPUs.
# =====================================================================
suppressPackageStartupMessages({
  library(randomForest)
  library(vivid)
  library(dbarts)
  library(bartMan)
  library(ggplot2)
})
t0 <- Sys.time()
OUT <- "data/precomputed/ch12"
IMG <- "images"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
SEED <- 2026

# ---- data: the chapter's lung clinic and its shared split ------------------
lung <- read.csv("data/lung_sim_complete.csv")
lsp  <- read.csv("data/splits/lung_split.csv")
stopifnot(identical(lsp$patient_id, lung$patient_id))
lung$male <- as.integer(lung$sex == "male")
feats <- c("age", "male", "ph.ecog", "ph.karno", "wt.loss")
ltr <- lung[lsp$set == "train", c(feats, "time")]
lte <- lung[lsp$set == "test",  c(feats, "time")]
stopifnot(nrow(ltr) == 171, nrow(lte) == 57)
mae  <- function(y, p) mean(abs(y - p))
rmse <- function(y, p) sqrt(mean((y - p)^2))

save_png <- function(file, plot_expr, width = 6, height = 5, res = 150) {
  ragg::agg_png(file.path(IMG, file), width = width, height = height, units = "in", res = res,
                background = "white")
  on.exit(dev.off())
  p <- plot_expr
  if (inherits(p, "ggplot")) print(p)
  invisible(NULL)
}

# ---- 1. random forest (the Lab's forest) ------------------------------------
set.seed(SEED)
rf <- randomForest(time ~ age + male + ph.ecog + ph.karno + wt.loss, data = ltr, ntree = 300)
rf_mae <- mae(lte$time, predict(rf, lte)); rf_rmse <- rmse(lte$time, predict(rf, lte))

# ---- 2. vivid -------------------------------------------------------------------
set.seed(SEED)
vi_rf <- vivi(data = ltr, fit = rf, response = "time", gridSize = 10,
              importanceType = "agnostic", nmax = 500, numPerm = 4, reorder = TRUE)
M <- matrix(as.numeric(vi_rf), nrow(vi_rf), dimnames = dimnames(vi_rf))
write.csv(data.frame(var = rownames(M), M, check.names = FALSE),
          file.path(OUT, "lung_vivid_matrix.csv"), row.names = FALSE)

# normalised H as well (same forest, same grid), for comparison with the unnormalised default
set.seed(SEED)
vi_rf_n <- vivi(data = ltr, fit = rf, response = "time", gridSize = 10,
                importanceType = "agnostic", nmax = 500, numPerm = 4, reorder = FALSE,
                normalized = TRUE)
Mn <- matrix(as.numeric(vi_rf_n), nrow(vi_rf_n), dimnames = dimnames(vi_rf_n))
write.csv(data.frame(var = rownames(Mn), Mn, check.names = FALSE),
          file.path(OUT, "lung_vivid_matrix_normalised.csv"), row.names = FALSE)

save_png("ch12_vivid_heatmap.png", viviHeatmap(mat = vi_rf, angle = 45), width = 6, height = 5)
save_png("ch12_vivid_network.png", viviNetwork(mat = vi_rf), width = 6, height = 5)

set.seed(SEED)
save_png("ch12_vivid_pdppairs.png",
         pdpPairs(data = ltr, fit = rf, response = "time",
                  vars = c("ph.ecog", "ph.karno", "wt.loss"), gridSize = 10, nIce = 50),
         width = 6, height = 6)
zen_done <- FALSE
if (requireNamespace("zenplots", quietly = TRUE)) {
  set.seed(SEED)
  save_png("ch12_vivid_pdpzen.png",
           pdpZen(data = ltr, fit = rf, response = "time", gridSize = 10, nmax = 500),
           width = 7, height = 5)
  zen_done <- TRUE
}

# ---- 3. BART (dbarts) and bartMan -------------------------------------------------
set.seed(SEED)
bart_fit <- dbarts::bart(x.train = as.matrix(ltr[feats]), y.train = ltr$time,
                         x.test = as.matrix(lte[feats]), ntree = 80L, nskip = 200L,
                         ndpost = 200L, keeptrees = TRUE, verbose = FALSE, seed = SEED)
post_mean <- colMeans(bart_fit$yhat.test)
bart_mae <- mae(lte$time, post_mean); bart_rmse <- rmse(lte$time, post_mean)
pred_tab <- data.frame(
  patient_id = lung$patient_id[lsp$set == "test"], observed = lte$time,
  post_mean = post_mean,
  q025 = apply(bart_fit$yhat.test, 2, quantile, 0.025),
  q975 = apply(bart_fit$yhat.test, 2, quantile, 0.975))
pred_tab$width95 <- pred_tab$q975 - pred_tab$q025
pred_tab$covered <- pred_tab$observed >= pred_tab$q025 & pred_tab$observed <= pred_tab$q975
# yhat.test holds draws of the average prediction f(x); a prediction interval for one patient's
# outcome adds the residual noise (sigma, one value per draw)
set.seed(SEED)
ypred <- bart_fit$yhat.test + matrix(rnorm(length(bart_fit$yhat.test)), nrow(bart_fit$yhat.test)) *
  bart_fit$sigma
pred_tab$p025 <- apply(ypred, 2, quantile, 0.025)
pred_tab$p975 <- apply(ypred, 2, quantile, 0.975)
pred_tab$covered_pred <- pred_tab$observed >= pred_tab$p025 & pred_tab$observed <= pred_tab$p975
sigma_mean <- mean(bart_fit$sigma)
write.csv(pred_tab, file.path(OUT, "lung_bart_test_predictions.csv"), row.names = FALSE)

# variable usage (split counts) per posterior draw, straight from dbarts
vc <- bart_fit$varcount; colnames(vc) <- feats
write.csv(data.frame(feature = feats, mean_splits = colMeans(vc), sd_splits = apply(vc, 2, sd)),
          file.path(OUT, "lung_bart_varcount.csv"), row.names = FALSE)

bart_df <- data.frame(ltr[feats], y = ltr$time)
sink(nullfile())                                 # silence the progress bars
trees_data <- extractTreeData(model = bart_fit, data = bart_df)
sink()
stdMat  <- viviBartMatrix(trees = trees_data, type = "standard", metric = "propMean")
vsupMat <- viviBartMatrix(trees = trees_data, type = "vsup", metric = "propMean",
                          metricError = "CV")
# long table: value (posterior mean inclusion proportion) and CV across posterior draws
vals <- vsupMat[[1]]; uncs <- vsupMat[[2]]
long <- do.call(rbind, lapply(rownames(vals), function(a) do.call(rbind, lapply(colnames(vals), function(b)
  data.frame(var1 = a, var2 = b, value = vals[a, b], cv = uncs[a, b])))))
write.csv(long, file.path(OUT, "lung_bart_vsup.csv"), row.names = FALSE)

save_png("ch12_bart_vsup.png",
         viviBartPlot(vsupMat, max_desat = 1, pow_desat = 0.6, max_light = 0.6, pow_light = 1,
                      label = "CV"), width = 7, height = 5)

# permutation-based check of the inclusion proportions (null model: 20 trees, permuted response)
set.seed(SEED)
pv <- permVimp(model = bart_fit, data = data.frame(ltr[feats], time = ltr$time), response = "time",
               numTreesPerm = 20)
pvd <- as.data.frame(pv$data)[, c("variable", "mean", "low", "high", "sds")]
names(pvd) <- c("feature", "ip_minus_null", "low", "high", "sd_null")
write.csv(pvd, file.path(OUT, "lung_bart_permvimp.csv"), row.names = FALSE)

saveRDS(list(vivi_rf = vi_rf, vivi_rf_normalised = vi_rf_n, stdMat = stdMat, vsupMat = vsupMat,
             bart_varcount = vc, pred = pred_tab),
        file.path(OUT, "lung_vivid_bartman.rds"))

summ <- data.frame(model = c("Random forest (randomForest, 300 trees)", "BART (dbarts, 80 trees)"),
                   test_MAE = c(rf_mae, bart_mae), test_RMSE = c(rf_rmse, bart_rmse),
                   sigma_mean = c(NA, sigma_mean))
write.csv(summ, file.path(OUT, "lung_vivid_bartman_models.csv"), row.names = FALSE)

v <- function(p) as.character(packageVersion(p))
writeLines(sprintf('{\n "R": "%s",\n "randomForest": "%s",\n "vivid": "%s",\n "dbarts": "%s",\n "bartMan": "%s",\n "ggplot2": "%s",\n "seed": %d,\n "pdpZen_drawn": %s,\n "seconds": %.0f\n}',
                   getRversion(), v("randomForest"), v("vivid"), v("dbarts"), v("bartMan"),
                   v("ggplot2"), SEED, tolower(zen_done),
                   as.numeric(difftime(Sys.time(), t0, units = "secs"))),
           file.path(OUT, "vivid_bartman_versions.json"))
print(summ); print(round(M, 2)); print(round(Mn, 2))
print(round(vals, 3)); print(round(uncs, 2)); print(pvd)
cat("Done in", round(as.numeric(difftime(Sys.time(), t0, units = "secs"))), "s\n")
