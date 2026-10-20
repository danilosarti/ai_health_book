# =====================================================================
# Chapter 3 — precomputed results (light: base R only, runs in seconds)
# Usage (from the book folder): Rscript scripts/precompute/ch03_workflow.R
# Writes data/precomputed/ch03/:
#   knn_curve.csv     k, training MAE, 5-fold CV MAE (+ SE), test MAE (Ward 7, 2 features)
#   summary.json      baselines, chosen k, nested CV, scaling, leave-one-site-out,
#                     ONCO-1 leakage results, "wrong way" feature selection
# =====================================================================
set.seed(303)
dir.create("data/precomputed/ch03", recursive = TRUE, showWarnings = FALSE)

mae  <- function(y, yhat) mean(abs(y - yhat))
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

# ---- Ward 7 with the book-wide split --------------------------------
icu <- read.csv("data/icu_aki.csv")
spl <- read.csv("data/splits/ward7_split.csv")
icu <- merge(icu, spl, by = "patient_id", sort = FALSE)
icu <- icu[order(icu$patient_id), ]
train <- icu[icu$set == "train", ]
test  <- icu[icu$set == "test", ]
feat  <- c("sofa", "baseline_creat")
yname <- "delta_creatinine_48h"

# kNN regression: standardise with the TRAINING rows only, Euclidean distance,
# average the outcomes of the k nearest training patients (stable tie-breaking).
knn_all_k <- function(Xtr, ytr, Xte, kmax, scale = TRUE) {
  Xtr <- as.matrix(Xtr); Xte <- as.matrix(Xte)
  if (scale) {
    mu <- colMeans(Xtr); s <- apply(Xtr, 2, sd)
    Xtr <- sweep(sweep(Xtr, 2, mu), 2, s, "/")
    Xte <- sweep(sweep(Xte, 2, mu), 2, s, "/")
  }
  # predictions for every k = 1..kmax at once (rows = test patients, cols = k)
  t(apply(Xte, 1, function(x) {
    d <- sqrt(colSums((t(Xtr) - x)^2))
    o <- order(d)[seq_len(kmax)]
    cumsum(ytr[o]) / seq_len(kmax)
  }))
}
kmax <- 150
ks <- seq_len(kmax)

# training (resubstitution) error: each patient is its own nearest neighbour
P_tr <- knn_all_k(train[feat], train[[yname]], train[feat], kmax)
train_mae <- colMeans(abs(P_tr - train[[yname]]))

# 5-fold CV on the training set, using the fixed folds
cv_by_fold <- sapply(1:5, function(v) {
  fit <- train[train$fold != v, ]; val <- train[train$fold == v, ]
  P <- knn_all_k(fit[feat], fit[[yname]], val[feat], kmax)
  colMeans(abs(P - val[[yname]]))
})
cv_mae <- rowMeans(cv_by_fold)
cv_se  <- apply(cv_by_fold, 1, sd) / sqrt(5)

P_te <- knn_all_k(train[feat], train[[yname]], test[feat], kmax)
test_mae <- colMeans(abs(P_te - test[[yname]]))

curve <- data.frame(k = ks, train_mae = round(train_mae, 4), cv_mae = round(cv_mae, 4),
                    cv_se = round(cv_se, 4), test_mae = round(test_mae, 4))
write.csv(curve, "data/precomputed/ch03/knn_curve.csv", row.names = FALSE)
k_best <- ks[which.min(cv_mae)]

# ---- baselines and a straight line (test set and 5-fold CV) ----------
cv_of <- function(fun) mean(sapply(1:5, function(v) {
  fit <- train[train$fold != v, ]; val <- train[train$fold == v, ]
  mae(val[[yname]], fun(fit, val))
}))
f_mean   <- function(fit, new) rep(mean(fit[[yname]]), nrow(new))
f_median <- function(fit, new) rep(median(fit[[yname]]), nrow(new))
f_sofa   <- function(fit, new) predict(lm(delta_creatinine_48h ~ sofa, data = fit), newdata = new)
f_line2  <- function(fit, new) predict(lm(delta_creatinine_48h ~ sofa + baseline_creat, data = fit), newdata = new)
fs <- list(f_mean, f_median, f_sofa, f_line2)
baselines <- data.frame(
  model = c("mean", "median", "sofa_line", "line_2features"),
  cv_mae = sapply(fs, cv_of),
  test_mae = sapply(fs, function(f) mae(test[[yname]], f(train, test)))
)

# ---- nested CV: choose k inside each outer training part ------------
nested <- lapply(1:5, function(v) {
  outer_fit <- train[train$fold != v, ]; outer_val <- train[train$fold == v, ]
  inner_fold <- sample(rep(1:5, length.out = nrow(outer_fit)))
  inner <- rowMeans(sapply(1:5, function(u) {
    fit <- outer_fit[inner_fold != u, ]; val <- outer_fit[inner_fold == u, ]
    P <- knn_all_k(fit[feat], fit[[yname]], val[feat], kmax)
    colMeans(abs(P - val[[yname]]))
  }))
  kv <- ks[which.min(inner)]
  P <- knn_all_k(outer_fit[feat], outer_fit[[yname]], outer_val[feat], kmax)
  c(fold = v, k = kv, mae = mae(outer_val[[yname]], P[, kv]))
})
nested <- as.data.frame(do.call(rbind, nested))

# ---- why scaling matters: record baseline creatinine in umol/L instead of mg/dL --
# (1 mg/dL = 88.4 umol/L). Standardised kNN gives identical results in either unit;
# unstandardised kNN is then dominated by creatinine, whose numbers are now large.
cv_k <- function(data, scale) mean(sapply(1:5, function(v) {
  fit <- data[data$fold != v, ]; val <- data[data$fold == v, ]
  P <- knn_all_k(fit[feat], fit[[yname]], val[feat], k_best, scale = scale)
  mae(val[[yname]], P[, k_best])
}))
train_umol <- train; train_umol$baseline_creat <- train_umol$baseline_creat * 88.4
scaling <- list(scaled_mgdl = cv_k(train, TRUE), scaled_umol = cv_k(train_umol, TRUE),
                unscaled_mgdl = cv_k(train, FALSE), unscaled_umol = cv_k(train_umol, FALSE))

# ---- leave-one-site-out ("internal-external") validation ------------
loso <- do.call(rbind, lapply(sort(unique(icu$site)), function(s) {
  fit <- icu[icu$site != s, ]; val <- icu[icu$site == s, ]
  P <- knn_all_k(fit[feat], fit[[yname]], val[feat], k_best)
  data.frame(site = s, n = nrow(val),
             knn_mae = mae(val[[yname]], P[, k_best]),
             mean_mae = mae(val[[yname]], mean(fit[[yname]])),
             bias = mean(val[[yname]] - P[, k_best]))
}))

# ---- tonight's patients: kNN predictions at k_best -------------------
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
P_tn <- knn_all_k(train[feat], train[[yname]], tonight[feat], kmax)
tonight_pred <- data.frame(name = tonight$name, k1 = P_tn[, 1], kbest = P_tn[, k_best],
                           k150 = P_tn[, kmax])
above_by_k <- lapply(seq_len(nrow(tonight)), function(i) which(P_tn[i, ] > 0.3))  # k values with prediction > 0.3
names(above_by_k) <- c("aoife", "tomas", "priya")

# ---- ONCO-1: leakage ---------------------------------------------------
onco <- read.csv("data/onco1.csv")
os <- read.csv("data/splits/onco1_split.csv")
onco <- merge(onco, os, by = "patient_id", sort = FALSE)
otr <- onco[onco$set == "train", ]; ote <- onco[onco$set == "test", ]
genes <- grep("^gene_", names(onco), value = TRUE)
leaky <- c("gene_08", "gene_14", "gene_19")
clin  <- c("treatment", "dose_intensity", "patient_age", "tumor_grade", "performance_score")
r2 <- function(y, yhat, ybar) 1 - sum((y - yhat)^2) / sum((y - ybar)^2)
fit_eval <- function(vars) {
  m <- lm(reformulate(vars, "response_percent"), data = otr)
  p <- predict(m, newdata = ote)
  c(rmse = rmse(ote$response_percent, p),
    r2 = r2(ote$response_percent, p, mean(otr$response_percent)))
}
cors <- sapply(genes, function(g) cor(otr[[g]], otr$response_percent))
onco_res <- list(
  baseline_rmse = rmse(ote$response_percent, mean(otr$response_percent)),
  with_leaky   = as.list(fit_eval(c(clin, genes))),
  without_leaky = as.list(fit_eval(c(clin, setdiff(genes, leaky)))),
  clinical_only = as.list(fit_eval(clin)),
  top_cor = as.list(round(sort(abs(cors), decreasing = TRUE)[1:6], 3)),
  post_tumor = as.list(fit_eval(c(clin, "baseline_tumor_mm", "post_tumor_mm")))
)
# What happens when the leaky model meets patients whose genes 08/14/19 no longer carry
# the outcome (e.g. measured before treatment): shuffle those three columns in the test set.
m_leaky <- lm(reformulate(c(clin, genes), "response_percent"), data = otr)
ote_shuf <- ote
for (g in leaky) ote_shuf[[g]] <- sample(ote_shuf[[g]])
onco_res$leaky_deployed_rmse <- rmse(ote$response_percent, predict(m_leaky, newdata = ote_shuf))
m_honest <- lm(reformulate(c(clin, setdiff(genes, leaky)), "response_percent"), data = otr)
onco_res$honest_deployed_rmse <- rmse(ote$response_percent, predict(m_honest, newdata = ote_shuf))
onco_res$true_gene_rank <- as.list(match(c("gene_01", "gene_05", "gene_12"),
                                         names(sort(abs(cors), decreasing = TRUE))))

# ---- "wrong way" vs "right way": selecting features before CV ----------
# Pure noise: 60 patients, 2 000 random "genes", an outcome unrelated to all of them.
# Model: pick the 20 genes most correlated with the outcome, average them (signed by
# their correlation) into one "signature score", and fit a straight line on the score.
n_ns <- 60; p_ns <- 2000; n_keep <- 20
Xn <- matrix(rnorm(n_ns * p_ns), n_ns, p_ns)
yn <- rnorm(n_ns)
fold_ns <- sample(rep(1:5, length.out = n_ns))
signature <- function(X, y) {
  r <- drop(cor(X, y)); sel <- order(abs(r), decreasing = TRUE)[1:n_keep]
  list(sel = sel, sgn = sign(r[sel]))
}
score <- function(X, sig) drop(X[, sig$sel, drop = FALSE] %*% sig$sgn) / n_keep
cv_r2 <- function(select_inside) {
  sig_all <- signature(Xn, yn)                     # uses ALL 60 patients (the wrong way)
  pred <- numeric(n_ns)
  for (v in 1:5) {
    tr <- fold_ns != v
    sig <- if (select_inside) signature(Xn[tr, ], yn[tr]) else sig_all
    d <- data.frame(y = yn[tr], s = score(Xn[tr, ], sig))
    m <- lm(y ~ s, data = d)
    pred[!tr] <- predict(m, newdata = data.frame(s = score(Xn[!tr, , drop = FALSE], sig)))
  }
  1 - sum((yn - pred)^2) / sum((yn - mean(yn))^2)
}
wrong_right <- list(n = n_ns, p = p_ns, kept = n_keep,
                    r2_wrong = cv_r2(FALSE), r2_right = cv_r2(TRUE))

summary_list <- list(
  ward7 = list(n_train = nrow(train), n_test = nrow(test), features = feat,
               k_best = k_best, cv_best = min(cv_mae), test_at_kbest = test_mae[k_best],
               cv_k1 = cv_mae[1], train_k1 = train_mae[1], test_k1 = test_mae[1],
               cv_k150 = cv_mae[kmax],
               baselines = baselines, nested = nested, nested_mae = mean(nested$mae),
               scaling = scaling, above_threshold_k = lapply(above_by_k, as.list), loso = loso, tonight = tonight_pred),
  onco1 = onco_res, wrong_right = wrong_right)

# jsonlite is installed with rmarkdown, so it is available wherever the book renders
jsonlite::write_json(summary_list, "data/precomputed/ch03/summary.json",
                     auto_unbox = TRUE, digits = 6, pretty = TRUE, dataframe = "rows")
str(summary_list, max.level = 2)
