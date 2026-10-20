# =====================================================================
# MODEL LEDGER — one source of truth for every model result of the running examples
# (Ward 7, ONCO-1, HomeBeat). Specification: LEDGER_SPEC (planning) and
# data/precomputed/ledger/ledger_README.md (definitions of every column).
#
# Usage (from the book folder):
#   LEDGER_PYTHON=~/venv-bart/bin/python Rscript scripts/precompute/ledger.R
#   Rscript scripts/precompute/ledger.R --skip-py     # reuse parts/py_preds.csv (no Python run)
# The Python part (scripts/precompute/ledger.py: scikit-learn, XGBoost, PyTorch, pymc-bart) is
# started in the background first, so that it runs on the second CPU while R fits its models.
# Total: about 3 minutes on 2 CPUs (R part ~1 min, Python part ~2.5 min in parallel).
#
# What this script does
#   1. fits the R models (lm, glm, glmnet, rpart, kNN, rstanarm, mice) on the shared splits
#      data/splits/{ward7,onco1,homebeat}_split.csv, re-using the chapters' own code
#      (ch03_workflow.R, ch04_glmnet.R, ch05_precompute.R, ch06 chunk `ward7-prune`,
#      ch08_stan.R, ch09 Lab) so that values that should not move do not move;
#   2. reads the Python predictions (parts/py_preds.csv);
#   3. computes EVERY metric in one place from the predictions:
#        regression     test MAE, RMSE, R² (against the TRAINING mean), 5-fold CV MAE
#        classification test AUC, Brier, calibration slope, calibration-in-the-large, CV log-loss
#      CV = out-of-fold predictions on the 5 shared folds, with each model's tuning value held at
#      the one chosen on the full training set (for tuned models this is slightly optimistic);
#   4. writes data/precomputed/ledger/*.csv.
# Deterministic: every random step has a fixed seed; glmnet and rpart use the fixed folds.
# =====================================================================
suppressPackageStartupMessages({
  library(glmnet); library(rpart); library(rstanarm); library(mice)
})
options(mc.cores = 1)
# the labels contain λ, α, Δ, ≥: make sure R writes them as UTF-8 even in a "C" session
if (!l10n_info()[["UTF-8"]]) invisible(Sys.setlocale("LC_CTYPE", "C.UTF-8"))
SEED <- 2026
OUT <- "data/precomputed/ledger"; PARTS <- file.path(OUT, "parts")
dir.create(PARTS, recursive = TRUE, showWarnings = FALSE)
args <- commandArgs(trailingOnly = TRUE)
t0 <- Sys.time()

# ---- 0. start the Python part in the background ----------------------------------------
py_exit <- file.path(PARTS, "py_exit")
if (!"--skip-py" %in% args) {
  if (file.exists(py_exit)) file.remove(py_exit)
  py <- Sys.getenv("LEDGER_PYTHON", "python3")
  system(sprintf("(%s scripts/precompute/ledger.py > %s 2>&1; echo $? > %s) &",
                 py, file.path(PARTS, "py_log.txt"), py_exit))   # the trailing & detaches it
  cat("Python part started in the background (log: parts/py_log.txt)\n")
}

# ---- helpers ---------------------------------------------------------------------------
P <- list()                                         # predictions, long format
put <- function(example, model_id, kind, id, pred)
  P[[length(P) + 1]] <<- data.frame(example = example, model_id = model_id, kind = kind,
                                    id = as.character(id), pred = as.numeric(pred))
TUNE <- list()
note <- function(example, model_id, tuning, software)
  TUNE[[length(TUNE) + 1]] <<- data.frame(example = example, model_id = model_id,
                                          tuning = tuning, software = software)
ver <- function(p) paste(p, as.character(packageVersion(p)))
Rv <- paste("R", getRversion())
std_fit <- function(X) list(mu = colMeans(X), s = apply(X, 2, sd))
std <- function(X, st) scale(X, st$mu, st$s)

# =========================================================================================
# 1. WARD 7
# =========================================================================================
cat("Ward 7 ...\n")
icu <- read.csv("data/icu_aki.csv")
sp7 <- read.csv("data/splits/ward7_split.csv")
stopifnot(identical(sp7$patient_id, icu$patient_id))
icu$set <- sp7$set; icu$fold <- sp7$fold
icu$site <- factor(icu$site)
icu$aki <- as.integer(icu$delta_creatinine_48h >= 0.3)            # KDIGO criterion
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
tonight$site <- factor(tonight$site, levels = levels(icu$site))
tn_names <- tonight$name
train <- icu[icu$set == "train", ]; test <- icu[icu$set == "test", ]
folds <- train$fold
y_tr <- train$delta_creatinine_48h
f10 <- delta_creatinine_48h ~ site + age + sofa + lactate + map_min + baseline_creat +
  protocol_adherence + bun + crp + albumin
P5 <- c("sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence")

# generic: refit a model on 4 folds, predict the 5th
oof <- function(fit_pred, data = train, fold = folds) {
  out <- numeric(nrow(data))
  for (v in 1:5) out[fold == v] <- fit_pred(data[fold != v, ], data[fold == v, ])
  out
}
w7 <- function(ex, mid, fit_pred, tuning, software, do_oof = TRUE) {
  put(ex, mid, "test", test$patient_id, fit_pred(train, test))
  put(ex, mid, "tonight", tn_names, fit_pred(train, tonight))
  if (do_oof) put(ex, mid, "oof", train$patient_id, oof(fit_pred))
  note(ex, mid, tuning, software)
}

# ---- 1a. regression ------------------------------------------------------------------
w7("ward7_reg", "mean", function(a, b) rep(mean(a$delta_creatinine_48h), nrow(b)), "—", Rv)
w7("ward7_reg", "sofa_line", function(a, b) predict(lm(delta_creatinine_48h ~ sofa, a), b), "—", Rv)
w7("ward7_reg", "p2_line", function(a, b) predict(lm(delta_creatinine_48h ~ sofa + baseline_creat, a), b), "—", Rv)
w7("ward7_reg", "ols", function(a, b) predict(lm(f10, a), b), "—", Rv)

# kNN exactly as in ch03_workflow.R (standardised on the training rows, stable ties)
knn_all_k <- function(Xtr, ytr, Xte, kmax) {
  Xtr <- as.matrix(Xtr); Xte <- as.matrix(Xte)
  mu <- colMeans(Xtr); s <- apply(Xtr, 2, sd)
  Xtr <- sweep(sweep(Xtr, 2, mu), 2, s, "/"); Xte <- sweep(sweep(Xte, 2, mu), 2, s, "/")
  t(apply(Xte, 1, function(x) {
    d <- sqrt(colSums((t(Xtr) - x)^2)); o <- order(d)[seq_len(kmax)]
    cumsum(ytr[o]) / seq_len(kmax)
  }))
}
mm10 <- function(d) model.matrix(~ site + age + sofa + lactate + map_min + baseline_creat +
                                   protocol_adherence + bun + crp + albumin, d)[, -1]
knn_k <- list()
for (ks in list(list(id = "knn_p2", X = function(d) d[, c("sofa", "baseline_creat")]),
                list(id = "knn_p10", X = mm10))) {
  cvm <- rowMeans(sapply(1:5, function(v) {
    a <- train[folds != v, ]; b <- train[folds == v, ]
    colMeans(abs(knn_all_k(ks$X(a), a$delta_creatinine_48h, ks$X(b), 150) - b$delta_creatinine_48h))
  }))
  k <- which.min(cvm); knn_k[[ks$id]] <- k
  w7("ward7_reg", ks$id, local({ k <- k; X <- ks$X
       function(a, b) knn_all_k(X(a), a$delta_creatinine_48h, X(b), k)[, k] }),
     sprintf("k = %d (5-fold CV over k = 1-150, MAE); standardised on training rows", k), Rv)
}

# penalised regressions (procedure of ch04_glmnet.R: standardised on training rows, shared folds,
# alpha grid 0, 0.1, ..., 1, elastic net = alpha with the lowest CV MSE)
Xtr10 <- mm10(train); st10 <- std_fit(Xtr10)
Ztr <- std(Xtr10, st10); Zte <- std(mm10(test), st10); Ztn <- std(mm10(tonight), st10)
alphas <- seq(0, 1, by = 0.1)
cvs <- lapply(alphas, function(a) cv.glmnet(Ztr, y_tr, alpha = a, foldid = folds, standardize = FALSE,
                                            lambda.min.ratio = if (a == 0) 1e-6 else 1e-4))
best_a <- which.min(sapply(cvs, function(cv) min(cv$cvm)))
glm_oof <- function(cv, a, Z, y, rule, family = "gaussian", fold = folds, ...) {
  out <- numeric(nrow(Z))
  for (v in 1:5) {
    g <- glmnet(Z[fold != v, ], y[fold != v], alpha = a, lambda = cv$lambda, family = family, ...)
    out[fold == v] <- predict(g, Z[fold == v, , drop = FALSE], s = cv[[rule]], type = "response")
  }
  out
}
for (m in list(list("ridge", 1), list("lasso", 11), list("enet", best_a))) for (rule in c("lambda.min", "lambda.1se")) {
  cv <- cvs[[m[[2]]]]; a <- alphas[m[[2]]]
  mid <- paste0(m[[1]], if (rule == "lambda.min") "_min" else "_1se")
  put("ward7_reg", mid, "test", test$patient_id, predict(cv, Zte, s = rule))
  put("ward7_reg", mid, "tonight", tn_names, predict(cv, Ztn, s = rule))
  put("ward7_reg", mid, "oof", train$patient_id, glm_oof(cv, a, Ztr, y_tr, rule, standardize = FALSE))
  b <- as.matrix(coef(cv, s = rule))[-1, 1]
  note("ward7_reg", mid, sprintf("alpha = %.1f, lambda = %.4g (%s, 5-fold CV MSE), %d of %d coefficients non-zero",
                                 a, cv[[rule]], if (rule == "lambda.min") "λ min" else "λ 1-SE", sum(b != 0), length(b)),
       ver("glmnet"))
}
enet_alpha_w7 <- alphas[best_a]

# trees: Chapter 6 chunk `ward7-prune` (rpart, xval = the shared folds)
big <- rpart(f10, data = train, control = rpart.control(cp = 0.001, minsplit = 20, xval = folds))
cpt <- big$cptable; bi <- which.min(cpt[, "xerror"])
cp_1se <- cpt[cpt[, "xerror"] <= cpt[bi, "xerror"] + cpt[bi, "xstd"], "CP"][1]
tree_1se <- prune(big, cp = cp_1se)
nleaf <- function(t) sum(t$frame$var == "<leaf>")
w7("ward7_reg", "tree_1se", function(a, b) {
  t <- if (identical(a, train)) tree_1se else rpart(f10, a, control = rpart.control(cp = cp_1se, minsplit = 20, xval = 0))
  predict(t, b) }, sprintf("1-SE rule on rpart's CV (shared folds): cp = %.4g, %d leaves", cp_1se, nleaf(tree_1se)), ver("rpart"))
w7("ward7_reg", "tree_large", function(a, b) {
  t <- if (identical(a, train)) big else rpart(f10, a, control = rpart.control(cp = 0.001, minsplit = 20, xval = 0))
  predict(t, b) }, sprintf("cp = 0.001, minsplit = 20, no pruning: %d leaves", nleaf(big)), ver("rpart"))

# Bayesian linear regression (rstanarm defaults: weakly informative, autoscaled priors)
stan_lin <- function(a, b) {
  f <- stan_glm(f10, data = a, family = gaussian(), chains = 2, iter = 2000, seed = SEED, refresh = 0)
  colMeans(posterior_epred(f, newdata = b))
}
w7("ward7_reg", "bayes_lin", stan_lin,
   "rstanarm default priors (normal, autoscaled); 2 chains x 2 000 iterations; posterior mean of E(y | x)", ver("rstanarm"))

# Chapter 9: the same OLS after imputation (icu_aki_missing.csv), site included (P10)
icm <- read.csv("data/icu_aki_missing.csv")
stopifnot(identical(icm$patient_id, icu$patient_id))
icm$site <- factor(icm$site, levels = levels(icu$site))
icm$set <- icu$set; icm$fold <- icu$fold
gappy <- names(which(colSums(is.na(icm[, all.vars(f10)[-1]])) > 0))
mtr <- icm[icm$set == "train", ]; mte <- icm[icm$set == "test", ]
f_ind <- update(f10, as.formula(paste(". ~ . +", paste0(gappy, "_missing", collapse = " + "))))
imp_mean <- function(a, b) {
  fill <- sapply(a[gappy], mean, na.rm = TRUE)                       # learned on training rows
  prep <- function(d) { for (v in gappy) { d[[paste0(v, "_missing")]] <- as.integer(is.na(d[[v]]))
                                           d[[v]][is.na(d[[v]])] <- fill[v] }; d }
  predict(lm(f_ind, prep(a)), prep(b))
}
put("ward7_reg", "ols_imp_mean", "test", mte$patient_id, imp_mean(mtr, mte))
put("ward7_reg", "ols_imp_mean", "tonight", tn_names, imp_mean(mtr, tonight))
put("ward7_reg", "ols_imp_mean", "oof", mtr$patient_id, oof(imp_mean, mtr, mtr$fold))
note("ward7_reg", "ols_imp_mean", sprintf("training-mean fill + missing indicators for %s", paste(gappy, collapse = ", ")), Rv)

# multiple imputation: mice (pmm) with the imputation model learned on the training rows only
# (`ignore` = rows to be predicted, whose outcome is also hidden), m = 20, then the average of the
# 20 OLS predictions.
MI_M <- 20
mi_pred <- function(a, b, extra = NULL) {        # extra: complete rows (tonight), no imputation needed
  b2 <- b; b2$delta_creatinine_48h <- NA
  d <- rbind(a[, all.vars(f10)], b2[, all.vars(f10)])
  ign <- c(rep(FALSE, nrow(a)), rep(TRUE, nrow(b)))
  imp <- mice(d, m = MI_M, method = "pmm", maxit = 10, seed = SEED, ignore = ign, printFlag = FALSE)
  fits <- lapply(seq_len(MI_M), function(i) { ci <- complete(imp, i); list(lm(f10, ci[!ign, ]), ci[ign, ]) })
  p <- rowMeans(sapply(fits, function(f) predict(f[[1]], f[[2]])))
  if (is.null(extra)) p else list(p, rowMeans(sapply(fits, function(f) predict(f[[1]], extra))))
}
mi_main <- mi_pred(mtr, mte, tonight)
put("ward7_reg", "ols_mi", "test", mte$patient_id, mi_main[[1]])
put("ward7_reg", "ols_mi", "tonight", tn_names, mi_main[[2]])
put("ward7_reg", "ols_mi", "oof", mtr$patient_id, oof(mi_pred, mtr, mtr$fold))
note("ward7_reg", "ols_mi", sprintf("mice pmm, m = %d, maxit 10, seed %d; imputation model fitted on training rows (ignore = rows to predict, their outcome hidden); prediction = mean over the %d imputations", MI_M, SEED, MI_M), ver("mice"))

# ---- 1b. AKI classification ----------------------------------------------------------
fP5 <- reformulate(P5, "aki")
w7("ward7_aki", "prevalence", function(a, b) rep(mean(a$aki), nrow(b)), "—", Rv)
w7("ward7_aki", "logit_sofa", function(a, b) predict(glm(aki ~ sofa, binomial, a), b, type = "response"), "—", Rv)
w7("ward7_aki", "logit_p5", function(a, b) predict(glm(fP5, binomial, a), b, type = "response"), "unpenalised (Chapter 5)", Rv)
st5 <- std_fit(as.matrix(train[P5]))
Z5 <- function(d) std(as.matrix(d[P5]), st5)
for (m in list(list("ridge_logit", 0), list("lasso_logit", 1))) {
  cv <- cv.glmnet(Z5(train), train$aki, family = "binomial", alpha = m[[2]], foldid = folds, standardize = FALSE)
  put("ward7_aki", m[[1]], "test", test$patient_id, predict(cv, Z5(test), s = "lambda.min", type = "response"))
  put("ward7_aki", m[[1]], "tonight", tn_names, predict(cv, Z5(tonight), s = "lambda.min", type = "response"))
  put("ward7_aki", m[[1]], "oof", train$patient_id,
      glm_oof(cv, m[[2]], Z5(train), train$aki, "lambda.min", family = "binomial", standardize = FALSE))
  note("ward7_aki", m[[1]], sprintf("alpha = %d, lambda min = %.4g (5-fold CV deviance); standardised on training rows", m[[2]], cv$lambda.min), ver("glmnet"))
}
train$akif <- factor(train$aki); fP5f <- reformulate(P5, "akif")
ctree <- rpart(fP5f, data = train, method = "class", control = rpart.control(cp = 0.001, minsplit = 20, xval = folds))
cpc <- ctree$cptable; bc <- which.min(cpc[, "xerror"])
cpc_1se <- cpc[cpc[, "xerror"] <= cpc[bc, "xerror"] + cpc[bc, "xstd"], "CP"][1]
ctree_1se <- prune(ctree, cp = cpc_1se)
w7("ward7_aki", "tree_1se", function(a, b) {
  a$akif <- factor(a$aki, levels = 0:1)
  t <- if (nrow(a) == nrow(train)) ctree_1se else rpart(fP5f, a, method = "class", control = rpart.control(cp = cpc_1se, minsplit = 20, xval = 0))
  predict(t, b)[, "1"] }, sprintf("1-SE rule on rpart's CV (shared folds): cp = %.4g, %d leaves", cpc_1se, nleaf(ctree_1se)), ver("rpart"))
w7("ward7_aki", "tree_depth2_p10", function(a, b) {
  a$akif <- factor(a$aki, levels = 0:1)
  t <- rpart(update(f10, akif ~ .), a, method = "class", control = rpart.control(maxdepth = 2, cp = 0, xval = 0))
  predict(t, b)[, "1"] }, "maxdepth = 2, cp = 0 (the Chapter 6 gate tree)", ver("rpart"))
# Bayesian logistic: exactly the call of ch08_stan.R
stan_logit <- function(a, b) {
  f <- stan_glm(fP5, data = a, family = binomial(link = "logit"), prior = normal(0, 2, autoscale = FALSE),
                prior_intercept = normal(0, 5, autoscale = FALSE), chains = 2, iter = 2000, seed = SEED, refresh = 0)
  colMeans(posterior_epred(f, newdata = b))
}
w7("ward7_aki", "bayes_logit", stan_logit,
   "priors normal(0, 2) on coefficients, normal(0, 5) on intercept (Chapter 8); 2 chains x 2 000 iterations; posterior mean risk", ver("rstanarm"))

# =========================================================================================
# 2. ONCO-1
# =========================================================================================
cat("ONCO-1 ...\n")
onco <- read.csv("data/onco1.csv")
spO <- read.csv("data/splits/onco1_split.csv")
spO <- spO[match(onco$patient_id, spO$patient_id), ]
leaky <- c("gene_08", "gene_14", "gene_19")
genes <- grep("^gene_", names(onco), value = TRUE); g47 <- setdiff(genes, leaky)
clin <- c("treatment", "dose_intensity", "patient_age", "tumor_grade", "performance_score")
# The two named patients (#0042 and #0917), one identity book-wide: see ledger_README.md
PAT <- c("#0042" = "P5440", "#0917" = "P9541")
idx_tr <- which(spO$set == "train"); idx_te <- which(spO$set == "test")
fO <- spO$fold[idx_tr]
prow <- match(PAT, onco$patient_id)
stopifnot(all(spO$set[prow] == "test"), all(onco$treatment[prow] == "chemo"))
yr <- onco$response_percent

# ---- 2a. regression (response %, all patients) — the model matrix of ch04_glmnet.R ----
d <- onco[, setdiff(names(onco), c("patient_id", "high_response", "baseline_tumor_mm", "post_tumor_mm", leaky))]
d$treatment <- factor(d$treatment, levels = c("no_chemo", "chemo"))
d$tumor_grade <- factor(d$tumor_grade, levels = c("G1", "G2", "G3"))
X_all <- model.matrix(response_percent ~ ., data = d)[, -1]
stO <- std_fit(X_all[idx_tr, ]); ZO <- std(X_all, stO)
oreg <- function(mid, fp, tuning, software) {          # fp(train rows, new rows) -> predictions
  put("onco1_reg", mid, "test", onco$patient_id[idx_te], fp(idx_tr, idx_te))
  put("onco1_reg", mid, "patient", names(PAT), fp(idx_tr, prow))
  out <- numeric(length(idx_tr))
  for (v in 1:5) out[fO == v] <- fp(idx_tr[fO != v], idx_tr[fO == v])
  put("onco1_reg", mid, "oof", onco$patient_id[idx_tr], out)
  note("onco1_reg", mid, tuning, software)
}
oreg("mean", function(a, b) rep(mean(yr[a]), length(b)), "—", Rv)
lmfp <- function(vars) function(a, b) predict(lm(reformulate(vars, "response_percent"), onco[a, ]), onco[b, ])
oreg("ols_clinical", lmfp(clin), "—", Rv)
oreg("ols", lmfp(c(clin, g47)), "—", Rv)
oreg("ols_leaky", lmfp(c(clin, genes)), "includes the leaky genes 08, 14, 19 (leakage warning only)", Rv)
ytrO <- yr[idx_tr]
cvsO <- lapply(alphas, function(a) cv.glmnet(ZO[idx_tr, ], ytrO, alpha = a, foldid = fO, standardize = FALSE,
                                             lambda.min.ratio = if (a == 0) 1e-6 else 1e-4))
bestO <- which.min(sapply(cvsO, function(cv) min(cv$cvm)))
for (m in list(list("ridge", 1), list("lasso", 11), list("enet", bestO))) for (rule in c("lambda.min", "lambda.1se")) {
  cv <- cvsO[[m[[2]]]]; a <- alphas[m[[2]]]
  mid <- paste0(m[[1]], if (rule == "lambda.min") "_min" else "_1se")
  put("onco1_reg", mid, "test", onco$patient_id[idx_te], predict(cv, ZO[idx_te, ], s = rule))
  put("onco1_reg", mid, "patient", names(PAT), predict(cv, ZO[prow, ], s = rule))
  put("onco1_reg", mid, "oof", onco$patient_id[idx_tr], glm_oof(cv, a, ZO[idx_tr, ], ytrO, rule, fold = fO, standardize = FALSE))
  b <- as.matrix(coef(cv, s = rule))[-1, 1]
  note("onco1_reg", mid, sprintf("alpha = %.1f, lambda = %.4g (%s, 5-fold CV MSE), %d of %d coefficients non-zero",
                                 a, cv[[rule]], if (rule == "lambda.min") "λ min" else "λ 1-SE", sum(b != 0), length(b)), ver("glmnet"))
}
enet_alpha_o1 <- alphas[bestO]

# ---- 2b. classification (high response), chemotherapy arm — the code of ch05_precompute.R,
#          with ONE change: genes are standardised with the TRAINING rows' mean and SD ----------
chemo <- onco[onco$treatment == "chemo", ]
chemo$tumor_grade <- factor(chemo$tumor_grade, levels = c("G1", "G2", "G3"))
chemo$dose10 <- chemo$dose_intensity * 10; chemo$age10 <- chemo$patient_age / 10
chemo <- merge(chemo, read.csv("data/splits/onco1_split.csv"), by = "patient_id", sort = FALSE)
chemo <- chemo[order(chemo$patient_id), ]
gmu <- colMeans(chemo[chemo$set == "train", g47]); gsd <- apply(chemo[chemo$set == "train", g47], 2, sd)
chemo[g47] <- scale(chemo[g47], gmu, gsd)
ctr <- chemo[chemo$set == "train", ]; cte <- chemo[chemo$set == "test", ]
cpat <- chemo[match(PAT, chemo$patient_id), ]
f_small <- high_response ~ dose10 + tumor_grade + performance_score + age10 + gene_01 + gene_05 + gene_12
f_all <- as.formula(paste("high_response ~ dose10 + tumor_grade + performance_score + age10 +", paste(g47, collapse = " + ")))
ocls <- function(mid, fp, tuning, software) {          # fp(train df, new df)
  put("onco1_class", mid, "test", cte$patient_id, fp(ctr, cte))
  put("onco1_class", mid, "patient", names(PAT), fp(ctr, cpat))
  out <- numeric(nrow(ctr))
  for (v in 1:5) out[ctr$fold == v] <- fp(ctr[ctr$fold != v, ], ctr[ctr$fold == v, ])
  put("onco1_class", mid, "oof", ctr$patient_id, out)
  note("onco1_class", mid, tuning, software)
}
ocls("prevalence", function(a, b) rep(mean(a$high_response), nrow(b)), "—", Rv)
ocls("logit_7", function(a, b) predict(glm(f_small, binomial, a), b, type = "response"), "unpenalised", Rv)
ocls("logit_all", function(a, b) suppressWarnings(predict(suppressWarnings(glm(f_all, binomial, a)), b, type = "response")),
     "unpenalised (quasi-separation warnings: some coefficients very large)", Rv)
Xc <- function(dd) model.matrix(f_all, dd)[, -1]
for (m in list(list("lasso_logit", 1), list("enet_logit", 0.5), list("ridge_logit", 0))) {
  cv <- cv.glmnet(Xc(ctr), ctr$high_response, family = "binomial", alpha = m[[2]], foldid = ctr$fold)
  put("onco1_class", m[[1]], "test", cte$patient_id, predict(cv, Xc(cte), s = "lambda.min", type = "response"))
  put("onco1_class", m[[1]], "patient", names(PAT), predict(cv, Xc(cpat), s = "lambda.min", type = "response"))
  put("onco1_class", m[[1]], "oof", ctr$patient_id,
      glm_oof(cv, m[[2]], Xc(ctr), ctr$high_response, "lambda.min", family = "binomial", fold = ctr$fold))
  b <- as.matrix(coef(cv, s = "lambda.min"))[-1, 1]
  note("onco1_class", m[[1]], sprintf("alpha = %.1f, lambda min = %.4g (5-fold CV deviance; glmnet standardize = TRUE), %d of %d non-zero",
                                      m[[2]], cv$lambda.min, sum(b != 0), length(b)), ver("glmnet"))
}

# =========================================================================================
# 3. HomeBeat: resting_hr ~ age + frailty + baseline_inflammation (Chapter 8 task)
# =========================================================================================
cat("HomeBeat ...\n")
hb <- read.csv("data/homebeat.csv")
sph <- read.csv("data/splits/homebeat_split.csv")
stopifnot(identical(sph$participant_id, hb$participant_id))
htr <- hb[sph$set == "train", ]; hte <- hb[sph$set == "test", ]; fH <- sph$fold[sph$set == "train"]
lena <- data.frame(age = 72, frailty = 0.55, baseline_inflammation = 4.1)    # as in Chapter 8
fhb <- resting_hr ~ age + frailty + baseline_inflammation
hbm <- function(mid, fp, tuning, software) {
  put("homebeat", mid, "test", hte$participant_id, fp(htr, hte))
  put("homebeat", mid, "lena", "Lena", fp(htr, lena))
  put("homebeat", mid, "oof", htr$participant_id, oof(fp, htr, fH))
  note("homebeat", mid, tuning, software)
}
hbm("mean", function(a, b) rep(mean(a$resting_hr), nrow(b)), "—", Rv)
hbm("ols", function(a, b) predict(lm(fhb, a), b), "—", Rv)
hbm("bayes_lin", function(a, b) {
  f <- stan_glm(resting_hr ~ frailty + age + baseline_inflammation, data = a, family = gaussian(),
                prior_intercept = normal(60, 20, autoscale = FALSE), prior = normal(0, 20, autoscale = FALSE),
                prior_aux = normal(0, 10, autoscale = FALSE), chains = 2, iter = 2000, seed = SEED, refresh = 0)
  colMeans(posterior_epred(f, newdata = b)) },
  "the Chapter 8 adjusted model and priors, refitted on the 400 training participants; posterior mean of E(y | x)", ver("rstanarm"))
cat(sprintf("R models done in %.0f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

# =========================================================================================
# 4. Python predictions, then all metrics
# =========================================================================================
if (!"--skip-py" %in% args) {
  cat("waiting for the Python part ...\n")
  t_wait <- Sys.time()
  while (!file.exists(py_exit)) {
    Sys.sleep(5)
    if (difftime(Sys.time(), t_wait, units = "mins") > 30) stop("ledger.py did not finish within 30 minutes")
  }
  st <- as.integer(readLines(py_exit))
  if (st != 0) stop("ledger.py failed; see ", file.path(PARTS, "py_log.txt"))
}
pyP <- read.csv(file.path(PARTS, "py_preds.csv"), colClasses = c(id = "character"), encoding = "UTF-8")
pyM <- read.csv(file.path(PARTS, "py_meta.csv"), encoding = "UTF-8")
PR <- rbind(do.call(rbind, P), pyP[, c("example", "model_id", "kind", "id", "pred")])
write.csv(PR, file.path(PARTS, "all_preds.csv"), row.names = FALSE, fileEncoding = "UTF-8")   # every prediction, R + Python
TU <- rbind(do.call(rbind, TUNE), pyM[, c("example", "model_id", "tuning", "software")])

# ---- the catalogue: order, chapter, exact name, predictor set, role --------------------
# role: canonical = part of the ledger ladder; teaching = labelled teaching set (P2);
#       variant = a chapter's alternative implementation kept so that the chapter can label it;
#       warning = shown only as the leakage warning; reference = baseline with no model.
cat_row <- function(example, order, chapter, model_id, model, predictor_set, role = "canonical")
  data.frame(example, order, chapter, model_id, model, predictor_set, role)
p10 <- "P10 (site, age, sofa, lactate, map_min, baseline_creat, protocol_adherence, bun, crp, albumin)"
p5 <- "P5 (sofa, lactate, map_min, baseline_creat, protocol_adherence)"
oc <- "clinical (treatment, dose, age, grade, performance) + 47 non-leaky genes"
occ <- "clinical (dose, age, grade, performance) + 47 non-leaky genes"
gtun <- function(ex, id) TU$tuning[TU$example == ex & TU$model_id == id][1]
nb_of <- function(s) regmatches(s, regexpr("[0-9]+ (rounds|epochs)", s))
CAT <- rbind(
  cat_row("ward7_reg", 1, 1, "mean", "Training mean (no model)", "none", "reference"),
  cat_row("ward7_reg", 2, 1, "sofa_line", "Linear regression on SOFA only", "SOFA"),
  cat_row("ward7_reg", 3, 3, "p2_line", "Linear regression, two predictors (teaching set)", "P2 (sofa, baseline_creat)", "teaching"),
  cat_row("ward7_reg", 4, 3, "knn_p2", sprintf("kNN regression, k = %d, standardised (teaching set)", knn_k$knn_p2), "P2 (sofa, baseline_creat)", "teaching"),
  cat_row("ward7_reg", 5, 3, "knn_p10", sprintf("kNN regression, k = %d by CV, standardised", knn_k$knn_p10), p10),
  cat_row("ward7_reg", 6, 4, "ols", "Linear regression (OLS)", p10),
  cat_row("ward7_reg", 7, 4, "ridge_min", "Ridge regression (λ min)", p10),
  cat_row("ward7_reg", 8, 4, "ridge_1se", "Ridge regression (λ 1-SE)", p10),
  cat_row("ward7_reg", 9, 4, "lasso_min", "Lasso (λ min)", p10),
  cat_row("ward7_reg", 10, 4, "lasso_1se", "Lasso (λ 1-SE)", p10),
  cat_row("ward7_reg", 11, 4, "enet_min", sprintf("Elastic net (α = %.1f, λ min)", enet_alpha_w7), p10),
  cat_row("ward7_reg", 12, 4, "enet_1se", sprintf("Elastic net (α = %.1f, λ 1-SE)", enet_alpha_w7), p10),
  cat_row("ward7_reg", 13, 6, "tree_1se", "Regression tree, pruned by the 1-SE rule (rpart)", p10),
  cat_row("ward7_reg", 14, 6, "tree_large", "Regression tree, grown large (rpart, cp = 0.001)", p10),
  cat_row("ward7_reg", 15, 6, "bagging500", "Bagged regression trees (500 trees)", p10),
  cat_row("ward7_reg", 16, 6, "rf500", "Random forest (500 trees)", p10),
  cat_row("ward7_reg", 17, 6, "xgb", sprintf("XGBoost (%s, number chosen by 5-fold CV)", nb_of(gtun("ward7_reg", "xgb"))), p10),
  cat_row("ward7_reg", 18, 6, "tree_sklearn_full", "Regression tree, fully grown (scikit-learn; old Chapter 6 table)", p10, "variant"),
  cat_row("ward7_reg", 19, 6, "tree_sklearn_depthcv", "Regression tree, depth 3 by CV (scikit-learn; old Chapter 6 table)", p10, "variant"),
  cat_row("ward7_reg", 20, 7, "mlp", sprintf("Neural network, MLP 32-16 (scikit-learn, %s)", nb_of(gtun("ward7_reg", "mlp"))), p10),
  cat_row("ward7_reg", 21, 8, "bayes_lin", "Bayesian linear regression (rstanarm)", p10),
  cat_row("ward7_reg", 22, 8, "bart", "BART, 50 trees (pymc-bart)", p10),
  cat_row("ward7_reg", 23, 9, "ols_imp_mean", "Linear regression after training-mean imputation + missing indicators (icu_aki_missing)", paste(p10, "+ 3 missing indicators")),
  cat_row("ward7_reg", 24, 9, "ols_mi", sprintf("Linear regression after multiple imputation (mice, m = %d; icu_aki_missing)", MI_M), p10),

  cat_row("ward7_aki", 1, 1, "prevalence", "Prevalence: training AKI rate (no model)", "none", "reference"),
  cat_row("ward7_aki", 2, 5, "logit_sofa", "Logistic regression, SOFA only", "SOFA"),
  cat_row("ward7_aki", 3, 5, "logit_p5", "Logistic regression (unpenalised), the headline model", p5),
  cat_row("ward7_aki", 4, 5, "ridge_logit", "Ridge logistic regression (λ min)", p5),
  cat_row("ward7_aki", 5, 5, "lasso_logit", "Lasso logistic regression (λ min)", p5),
  cat_row("ward7_aki", 6, 6, "tree_1se", "Classification tree, pruned by the 1-SE rule (rpart)", p5),
  cat_row("ward7_aki", 7, 6, "tree_depth2_p10", "Classification tree, depth 2 (rpart; the Chapter 6 gate tree)", p10, "variant"),
  cat_row("ward7_aki", 8, 6, "rf500", "Random forest classifier (500 trees)", p5),
  cat_row("ward7_aki", 9, 6, "xgb", sprintf("XGBoost classifier (%s, number chosen by 5-fold CV)", nb_of(gtun("ward7_aki", "xgb"))), p5),
  cat_row("ward7_aki", 10, 7, "mlp", sprintf("Neural network classifier, MLP 32-16 (scikit-learn, %s)", nb_of(gtun("ward7_aki", "mlp"))), p5),
  cat_row("ward7_aki", 11, 8, "bayes_logit", "Bayesian logistic regression (rstanarm)", p5),
  cat_row("ward7_aki", 12, 8, "bart_p10", "BART → P(Δ creatinine ≥ 0.3), from the regression BART (P10, not P5)", p10, "second predictor set"),

  cat_row("onco1_reg", 1, 3, "mean", "Training mean (no model)", "none", "reference"),
  cat_row("onco1_reg", 2, 3, "ols_clinical", "Linear regression, clinical predictors only", "clinical (treatment, dose, age, grade, performance)"),
  cat_row("onco1_reg", 3, 3, "ols_leaky", "Linear regression WITH the leaky genes 08, 14, 19 (leakage warning only)", "clinical + all 50 genes", "warning"),
  cat_row("onco1_reg", 4, 3, "ols", "Linear regression (OLS)", oc),
  cat_row("onco1_reg", 5, 4, "ridge_min", "Ridge regression (λ min)", oc),
  cat_row("onco1_reg", 6, 4, "ridge_1se", "Ridge regression (λ 1-SE)", oc),
  cat_row("onco1_reg", 7, 4, "lasso_min", "Lasso (λ min)", oc),
  cat_row("onco1_reg", 8, 4, "lasso_1se", "Lasso (λ 1-SE)", oc),
  cat_row("onco1_reg", 9, 4, "enet_min", sprintf("Elastic net (α = %.1f, λ min)", enet_alpha_o1), oc),
  cat_row("onco1_reg", 10, 4, "enet_1se", sprintf("Elastic net (α = %.1f, λ 1-SE)", enet_alpha_o1), oc),
  cat_row("onco1_reg", 11, 6, "rf500", "Random forest (500 trees)", oc),
  cat_row("onco1_reg", 12, 6, "xgb", sprintf("XGBoost (%s, number chosen by 5-fold CV)", nb_of(gtun("onco1_reg", "xgb"))), oc),
  cat_row("onco1_reg", 13, 7, "mlp_multi", sprintf("Multi-output neural network, response-%% head (PyTorch, %s)", nb_of(gtun("onco1_reg", "mlp_multi"))), oc),

  cat_row("onco1_class", 1, 5, "prevalence", "Prevalence: training high-response rate (no model)", "none", "reference"),
  cat_row("onco1_class", 2, 5, "logit_7", "Logistic regression, 7 chosen predictors (dose, grade, performance, age, genes 01/05/12)", "7 chosen predictors"),
  cat_row("onco1_class", 3, 5, "logit_all", "Logistic regression (unpenalised), all 51 predictors", occ),
  cat_row("onco1_class", 4, 5, "lasso_logit", "Lasso logistic regression (λ min)", occ),
  cat_row("onco1_class", 5, 5, "enet_logit", "Elastic net logistic regression (α = 0.5, λ min)", occ),
  cat_row("onco1_class", 6, 5, "ridge_logit", "Ridge logistic regression (λ min)", occ),
  cat_row("onco1_class", 7, 6, "ridge_logistic_C1_sklearn", "Ridge logistic regression (C = 1, scikit-learn; the old Chapter 6 'logistic regression')", occ, "variant"),
  cat_row("onco1_class", 8, 6, "tree_depth4", "Classification tree, depth 4 (scikit-learn)", occ),
  cat_row("onco1_class", 9, 6, "rf500", "Random forest classifier (500 trees)", occ),
  cat_row("onco1_class", 10, 6, "xgb", sprintf("XGBoost classifier (%s, number chosen by 5-fold CV)", nb_of(gtun("onco1_class", "xgb"))), occ),
  cat_row("onco1_class", 11, 7, "mlp_multi", "Multi-output neural network, high-response head (PyTorch; trained on all 800 training patients)", oc),

  cat_row("homebeat", 1, 8, "mean", "Training mean (no model)", "none", "reference"),
  cat_row("homebeat", 2, 8, "ols", "Linear regression (OLS)", "age, frailty, baseline_inflammation"),
  cat_row("homebeat", 3, 8, "bayes_lin", "Bayesian linear regression (rstanarm, Chapter 8 priors)", "age, frailty, baseline_inflammation"),
  cat_row("homebeat", 4, 8, "rf500", "Random forest (500 trees)", "age, frailty, baseline_inflammation"),
  cat_row("homebeat", 5, 8, "xgb", sprintf("XGBoost (%s, number chosen by 5-fold CV)", nb_of(gtun("homebeat", "xgb"))), "age, frailty, baseline_inflammation"),
  cat_row("homebeat", 6, 8, "xgb300", "XGBoost (300 rounds fixed; the Chapter 8 setting)", "age, frailty, baseline_inflammation", "variant"),
  cat_row("homebeat", 7, 8, "bart", "BART, 50 trees (pymc-bart)", "age, frailty, baseline_inflammation")
)
CAT <- merge(CAT, TU, by = c("example", "model_id"), all.x = TRUE, sort = FALSE)
stopifnot(!anyNA(CAT$tuning))

# ---- metrics ---------------------------------------------------------------------------
auc <- function(y, p) { r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
clip <- function(p, e = 1e-6) pmin(pmax(p, e), 1 - e)
# calibration slope: logistic regression of y on logit(p); NA if p is constant or the fit does not converge
cal_slope <- function(y, p) {
  if (sd(p) < 1e-12) return(NA_real_)
  g <- suppressWarnings(glm(y ~ qlogis(clip(p)), family = binomial, control = glm.control(maxit = 100)))
  if (!g$converged) NA_real_ else unname(coef(g)[2])
}
# calibration-in-the-large: intercept a of logit P(y) = a + logit(p) (offset), by 1-D maximum likelihood
citl <- function(y, p) {
  o <- qlogis(clip(p))
  optimize(function(a) -sum(y * plogis(a + o, log.p = TRUE) + (1 - y) * plogis(-(a + o), log.p = TRUE)),
           c(-20, 20), tol = 1e-10)$minimum
}
logloss <- function(y, p) { p <- clip(p, 1e-15); -mean(y * log(p) + (1 - y) * log(1 - p)) }
truth <- list(
  ward7_reg = setNames(icu$delta_creatinine_48h, icu$patient_id),
  ward7_aki = setNames(icu$aki, icu$patient_id),
  onco1_reg = setNames(onco$response_percent, onco$patient_id),
  onco1_class = setNames(onco$high_response, onco$patient_id),
  homebeat = setNames(hb$resting_hr, hb$participant_id))
ybar_train <- c(ward7_reg = mean(train$delta_creatinine_48h), onco1_reg = mean(ytrO), homebeat = mean(htr$resting_hr))
is_reg <- c(ward7_reg = TRUE, ward7_aki = FALSE, onco1_reg = TRUE, onco1_class = FALSE, homebeat = TRUE)

metric_row <- function(ex, mid) {
  g <- PR[PR$example == ex & PR$model_id == mid, ]
  te <- g[g$kind == "test", ]; tr <- g[g$kind == "oof", ]
  if (nrow(te) == 0) return(NULL)
  y <- truth[[ex]][te$id]; p <- te$pred
  ntr <- c(ward7_reg = 1000, ward7_aki = 1000, onco1_reg = 800, onco1_class = nrow(ctr), homebeat = 400)[[ex]]
  if (ex == "onco1_class" && mid == "mlp_multi") ntr <- 800
  if (is_reg[[ex]]) {
    data.frame(n_train = ntr, n_test = length(y),
               cv_MAE = if (nrow(tr)) mean(abs(truth[[ex]][tr$id] - tr$pred)) else NA,
               test_MAE = mean(abs(y - p)), test_RMSE = sqrt(mean((y - p)^2)),
               test_R2 = 1 - sum((y - p)^2) / sum((y - ybar_train[[ex]])^2))
  } else {
    data.frame(n_train = ntr, n_test = length(y),
               cv_logloss = if (nrow(tr)) logloss(truth[[ex]][tr$id], tr$pred) else NA,
               test_AUC = auc(y, p), test_Brier = mean((p - y)^2),
               cal_slope = cal_slope(y, p), cal_intercept = citl(y, p))
  }
}
sig <- function(df) { df[] <- lapply(df, function(x) if (is.numeric(x) && !is.integer(x)) signif(x, 7) else x); df }
for (ex in names(truth)) {
  cc <- CAT[CAT$example == ex, ]; cc <- cc[order(cc$order), ]
  M <- do.call(rbind, lapply(cc$model_id, function(m) metric_row(ex, m)))
  tab <- cbind(cc[, c("order", "chapter", "model_id", "model", "predictor_set", "role")], M,
               cc[, c("tuning", "software")])
  if (ex == "homebeat") {     # stress_score: a separate task, Chapter 19 TSTR numbers recorded as-is
    m19 <- read.csv("data/precomputed/ch19/homebeat_metrics.csv")
    tab$task <- "resting_hr ~ age + frailty + baseline_inflammation"
    s19 <- data.frame(order = 100 + seq_len(nrow(m19)), chapter = 19, model_id = paste0("tstr_", m19$source),
                      model = ifelse(m19$source == "real_holdout",
                                   "OLS trained on the 400 real training records, tested on the real holdout (reference)",
                                   paste0("OLS trained on synthetic records (", m19$source, "), tested on the real holdout (TSTR)")),
                      predictor_set = "7 HomeBeat signals", role = "separate task", n_train = 400, n_test = 100,
                      cv_MAE = NA, test_MAE = NA, test_RMSE = m19$tstr_rmse, test_R2 = NA,
                      tuning = "as published in data/precomputed/ch19/homebeat_metrics.csv (copied, not refitted)",
                      software = "scripts/precompute/ch19_generative.py", task = "stress_score (Chapter 19, TSTR)")
    tab <- rbind(tab, s19)
  }
  write.csv(sig(tab), file.path(OUT, paste0(ex, ".csv")), row.names = FALSE, na = "", fileEncoding = "UTF-8")
}

# ---- named patients --------------------------------------------------------------------
cat_of <- function(ex) { cc <- CAT[CAT$example == ex, ]; cc[order(cc$order), ] }
tn <- do.call(rbind, lapply(c("ward7_reg", "ward7_aki"), function(ex) {
  cc <- cat_of(ex)
  do.call(rbind, lapply(seq_len(nrow(cc)), function(i) {
    g <- PR[PR$example == ex & PR$model_id == cc$model_id[i] & PR$kind == "tonight", ]
    if (!nrow(g) || all(is.na(g$pred))) return(NULL)
    v <- setNames(g$pred, g$id)
    data.frame(example = ex, quantity = if (ex == "ward7_reg") "delta_creatinine_48h (mg/dL)" else "P(AKI)",
               order = cc$order[i], chapter = cc$chapter[i], model_id = cc$model_id[i], model = cc$model[i],
               predictor_set = cc$predictor_set[i], role = cc$role[i],
               Aoife = v[tn_names[1]], Tomas = v[tn_names[2]], Priya = v[tn_names[3]], row.names = NULL)
  }))
}))
write.csv(sig(tn), file.path(OUT, "ward7_tonight.csv"), row.names = FALSE, na = "", fileEncoding = "UTF-8")

prof <- onco[prow, c("patient_id", "treatment", "dose_intensity", "patient_age", "tumor_grade", "performance_score",
                     "baseline_tumor_mm", "post_tumor_mm", "response_percent", "high_response")]
prof$label <- names(PAT); prof$set <- spO$set[prow]
op <- do.call(rbind, lapply(c("onco1_reg", "onco1_class"), function(ex) {
  cc <- cat_of(ex)
  do.call(rbind, lapply(seq_len(nrow(cc)), function(i) {
    g <- PR[PR$example == ex & PR$model_id == cc$model_id[i] & PR$kind == "patient", ]
    if (!nrow(g)) return(NULL)
    data.frame(example = ex, quantity = if (ex == "onco1_reg") "predicted response %" else "P(high response)",
               order = cc$order[i], chapter = cc$chapter[i], model_id = cc$model_id[i], model = cc$model[i],
               role = cc$role[i], label = g$id, prediction = g$pred)
  }))
}))
op <- merge(prof[, c("label", "patient_id", "set", "treatment", "dose_intensity", "patient_age", "tumor_grade",
                     "performance_score", "response_percent", "high_response")], op, by = "label")
op <- op[order(op$example != "onco1_reg", op$order, op$label), ]
write.csv(sig(op), file.path(OUT, "onco1_patients.csv"), row.names = FALSE, na = "", fileEncoding = "UTF-8")

cc <- cat_of("homebeat")
le <- do.call(rbind, lapply(seq_len(nrow(cc)), function(i) {
  g <- PR[PR$example == "homebeat" & PR$model_id == cc$model_id[i] & PR$kind == "lena", ]
  data.frame(order = cc$order[i], chapter = cc$chapter[i], model_id = cc$model_id[i], model = cc$model[i],
             role = cc$role[i], fitted_on = "400 training participants (homebeat_split.csv)",
             lena_mean_hr = g$pred, source = "ledger")
}))
l8 <- read.csv("data/precomputed/ch08/homebeat_lena.csv")
le <- rbind(le, data.frame(order = 99, chapter = 8, model_id = "ch08_stan_all500",
                           model = "Bayesian linear regression, Chapter 8 adjusted model (reference)", role = "reference",
                           fitted_on = "all 500 participants (Chapter 8 inference analysis)",
                           lena_mean_hr = l8$mean[l8$quantity == "mean_hr"],
                           source = "data/precomputed/ch08/homebeat_lena.csv (copied)"))
write.csv(sig(le), file.path(OUT, "homebeat_lena.csv"), row.names = FALSE, na = "", fileEncoding = "UTF-8")

# provenance
writeLines(c(sprintf('{"R": "%s", "glmnet": "%s", "rpart": "%s", "rstanarm": "%s", "mice": "%s",',
                     getRversion(), packageVersion("glmnet"), packageVersion("rpart"), packageVersion("rstanarm"), packageVersion("mice")),
             sprintf(' "seed": %d, "onco1_patients": {"#0042": "%s", "#0917": "%s"},', SEED, PAT[1], PAT[2]),
             sprintf(' "run_seconds": %.0f}', as.numeric(difftime(Sys.time(), t0, units = "secs")))),
           file.path(OUT, "versions.json"))
cat(sprintf("Ledger written to %s in %.0f s\n", OUT, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
