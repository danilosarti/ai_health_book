# =====================================================================
# Chapter 4: precomputed penalised regressions on ONCO-1 (browser subset)
# Produces, in data/precomputed/ch04/:
#   lambda_path.json   coefficient paths + CV curves for ridge, elastic net, lasso
#   comparison.csv     train/test MAE and RMSE of OLS, ridge, lasso, elastic net
#                      (n_train = 800 and a small-sample run with the first 80 training patients)
#   coefficients.csv   standardised coefficients of each model (n_train = 800)
#   enet_alpha.csv     cross-validated error at lambda.min for each alpha
# Run from the book folder:  Rscript scripts/precompute/ch04_glmnet.R
# Needs: glmnet (Friedman, Hastie & Tibshirani 2010).
# Train/test split and CV folds: the shared book-wide split data/splits/onco1_split.csv.
# =====================================================================
suppressPackageStartupMessages(library(glmnet))
out <- "data/precomputed/ch04"; dir.create(out, recursive = TRUE, showWarnings = FALSE)

onco <- read.csv("data/onco1.csv")
# Columns that must not enter the model:
#  - identifiers, the binary version of the outcome, the tumour sizes that define it;
#  - genes 08, 14, 19: noisy copies of the outcome (leakage, see Chapter 3).
leaky <- c("gene_08", "gene_14", "gene_19")
drop  <- c("patient_id", "high_response", "baseline_tumor_mm", "post_tumor_mm", leaky)
d <- onco[, setdiff(names(onco), drop)]
d$treatment   <- factor(d$treatment, levels = c("no_chemo", "chemo"))
d$tumor_grade <- factor(d$tumor_grade, levels = c("G1", "G2", "G3"))
X_all <- model.matrix(response_percent ~ ., data = d)[, -1]
y_all <- d$response_percent
p <- ncol(X_all)

split <- read.csv("data/splits/onco1_split.csv")
split <- split[match(onco$patient_id, split$patient_id), ]
stopifnot(!anyNA(split$set))
idx_train <- which(split$set == "train")
idx_test  <- which(split$set == "test")

standardise <- function(Xtr, Xte) {
  mu <- colMeans(Xtr); s <- apply(Xtr, 2, sd)
  list(tr = scale(Xtr, mu, s), te = scale(Xte, mu, s))
}
mae  <- function(y, yhat) mean(abs(y - yhat))
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

fit_all <- function(rows_train, tag) {
  S  <- standardise(X_all[rows_train, ], X_all[idx_test, ])
  Xtr <- S$tr; Xte <- S$te; ytr <- y_all[rows_train]; yte <- y_all[idx_test]
  foldid <- split$fold[rows_train]                 # shared folds, the same for every model
  ols <- lm(ytr ~ Xtr)
  b_ols <- coef(ols); b_ols[is.na(b_ols)] <- 0
  pr_ols <- function(X) drop(cbind(1, X) %*% b_ols)
  res <- list(); coefs <- data.frame(term = colnames(X_all), OLS = b_ols[-1])
  res[["OLS"]] <- c(mae(ytr, pr_ols(Xtr)), rmse(ytr, pr_ols(Xtr)), mae(yte, pr_ols(Xte)), rmse(yte, pr_ols(Xte)), p, NA, NA)
  alphas <- seq(0, 1, by = 0.1)
  # a long lambda grid (down to 1e-6 x lambda_max) so that the ridge path reaches close to OLS
  cvs <- lapply(alphas, function(a) cv.glmnet(Xtr, ytr, alpha = a, foldid = foldid, standardize = FALSE,
                                              lambda.min.ratio = if (a == 0) 1e-6 else 1e-4))
  best <- which.min(sapply(cvs, function(cv) min(cv$cvm)))
  models <- list(Ridge = list(cv = cvs[[1]], a = 0), Lasso = list(cv = cvs[[11]], a = 1),
                 `Elastic net` = list(cv = cvs[[best]], a = alphas[best]))
  for (nm in names(models)) for (rule in c("lambda.min", "lambda.1se")) {
    cv <- models[[nm]]$cv; lam <- cv[[rule]]
    ftr <- drop(predict(cv, Xtr, s = rule)); fte <- drop(predict(cv, Xte, s = rule))
    b <- drop(as.matrix(coef(cv, s = rule)))[-1]
    lab <- paste0(nm, if (rule == "lambda.min") " (λ min)" else " (λ 1-SE)")
    res[[lab]] <- c(mae(ytr, ftr), rmse(ytr, ftr), mae(yte, fte), rmse(yte, fte), sum(b != 0), models[[nm]]$a, lam)
    coefs[[lab]] <- b
  }
  tab <- data.frame(scenario = tag, model = names(res), do.call(rbind, res), row.names = NULL)
  names(tab)[3:9] <- c("train_mae", "train_rmse", "test_mae", "test_rmse", "nonzero", "alpha", "lambda")
  list(tab = tab, coefs = coefs, cvs = cvs, alphas = alphas, Xtr = Xtr, ytr = ytr, Xte = Xte, yte = yte,
       foldid = foldid, best = best)
}

big   <- fit_all(idx_train, "n_train = 800")
small <- fit_all(idx_train[1:80], "n_train = 80")
num <- function(df) { df[] <- lapply(df, function(x) if (is.numeric(x)) signif(x, 5) else x); df }
write.csv(num(rbind(big$tab, small$tab)), file.path(out, "comparison.csv"), row.names = FALSE)
write.csv(num(big$coefs), file.path(out, "coefficients.csv"), row.names = FALSE)
write.csv(data.frame(alpha = big$alphas,
                     cv_mse_min = signif(sapply(big$cvs, function(cv) min(cv$cvm)), 5)),
          file.path(out, "enet_alpha.csv"), row.names = FALSE)

# ---- coefficient paths for the widget (n_train = 800) ----
path_of <- function(cv, Xte, yte) {
  B <- as.matrix(cv$glmnet.fit$beta)              # p x nlambda
  te <- apply(predict(cv$glmnet.fit, Xte), 2, function(f) rmse(yte, f))
  list(lambda = signif(cv$lambda, 5),
       beta   = lapply(seq_len(ncol(B)), function(k) round(B[, k], 4)),
       cvm    = signif(sqrt(cv$cvm), 5),          # CV RMSE
       cvlo   = signif(sqrt(pmax(cv$cvm - cv$cvsd, 0)), 5),
       cvhi   = signif(sqrt(cv$cvm + cv$cvsd), 5),
       nzero  = as.integer(cv$nzero),
       test_rmse = signif(te, 5),
       idx_min = which(cv$lambda == cv$lambda.min) - 1,   # 0-based for JavaScript
       idx_1se = which(cv$lambda == cv$lambda.1se) - 1)
}
role <- setNames(rep("noise gene", p), colnames(X_all))
role[c("treatmentchemo", "dose_intensity", "patient_age", "tumor_gradeG2", "tumor_gradeG3",
       "performance_score")] <- "clinical"
role[c("gene_01", "gene_05", "gene_12")] <- "causal gene"
json <- list(
  note = "ONCO-1 browser subset; 800 training patients (shared split), 5-fold CV; predictors standardised (coefficients per SD); genes 08/14/19 excluded (leakage).",
  predictors = colnames(X_all), role = unname(role),
  ridge = path_of(big$cvs[[1]], big$Xte, big$yte),
  enet  = path_of(big$cvs[[6]], big$Xte, big$yte),        # alpha = 0.5
  lasso = path_of(big$cvs[[11]], big$Xte, big$yte))
# scalars stay scalars, vectors become arrays
to_json2 <- function(x) {
  if (is.list(x)) {
    if (!is.null(names(x))) return(paste0("{", paste0('"', names(x), '":', vapply(x, to_json2, ""), collapse = ","), "}"))
    return(paste0("[", paste(vapply(x, to_json2, ""), collapse = ","), "]"))
  }
  v <- if (is.character(x)) paste0('"', x, '"') else format(x, scientific = FALSE, trim = TRUE, drop0trailing = TRUE)
  if (length(x) == 1 && !(is.character(x) && length(x) > 1)) {
    if (is.character(x) && !is.null(attr(x, "array"))) return(paste0("[", v, "]"))
    return(v)
  }
  paste0("[", paste(v, collapse = ","), "]")
}
writeLines(to_json2(json), file.path(out, "lambda_path.json"))
print(rbind(big$tab, small$tab)[, 1:9], digits = 3)
cat("best enet alpha:", big$alphas[big$best], "\n")
