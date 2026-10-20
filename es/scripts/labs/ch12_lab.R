# Chapter 12 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 0. Packages and data -----------------------------------------------------
library(randomForest)
lung  <- read.csv("data/lung_sim_complete.csv")
lung$male <- as.integer(lung$sex == "male")
lsp   <- read.csv("data/splits/lung_split.csv")
ltr   <- lung[lsp$set == "train", ];  lte <- lung[lsp$set == "test", ]
feats <- c("age", "male", "ph.ecog", "ph.karno", "wt.loss")
mae   <- function(y, yhat) mean(abs(y - yhat))

# 1. Two models of days of follow-up (remember: censoring is ignored) ------
lin <- lm(time ~ age + male + ph.ecog + ph.karno + wt.loss, data = ltr)
set.seed(2026)
rf  <- randomForest(time ~ age + male + ph.ecog + ph.karno + wt.loss, data = ltr, ntree = 300)
c(linear = mae(lte$time, predict(lin, lte)), forest = mae(lte$time, predict(rf, lte)))

# 2. Permutation importance for any model ----------------------------------
perm_importance <- function(predict_fun, data, y, features, n_shuffles = 10) {
  base <- mae(y, predict_fun(data))
  sapply(features, function(f) {
    mean(replicate(n_shuffles, {
      d <- data; d[[f]] <- sample(d[[f]])
      mae(y, predict_fun(d))
    })) - base                                    # shuffled minus original
  })
}
set.seed(2026)
round(perm_importance(function(d) predict(lin, d), lte, lte$time, feats), 1)
round(perm_importance(function(d) predict(rf, d), lte, lte$time, feats), 1)

# 3. ICE curves and partial dependence ---------------------------------------
ice_curves <- function(predict_fun, data, feature, grid) {
  sapply(grid, function(g) { d <- data; d[[feature]] <- g; predict_fun(d) })  # patients x grid
}
grid <- seq(1, 25, by = 1)
ice  <- ice_curves(function(d) predict(rf, d), ltr, "wt.loss", grid)
matplot(grid, t(ice - ice[, 1]), type = "l", lty = 1, col = "#00000020",
        xlab = "Weight loss (kg)", ylab = "Change in predicted days (centred ICE)")
lines(grid, colMeans(ice - ice[, 1]), lwd = 4, col = "#1f6f8b")

# 4. Friedman's H for one pair (at 60 sampled training patients) -------------
pd_at <- function(predict_fun, data, cols, points) {
  apply(points[, cols, drop = FALSE], 1, function(v) {
    d <- data; for (k in seq_along(cols)) d[[cols[k]]] <- v[k]; mean(predict_fun(d))
  })
}
h_stat <- function(predict_fun, data, a, b) {
  fa <- pd_at(predict_fun, data, a, data);  fa <- fa - mean(fa)
  fb <- pd_at(predict_fun, data, b, data);  fb <- fb - mean(fb)
  fab <- pd_at(predict_fun, data, c(a, b), data); fab <- fab - mean(fab)
  c(H = sqrt(sum((fab - fa - fb)^2) / sum(fab^2)), H_days = sqrt(mean((fab - fa - fb)^2)))
}
set.seed(2026)
sub <- ltr[sample(nrow(ltr), 60), ]
round(h_stat(function(d) predict(rf, d), sub, "ph.ecog", "wt.loss"), 3)
round(h_stat(function(d) predict(rf, d), sub, "age", "male"), 3)

# 5. Exact Shapley values for Ward 7 patients: the OLS model of Chapter 4 -----
#    (P10: the ten predictors of Chapters 4 and 6; site counts as one player)
icu   <- read.csv("data/icu_aki.csv")
split <- read.csv("data/splits/ward7_split.csv")
train <- icu[split$set == "train", ];  test <- icu[split$set == "test", ]
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
f10 <- c("site", "age", "sofa", "lactate", "map_min", "baseline_creat",
         "protocol_adherence", "bun", "crp", "albumin")
ols7 <- lm(reformulate(f10, "delta_creatinine_48h"), data = train)
round(mae(test$delta_creatinine_48h, predict(ols7, test)), 4)   # 0.1413, the ledger value (Chapter 4)
shapley_exact <- function(predict_fun, x, background, features) {
  p <- length(features)
  masks <- as.matrix(expand.grid(rep(list(0:1), p)))
  v <- apply(masks, 1, function(m) {
    d <- background
    for (j in which(m == 1)) d[[features[j]]] <- x[[features[j]]]
    mean(predict_fun(d))
  })
  key <- apply(masks, 1, paste, collapse = "")
  phi <- setNames(numeric(p), features)
  for (r in seq_len(nrow(masks))) {
    m <- masks[r, ]; s <- sum(m)
    for (j in which(m == 0)) {
      m2 <- m; m2[j] <- 1
      phi[j] <- phi[j] + factorial(s) * factorial(p - s - 1) / factorial(p) *
        (v[match(paste(m2, collapse = ""), key)] - v[r])
    }
  }
  c(baseline = v[1], phi, prediction = v[nrow(masks)])
}
set.seed(2026)
background <- train[sample(nrow(train), 100), ]
sh <- sapply(1:3, function(i) shapley_exact(function(d) predict(ols7, d), tonight[i, ], background, f10))
colnames(sh) <- tonight$name
round(sh, 3)     # each column: baseline + contributions = prediction

# 6. The random forest of Chapter 6 (500 trees, scikit-learn) --------------------
#    R cannot refit the scikit-learn forest exactly, and its exact Shapley values need
#    about a million predictions per patient: read those computed in advance by
#    scripts/precompute/ch12_explain.py (background: 200 training patients).
shf <- read.csv("data/precomputed/ch12/ward7_shapley.csv", encoding = "UTF-8")
shf <- shf[shf$model == "Random forest", ]
round(xtabs(phi ~ feature + name, data = shf), 3)
unique(shf[, c("name", "baseline", "prediction")])   # forest predictions 0.29 / -0.09 / 0.83, as in the ledger
led <- read.csv("data/precomputed/ledger/ward7_reg.csv", encoding = "UTF-8")
led[led$model_id %in% c("ols", "rf500"), c("model", "test_MAE", "test_RMSE", "test_R2")]   # test results of both models from the ledger

# 7. Optional, on your computer: the vivid and bartMan analyses of the chapter -------
#    install.packages(c("vivid", "bartMan", "dbarts"))  (these packages do not run in the browser)
if (requireNamespace("vivid", quietly = TRUE) && requireNamespace("bartMan", quietly = TRUE) &&
    requireNamespace("dbarts", quietly = TRUE)) {
  library(vivid); library(dbarts); library(bartMan)
  trn <- ltr[, c(feats, "time")]                     # vivi() uses every other column as a feature
  set.seed(2026)
  viv <- vivi(data = trn, fit = rf, response = "time", gridSize = 10,
              importanceType = "agnostic", numPerm = 4)
  print(round(viv, 2))                               # importance (diagonal), unnormalised H (days)
  print(viviHeatmap(mat = viv, angle = 45))
  print(viviNetwork(mat = viv))
  print(pdpPairs(data = trn, fit = rf, response = "time",
                 vars = c("ph.ecog", "ph.karno", "wt.loss"), gridSize = 10, nIce = 50))
  set.seed(2026)
  bart_fit <- bart(x.train = as.matrix(ltr[feats]), y.train = ltr$time,
                   x.test = as.matrix(lte[feats]), ntree = 80L, nskip = 200L,
                   ndpost = 200L, keeptrees = TRUE, verbose = FALSE, seed = 2026)
  print(round(c(BART_test_MAE = mae(lte$time, colMeans(bart_fit$yhat.test))), 1))
  trees <- extractTreeData(model = bart_fit, data = data.frame(ltr[feats], y = ltr$time))  # a minute or two
  vsupMat <- viviBartMatrix(trees = trees, type = "vsup", metric = "propMean", metricError = "CV")
  print(lapply(vsupMat, round, 3))                   # inclusion proportions and their CV
  print(viviBartPlot(vsupMat, label = "CV"))
} else {
  message("Step 7 needs the R packages vivid, bartMan and dbarts: run it on your computer.")
}
