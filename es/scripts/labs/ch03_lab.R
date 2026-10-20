# Chapter 3 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Load the data and the book-wide split ------------------------------------
icu     <- read.csv("data/icu_aki.csv")
split   <- read.csv("data/splits/ward7_split.csv")
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
stopifnot(identical(icu$patient_id, split$patient_id))
train <- icu[split$set == "train", ]
fold  <- split$fold[split$set == "train"]
test  <- icu[split$set == "test", ]                  # locked until step 6
feat  <- c("sofa", "baseline_creat")

# 2. Metrics -------------------------------------------------------------------
mae  <- function(y, yhat) mean(abs(y - yhat))
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
r2   <- function(y, yhat, ybar) 1 - sum((y - yhat)^2) / sum((y - ybar)^2)

# 3. k-nearest neighbours, written out in full ---------------------------------
knn_predict <- function(train_x, train_y, new_x, k) {
  mu <- colMeans(train_x); s <- apply(train_x, 2, sd)  # learned from training rows only
  zt <- scale(as.matrix(train_x), mu, s)
  zn <- scale(as.matrix(new_x), mu, s)
  apply(zn, 1, function(z) {
    d <- sqrt(colSums((t(zt) - z)^2))
    mean(train_y[order(d)[1:k]])
  })
}

# 4. Five-fold cross-validation on the training set (test set untouched) ------
cv_mae <- function(predict_fun) {
  mean(sapply(1:5, function(v) {
    fit <- train[fold != v, ]; val <- train[fold == v, ]
    mae(val$delta_creatinine_48h, predict_fun(fit, val))
  }))
}
baselines <- c(
  mean      = cv_mae(function(fit, new) rep(mean(fit$delta_creatinine_48h), nrow(new))),
  sofa_line = cv_mae(function(fit, new) predict(lm(delta_creatinine_48h ~ sofa, data = fit), new)),
  two_line  = cv_mae(function(fit, new) predict(lm(delta_creatinine_48h ~ sofa + baseline_creat, data = fit), new)))
round(baselines, 3)

ks <- c(1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150)
cv_knn <- sapply(ks, function(k)
  cv_mae(function(fit, new) knn_predict(fit[feat], fit$delta_creatinine_48h, new[feat], k)))
k_best <- ks[which.min(cv_knn)]
data.frame(k = ks, cv_mae = round(cv_knn, 3))

# 5. Plot the tuning curve -----------------------------------------------------
plot(ks, cv_knn, type = "b", log = "x", pch = 16, col = "#1f6f8b",
     xlab = "Number of neighbours k (log scale)", ylab = "Cross-validated MAE (mg/dL)")
abline(h = baselines, lty = c(2, 3, 4))
legend("topright", c("kNN", names(baselines)), lty = c(1, 2, 3, 4), bty = "n")

# 6. Open the test set ONCE: refit on all training patients --------------------
pred_test <- knn_predict(train[feat], train$delta_creatinine_48h, test[feat], k_best)
round(c(k = k_best,
        test_MAE  = mae(test$delta_creatinine_48h, pred_test),
        test_RMSE = rmse(test$delta_creatinine_48h, pred_test),
        test_R2   = r2(test$delta_creatinine_48h, pred_test, mean(train$delta_creatinine_48h))), 3)

# 7. kNN with all ten predictors (P10; site as dummy columns), k = 1..150 by CV --
X10 <- function(d) model.matrix(~ site + age + sofa + lactate + map_min + baseline_creat +
                                  protocol_adherence + bun + crp + albumin, d)[, -1]
knn_all_k <- function(train_x, train_y, new_x, kmax) {   # predictions for k = 1..kmax at once
  mu <- colMeans(train_x); s <- apply(train_x, 2, sd)
  zt <- scale(train_x, mu, s); zn <- scale(new_x, mu, s)
  t(apply(zn, 1, function(z) {
    o <- order(sqrt(colSums((t(zt) - z)^2)))[1:kmax]
    cumsum(train_y[o]) / (1:kmax)
  }))
}
cv10 <- rowMeans(sapply(1:5, function(v) {
  fit <- train[fold != v, ]; val <- train[fold == v, ]
  colMeans(abs(knn_all_k(X10(fit), fit$delta_creatinine_48h, X10(val), 150) - val$delta_creatinine_48h))
}))
k10 <- which.min(cv10)
pred10 <- knn_all_k(X10(train), train$delta_creatinine_48h, X10(test), k10)[, k10]
round(c(k = k10, cv_MAE = min(cv10), test_MAE = mae(test$delta_creatinine_48h, pred10)), 3)

# 8. Deploy: predictions for tonight's patients --------------------------------
data.frame(name = tonight$name, sofa = tonight$sofa, baseline_creat = tonight$baseline_creat,
           predicted_change = round(knn_predict(train[feat], train$delta_creatinine_48h, tonight[feat], k_best), 2))
