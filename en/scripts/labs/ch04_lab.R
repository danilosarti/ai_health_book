# Chapter 4 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Ward 7: load and split (the shared book-wide split) ---------------------
icu     <- read.csv("data/icu_aki.csv", stringsAsFactors = TRUE)
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
tonight$site <- factor(tonight$site, levels = levels(icu$site))
split_w7 <- read.csv("data/splits/ward7_split.csv")
set_w7 <- split_w7$set[match(icu$patient_id, split_w7$patient_id)]
train <- icu[set_w7 == "train", ]
test  <- icu[set_w7 == "test", ]

# 2. Multiple linear regression ------------------------------------------------
f <- delta_creatinine_48h ~ site + age + sofa + lactate + map_min +
       baseline_creat + protocol_adherence + bun + crp + albumin
fit <- lm(f, data = train)
summary(fit)
round(confint(fit), 4)
drop1(fit, test = "F")              # each term tested last: order does not matter

# 3. Diagnostics ----------------------------------------------------------------
par(mfrow = c(2, 2)); plot(fit); par(mfrow = c(1, 1))
cd <- cooks.distance(fit)
c(max_cook = max(cd), flagged = sum(cd > 4 / nrow(train)))
X <- model.matrix(fit)[, -1]
vif <- sapply(colnames(X), function(j)
  1 / (1 - summary(lm(X[, j] ~ X[, -match(j, colnames(X))]))$r.squared))
round(vif, 2)

# 4. Test-set error against the 'average' baseline -------------------------------
mae  <- function(y, yhat) mean(abs(y - yhat))
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
y_te <- test$delta_creatinine_48h
pred <- predict(fit, newdata = test)
rbind(average = c(MAE = mae(y_te, mean(train$delta_creatinine_48h)),
                  RMSE = rmse(y_te, mean(train$delta_creatinine_48h))),
      linear  = c(MAE = mae(y_te, pred), RMSE = rmse(y_te, pred)))

# 5. Tonight's patients, with 95% prediction intervals ----------------------------
cbind(tonight["name"], round(predict(fit, newdata = tonight, interval = "prediction"), 2))

# 6. ONCO-1: OLS, ridge, elastic net and lasso -------------------------------------
if (!requireNamespace("glmnet", quietly = TRUE)) install.packages("glmnet")
library(glmnet)
onco <- read.csv("data/onco1.csv")
drop <- c("patient_id", "high_response", "baseline_tumor_mm", "post_tumor_mm",
          "gene_08", "gene_14", "gene_19")      # identifier, outcome copies, leaky genes
d <- onco[, setdiff(names(onco), drop)]
d$treatment <- factor(d$treatment, levels = c("no_chemo", "chemo"))
Xo <- model.matrix(response_percent ~ ., data = d)[, -1]
yo <- d$response_percent
split_o <- read.csv("data/splits/onco1_split.csv")
split_o <- split_o[match(onco$patient_id, split_o$patient_id), ]
tr <- which(split_o$set == "train")
foldid <- split_o$fold[tr]                         # shared folds, the same for every model
Z <- scale(Xo, colMeans(Xo[tr, ]), apply(Xo[tr, ], 2, sd))   # standardised with training means and SDs
cvs <- list(ridge = cv.glmnet(Z[tr, ], yo[tr], alpha = 0,   foldid = foldid, standardize = FALSE,
                              lambda.min.ratio = 1e-6),  # long grid: the ridge path reaches OLS
            enet  = cv.glmnet(Z[tr, ], yo[tr], alpha = 0.8, foldid = foldid, standardize = FALSE),
            lasso = cv.glmnet(Z[tr, ], yo[tr], alpha = 1,   foldid = foldid, standardize = FALSE))
ols <- lm(yo[tr] ~ Z[tr, ])
test_err <- function(p) c(MAE = mae(yo[-tr], p), RMSE = rmse(yo[-tr], p))
round(cbind(OLS = test_err(cbind(1, Z[-tr, ]) %*% coef(ols)),
            sapply(cvs, function(cv) test_err(predict(cv, Z[-tr, ], s = "lambda.min"))),
            sapply(cvs, function(cv) test_err(predict(cv, Z[-tr, ], s = "lambda.1se")))), 3)
# columns: OLS, then ridge / elastic net (alpha = 0.8, chosen by CV) / lasso at lambda.min and lambda.1se
plot(cvs$lasso)                                   # CV curve with lambda.min and lambda.1se
plot(cvs$lasso$glmnet.fit, xvar = "lambda")       # coefficient paths
b <- as.matrix(coef(cvs$lasso, s = "lambda.1se"))
round(b[b[, 1] != 0, , drop = FALSE], 3)          # predictors kept by the lasso (per SD)
