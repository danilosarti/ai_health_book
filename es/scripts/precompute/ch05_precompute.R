# =====================================================================
# Chapter 5: precomputed results (classification, thresholds, calibration)
# Usage (from the book folder): Rscript scripts/precompute/ch05_precompute.R
# Writes small CSV files to data/precomputed/ch05/. Needs base R + glmnet.
# =====================================================================
out <- "data/precomputed/ch05"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

# ---- 1. Ward 7: AKI yes/no -------------------------------------------
icu <- read.csv("data/icu_aki.csv")
icu$aki <- as.integer(icu$delta_creatinine_48h >= 0.3)        # KDIGO criterion
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")

split <- read.csv("data/splits/ward7_split.csv")               # shared split (Chapter 3)
icu <- merge(icu, split, by = "patient_id", sort = FALSE)
icu <- icu[order(icu$patient_id), ]
train <- icu[icu$set == "train", ]
test  <- icu[icu$set == "test", ]

f_model <- aki ~ sofa + lactate + map_min + baseline_creat + protocol_adherence
fit      <- glm(f_model, family = binomial, data = train)
fit_sofa <- glm(aki ~ sofa, family = binomial, data = train)

pred <- data.frame(patient_id = test$patient_id, site = test$site, aki = test$aki,
                   p_model = round(predict(fit, test, type = "response"), 4),
                   p_sofa  = round(predict(fit_sofa, test, type = "response"), 4))
write.csv(pred, file.path(out, "ward7_test_predictions.csv"), row.names = FALSE)

co <- summary(fit)$coefficients
coefs <- data.frame(term = rownames(co), estimate = co[, 1], se = co[, 2],
                    or = exp(co[, 1]), or_low = exp(co[, 1] - 1.96 * co[, 2]),
                    or_high = exp(co[, 1] + 1.96 * co[, 2]), row.names = NULL)
write.csv(coefs, file.path(out, "ward7_coefficients.csv"), row.names = FALSE)

tn <- data.frame(name = tonight$name,
                 p_model = round(predict(fit, tonight, type = "response"), 3),
                 p_sofa  = round(predict(fit_sofa, tonight, type = "response"), 3))
write.csv(tn, file.path(out, "ward7_tonight_predictions.csv"), row.names = FALSE, fileEncoding = "UTF-8")

# Transport: develop on ICU_A–C, apply to ICU_D (a hospital with more AKI)
dev <- icu[icu$site != "ICU_D", ]; ext <- icu[icu$site == "ICU_D", ]
fit_dev <- glm(f_model, family = binomial, data = dev)
ext_pred <- data.frame(patient_id = ext$patient_id, aki = ext$aki,
                       p_model = round(predict(fit_dev, ext, type = "response"), 4))
write.csv(ext_pred, file.path(out, "ward7_siteD_predictions.csv"), row.names = FALSE)

# ---- 2. ONCO-1: high response ------------------------------------------
onco <- read.csv("data/onco1.csv")
tab <- as.data.frame.matrix(table(onco$treatment, onco$high_response))
sep <- data.frame(treatment = rownames(tab), non_responders = tab[, "0"], responders = tab[, "1"])
write.csv(sep, file.path(out, "onco_treatment_table.csv"), row.names = FALSE)

chemo <- onco[onco$treatment == "chemo", ]
leaky <- c("gene_08", "gene_14", "gene_19")                   # copies of the outcome (see sim_onco1.R)
genes <- setdiff(grep("^gene_", names(chemo), value = TRUE), leaky)
chemo[genes] <- scale(chemo[genes])                           # genes per standard deviation
chemo$dose10 <- chemo$dose_intensity * 10                     # dose per 0.1 unit
chemo$age10  <- chemo$patient_age / 10                        # age per 10 years

osplit <- read.csv("data/splits/onco1_split.csv")              # shared split (Chapter 3)
chemo <- merge(chemo, osplit, by = "patient_id", sort = FALSE)
chemo <- chemo[order(chemo$patient_id), ]
ctr <- chemo[chemo$set == "train", ]; cte <- chemo[chemo$set == "test", ]

auc <- function(y, p) { r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }

f_small <- high_response ~ dose10 + tumor_grade + performance_score + age10 + gene_01 + gene_05 + gene_12
glm_small <- glm(f_small, family = binomial, data = ctr)
cs <- summary(glm_small)$coefficients
write.csv(data.frame(term = rownames(cs), estimate = cs[, 1], se = cs[, 2], or = exp(cs[, 1]),
                     row.names = NULL), file.path(out, "onco_chemo_glm.csv"), row.names = FALSE)

f_all <- as.formula(paste("high_response ~ dose10 + tumor_grade + performance_score + age10 +",
                          paste(genes, collapse = " + ")))
glm_all <- suppressWarnings(glm(f_all, family = binomial, data = ctr))

X_tr <- model.matrix(f_all, ctr)[, -1]; X_te <- model.matrix(f_all, cte)[, -1]
library(glmnet)
res <- list(); lasso_coef <- NULL
p_list <- list("Logistic, 7 chosen predictors" = predict(glm_small, cte, type = "response"),
               "Logistic, all 51 predictors"   = suppressWarnings(predict(glm_all, cte, type = "response")))
for (a in c(1, 0.5, 0)) {
  cv <- cv.glmnet(X_tr, ctr$high_response, family = "binomial", alpha = a,
                  foldid = ctr$fold)                          # the shared 5 folds
  lab <- c("1" = "Lasso logistic", "0.5" = "Elastic net logistic (alpha = 0.5)", "0" = "Ridge logistic")[as.character(a)]
  p_list[[lab]] <- as.numeric(predict(cv, X_te, s = "lambda.min", type = "response"))
  b <- as.matrix(coef(cv, s = "lambda.min"))
  res[[lab]] <- sum(b[-1, 1] != 0)
  if (a == 1) lasso_coef <- data.frame(term = rownames(b)[-1], estimate = b[-1, 1])
}
cmp <- data.frame(model = names(p_list),
                  auc = sapply(p_list, function(p) auc(cte$high_response, p)),
                  brier = sapply(p_list, function(p) mean((p - cte$high_response)^2)),
                  nonzero = c(7 + 1, ncol(X_tr), unlist(res)), row.names = NULL)
cmp$nonzero[1] <- ncol(model.matrix(f_small, ctr)) - 1
write.csv(cmp, file.path(out, "onco_penalised_comparison.csv"), row.names = FALSE)
lasso_coef <- lasso_coef[lasso_coef$estimate != 0, ]
lasso_coef <- lasso_coef[order(-abs(lasso_coef$estimate)), ]
write.csv(lasso_coef, file.path(out, "onco_lasso_coefficients.csv"), row.names = FALSE)

# (The two invented "new patients" #0042 / #0917 were removed: the book now uses two real ONCO-1
#  test patients, P5440 and P9541, whose predictions come from the model ledger,
#  data/precomputed/ledger/onco1_patients.csv.)
print(cmp); print(head(lasso_coef, 12)); print(cs)
cat("done\n")
