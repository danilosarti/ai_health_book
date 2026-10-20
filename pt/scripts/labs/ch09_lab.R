# Chapter 9 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Load the data -------------------------------------------------------
icu   <- read.csv("data/icu_aki_missing.csv")   # what the ward has
full  <- read.csv("data/icu_aki.csv")           # complete version (simulation only)
split <- read.csv("data/splits/ward7_split.csv")
icu$set  <- split$set[match(icu$patient_id, split$patient_id)]
full$set <- split$set[match(full$patient_id, split$patient_id)]

# 2. How much is missing, and where? ------------------------------------
round(100 * colMeans(is.na(icu)), 1)       # % missing per column
sum(!complete.cases(icu))                  # patients with at least one gap
table(lactate = is.na(icu$lactate), creat = is.na(icu$baseline_creat))

# 3. Who has gaps? ------------------------------------------------------
tapply(icu$sofa, is.na(icu$lactate), mean)
tapply(icu$age,  is.na(icu$baseline_creat), mean)
summary(glm(is.na(lactate) ~ sofa + age + map_min, data = icu,
            family = binomial))$coefficients
# The data cannot show the MNAR part: compare with the truth (simulation only)
c(true = mean(full$baseline_creat), recorded = mean(icu$baseline_creat, na.rm = TRUE))

# 4. Prediction with gaps, without leakage --------------------------------
train <- icu[icu$set == "train", ]
test  <- icu[icu$set == "test", ]
gappy <- c("lactate", "baseline_creat", "albumin")
fill  <- sapply(train[gappy], mean, na.rm = TRUE)    # learned on TRAINING rows only
prep  <- function(d) {
  for (v in gappy) {
    d[[paste0(v, "_missing")]] <- as.integer(is.na(d[[v]]))   # missing indicator
    d[[v]][is.na(d[[v]])] <- fill[v]                          # training mean
  }
  d
}
tr <- prep(train); te <- prep(test)
preds <- c("site", "age", "sofa", "lactate", "map_min", "baseline_creat",
           "protocol_adherence", "bun", "crp", "albumin")   # P10, as in Chapters 4-8
f_base <- reformulate(preds, response = "delta_creatinine_48h")
f_ind  <- reformulate(c(preds, paste0(gappy, "_missing")), response = "delta_creatinine_48h")

m_cc   <- lm(f_base, data = train)     # complete cases only (lm drops rows with a gap)
m_imp  <- lm(f_ind,  data = tr)        # mean imputation + indicators
m_full <- lm(f_base, data = full[full$set == "train", ])   # if nothing were missing

mae <- function(y, yhat) mean(abs(y - yhat))
y_te   <- test$delta_creatinine_48h
p_full <- predict(m_full, full[full$set == "test", ])
p_imp  <- predict(m_imp, te)
p_cc   <- predict(m_cc, newdata = test)
ok     <- !is.na(p_cc)                 # test patients the complete-case model can predict
c(n_train_cc = nrow(model.frame(m_cc)), n_test = nrow(test), cc_can_predict = sum(ok))
# all 250 test patients
round(c(complete_data = mae(y_te, p_full), impute_indicator = mae(y_te, p_imp)), 3)
# only the test patients the complete-case model can predict
round(c(n = sum(ok), complete_case = mae(y_te[ok], p_cc[ok]),
        complete_data = mae(y_te[ok], p_full[ok]), impute_indicator = mae(y_te[ok], p_imp[ok])), 3)
round(coef(m_imp)[paste0(gappy, "_missing")], 3)   # does the gap itself carry information?

# 5. Multiple imputation -------------------------------------------------
# mice does not run in the browser. The book's model ledger holds the prediction result
# (mice, m = 20, imputation model fitted on the training rows only):
w7 <- read.csv("data/precomputed/ledger/ward7_reg.csv")
w7[w7$model_id %in% c("ols", "ols_imp_mean", "ols_mi"), c("model", "n_test", "test_MAE", "test_RMSE")]
# For inference, on your computer (install.packages("mice")):
# library(mice); icu$site <- factor(icu$site)
# imp <- mice(icu[, c(preds, "delta_creatinine_48h")], m = 20, seed = 2026)
# summary(pool(with(imp, lm(delta_creatinine_48h ~ site + age + sofa + lactate + map_min +
#   baseline_creat + protocol_adherence + bun + crp + albumin))))
