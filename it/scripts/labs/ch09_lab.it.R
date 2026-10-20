# Laboratorio del capitolo 9
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Caricare i dati -------------------------------------------------------
icu   <- read.csv("data/icu_aki_missing.csv")   # ciò che il reparto ha
full  <- read.csv("data/icu_aki.csv")           # versione completa (solo nella simulazione)
split <- read.csv("data/splits/ward7_split.csv")
icu$set  <- split$set[match(icu$patient_id, split$patient_id)]
full$set <- split$set[match(full$patient_id, split$patient_id)]

# 2. Quanto manca, e dove? ------------------------------------------------
round(100 * colMeans(is.na(icu)), 1)       # % di valori mancanti per colonna
sum(!complete.cases(icu))                  # pazienti con almeno una lacuna
table(lactate = is.na(icu$lactate), creat = is.na(icu$baseline_creat))

# 3. Chi ha lacune? -------------------------------------------------------
tapply(icu$sofa, is.na(icu$lactate), mean)
tapply(icu$age,  is.na(icu$baseline_creat), mean)
summary(glm(is.na(lactate) ~ sofa + age + map_min, data = icu,
            family = binomial))$coefficients
# I dati non possono mostrare la parte MNAR: confronta con la verità (solo nella simulazione)
c(vera = mean(full$baseline_creat), registrata = mean(icu$baseline_creat, na.rm = TRUE))

# 4. Previsione con lacune, senza fughe di informazione --------------------
train <- icu[icu$set == "train", ]
test  <- icu[icu$set == "test", ]
gappy <- c("lactate", "baseline_creat", "albumin")
fill  <- sapply(train[gappy], mean, na.rm = TRUE)    # appreso SOLO dalle righe di ADDESTRAMENTO
prep  <- function(d) {
  for (v in gappy) {
    d[[paste0(v, "_missing")]] <- as.integer(is.na(d[[v]]))   # indicatore di dato mancante
    d[[v]][is.na(d[[v]])] <- fill[v]                          # media dell'addestramento
  }
  d
}
tr <- prep(train); te <- prep(test)
preds <- c("site", "age", "sofa", "lactate", "map_min", "baseline_creat",
           "protocol_adherence", "bun", "crp", "albumin")   # P10, come nei Capitoli 4-8
f_base <- reformulate(preds, response = "delta_creatinine_48h")
f_ind  <- reformulate(c(preds, paste0(gappy, "_missing")), response = "delta_creatinine_48h")

m_cc   <- lm(f_base, data = train)     # solo casi completi (lm scarta le righe con lacune)
m_imp  <- lm(f_ind,  data = tr)        # imputazione con la media + indicatori
m_full <- lm(f_base, data = full[full$set == "train", ])   # se non mancasse nulla

mae <- function(y, yhat) mean(abs(y - yhat))
y_te   <- test$delta_creatinine_48h
p_full <- predict(m_full, full[full$set == "test", ])
p_imp  <- predict(m_imp, te)
p_cc   <- predict(m_cc, newdata = test)
ok     <- !is.na(p_cc)                 # pazienti di test che il modello sui casi completi può prevedere
c(n_train_cc = nrow(model.frame(m_cc)), n_test = nrow(test), cc_can_predict = sum(ok))
# tutti i 250 pazienti di test
round(c(complete_data = mae(y_te, p_full), impute_indicator = mae(y_te, p_imp)), 3)
# solo i pazienti di test che il modello sui casi completi può prevedere
round(c(n = sum(ok), complete_case = mae(y_te[ok], p_cc[ok]),
        complete_data = mae(y_te[ok], p_full[ok]), impute_indicator = mae(y_te[ok], p_imp[ok])), 3)
round(coef(m_imp)[paste0(gappy, "_missing")], 3)   # la lacuna stessa porta informazione?

# 5. Imputazione multipla ----------------------------------------------
# mice non funziona nel browser. Il registro dei modelli del libro contiene il risultato della
# previsione (mice, m = 20, modello di imputazione stimato solo sulle righe di addestramento):
w7 <- read.csv("data/precomputed/ledger/ward7_reg.csv")
w7[w7$model_id %in% c("ols", "ols_imp_mean", "ols_mi"), c("model", "n_test", "test_MAE", "test_RMSE")]
# Per l'inferenza, sul tuo computer (install.packages("mice")):
# library(mice); icu$site <- factor(icu$site)
# imp <- mice(icu[, c(preds, "delta_creatinine_48h")], m = 20, seed = 2026)
# summary(pool(with(imp, lm(delta_creatinine_48h ~ site + age + sofa + lactate + map_min +
#   baseline_creat + protocol_adherence + bun + crp + albumin))))
