# Laboratório do capítulo 9
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Carregar os dados ---------------------------------------------------
icu   <- read.csv("data/icu_aki_missing.csv")   # o que a ala tem
full  <- read.csv("data/icu_aki.csv")           # versão completa (apenas na simulação)
split <- read.csv("data/splits/ward7_split.csv")
icu$set  <- split$set[match(icu$patient_id, split$patient_id)]
full$set <- split$set[match(full$patient_id, split$patient_id)]

# 2. Quanto está faltando, e onde? --------------------------------------
round(100 * colMeans(is.na(icu)), 1)       # % ausente por coluna
sum(!complete.cases(icu))                  # pacientes com pelo menos uma lacuna
table(lactate = is.na(icu$lactate), creat = is.na(icu$baseline_creat))

# 3. Quem tem lacunas? --------------------------------------------------
tapply(icu$sofa, is.na(icu$lactate), mean)
tapply(icu$age,  is.na(icu$baseline_creat), mean)
summary(glm(is.na(lactate) ~ sofa + age + map_min, data = icu,
            family = binomial))$coefficients
# Os dados não conseguem mostrar a parte MNAR: compare com a verdade (apenas na simulação)
c(verdadeira = mean(full$baseline_creat), registrada = mean(icu$baseline_creat, na.rm = TRUE))

# 4. Predição com lacunas, sem vazamento ---------------------------------
train <- icu[icu$set == "train", ]
test  <- icu[icu$set == "test", ]
gappy <- c("lactate", "baseline_creat", "albumin")
fill  <- sapply(train[gappy], mean, na.rm = TRUE)    # aprendido APENAS com as linhas de TREINO
prep  <- function(d) {
  for (v in gappy) {
    d[[paste0(v, "_missing")]] <- as.integer(is.na(d[[v]]))   # indicador de ausência
    d[[v]][is.na(d[[v]])] <- fill[v]                          # média do treino
  }
  d
}
tr <- prep(train); te <- prep(test)
preds <- c("site", "age", "sofa", "lactate", "map_min", "baseline_creat",
           "protocol_adherence", "bun", "crp", "albumin")   # P10, como nos Capítulos 4-8
f_base <- reformulate(preds, response = "delta_creatinine_48h")
f_ind  <- reformulate(c(preds, paste0(gappy, "_missing")), response = "delta_creatinine_48h")

m_cc   <- lm(f_base, data = train)     # apenas casos completos (lm descarta as linhas com lacuna)
m_imp  <- lm(f_ind,  data = tr)        # imputação pela média + indicadores
m_full <- lm(f_base, data = full[full$set == "train", ])   # se nada estivesse faltando

mae <- function(y, yhat) mean(abs(y - yhat))
y_te   <- test$delta_creatinine_48h
p_full <- predict(m_full, full[full$set == "test", ])
p_imp  <- predict(m_imp, te)
p_cc   <- predict(m_cc, newdata = test)
ok     <- !is.na(p_cc)                 # pacientes de teste que o modelo de casos completos consegue prever
c(n_train_cc = nrow(model.frame(m_cc)), n_test = nrow(test), cc_can_predict = sum(ok))
# todos os 250 pacientes de teste
round(c(complete_data = mae(y_te, p_full), impute_indicator = mae(y_te, p_imp)), 3)
# só os pacientes de teste que o modelo de casos completos consegue prever
round(c(n = sum(ok), complete_case = mae(y_te[ok], p_cc[ok]),
        complete_data = mae(y_te[ok], p_full[ok]), impute_indicator = mae(y_te[ok], p_imp[ok])), 3)
round(coef(m_imp)[paste0(gappy, "_missing")], 3)   # a própria lacuna carrega informação?

# 5. Imputação múltipla ------------------------------------------------
# O mice não roda no navegador. O registro de modelos do livro guarda o resultado da predição
# (mice, m = 20, modelo de imputação ajustado apenas com as linhas de treino):
w7 <- read.csv("data/precomputed/ledger/ward7_reg.csv")
w7[w7$model_id %in% c("ols", "ols_imp_mean", "ols_mi"), c("model", "n_test", "test_MAE", "test_RMSE")]
# Para inferência, no seu computador (install.packages("mice")):
# library(mice); icu$site <- factor(icu$site)
# imp <- mice(icu[, c(preds, "delta_creatinine_48h")], m = 20, seed = 2026)
# summary(pool(with(imp, lm(delta_creatinine_48h ~ site + age + sofa + lactate + map_min +
#   baseline_creat + protocol_adherence + bun + crp + albumin))))
