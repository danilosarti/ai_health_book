# Laboratorio del capítulo 9
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. Cargar los datos ----------------------------------------------------
icu   <- read.csv("data/icu_aki_missing.csv")   # lo que tiene la unidad
full  <- read.csv("data/icu_aki.csv")           # versión completa (solo en la simulación)
split <- read.csv("data/splits/ward7_split.csv")
icu$set  <- split$set[match(icu$patient_id, split$patient_id)]
full$set <- split$set[match(full$patient_id, split$patient_id)]

# 2. ¿Cuánto falta, y dónde? ---------------------------------------------
round(100 * colMeans(is.na(icu)), 1)       # % faltante por columna
sum(!complete.cases(icu))                  # pacientes con al menos un hueco
table(lactate = is.na(icu$lactate), creat = is.na(icu$baseline_creat))

# 3. ¿Quién tiene huecos? ------------------------------------------------
tapply(icu$sofa, is.na(icu$lactate), mean)
tapply(icu$age,  is.na(icu$baseline_creat), mean)
summary(glm(is.na(lactate) ~ sofa + age + map_min, data = icu,
            family = binomial))$coefficients
# Los datos no pueden mostrar la parte MNAR: compara con la verdad (solo en la simulación)
c(verdadera = mean(full$baseline_creat), registrada = mean(icu$baseline_creat, na.rm = TRUE))

# 4. Predicción con huecos, sin fuga de información ----------------------
train <- icu[icu$set == "train", ]
test  <- icu[icu$set == "test", ]
gappy <- c("lactate", "baseline_creat", "albumin")
fill  <- sapply(train[gappy], mean, na.rm = TRUE)    # aprendido SOLO con las filas de ENTRENAMIENTO
prep  <- function(d) {
  for (v in gappy) {
    d[[paste0(v, "_missing")]] <- as.integer(is.na(d[[v]]))   # indicador de ausencia
    d[[v]][is.na(d[[v]])] <- fill[v]                          # media del entrenamiento
  }
  d
}
tr <- prep(train); te <- prep(test)
preds <- c("site", "age", "sofa", "lactate", "map_min", "baseline_creat",
           "protocol_adherence", "bun", "crp", "albumin")   # P10, como en los Capítulos 4-8
f_base <- reformulate(preds, response = "delta_creatinine_48h")
f_ind  <- reformulate(c(preds, paste0(gappy, "_missing")), response = "delta_creatinine_48h")

m_cc   <- lm(f_base, data = train)     # solo casos completos (lm descarta las filas con hueco)
m_imp  <- lm(f_ind,  data = tr)        # imputación con la media + indicadores
m_full <- lm(f_base, data = full[full$set == "train", ])   # si no faltara nada

mae <- function(y, yhat) mean(abs(y - yhat))
y_te   <- test$delta_creatinine_48h
p_full <- predict(m_full, full[full$set == "test", ])
p_imp  <- predict(m_imp, te)
p_cc   <- predict(m_cc, newdata = test)
ok     <- !is.na(p_cc)                 # pacientes de prueba que el modelo de casos completos puede predecir
c(n_train_cc = nrow(model.frame(m_cc)), n_test = nrow(test), cc_can_predict = sum(ok))
# los 250 pacientes de prueba
round(c(complete_data = mae(y_te, p_full), impute_indicator = mae(y_te, p_imp)), 3)
# solo los pacientes de prueba que el modelo de casos completos puede predecir
round(c(n = sum(ok), complete_case = mae(y_te[ok], p_cc[ok]),
        complete_data = mae(y_te[ok], p_full[ok]), impute_indicator = mae(y_te[ok], p_imp[ok])), 3)
round(coef(m_imp)[paste0(gappy, "_missing")], 3)   # ¿lleva información el propio hueco?

# 5. Imputación múltiple -------------------------------------------------
# mice no funciona en el navegador. El registro de modelos del libro guarda el resultado de la predicción
# (mice, m = 20, modelo de imputación ajustado solo con las filas de entrenamiento):
w7 <- read.csv("data/precomputed/ledger/ward7_reg.csv")
w7[w7$model_id %in% c("ols", "ols_imp_mean", "ols_mi"), c("model", "n_test", "test_MAE", "test_RMSE")]
# Para la inferencia, en tu equipo (install.packages("mice")):
# library(mice); icu$site <- factor(icu$site)
# imp <- mice(icu[, c(preds, "delta_creatinine_48h")], m = 20, seed = 2026)
# summary(pool(with(imp, lm(delta_creatinine_48h ~ site + age + sofa + lactate + map_min +
#   baseline_creat + protocol_adherence + bun + crp + albumin))))
