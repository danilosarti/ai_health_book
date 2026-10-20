# Laboratorio del capítulo 1
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. Cargar los datos -----------------------------------------------------
icu     <- read.csv("data/icu_aki.csv")
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
dim(icu)
str(icu)

# 2. Describir --------------------------------------------------------------
summary(icu$delta_creatinine_48h)
icu$aki <- icu$delta_creatinine_48h >= 0.3          # criterio KDIGO
mean(icu$aki)                                        # riesgo de referencia 'sin modelo'
round(tapply(icu$aki, icu$site, mean), 2)            # ¿difiere entre hospitales?

# 3. División entrenamiento / prueba: la división fija del libro (1000 / 250) --
split <- read.csv("data/splits/ward7_split.csv")
stopifnot(identical(split$patient_id, icu$patient_id))
train <- icu[split$set == "train", ]
test  <- icu[split$set == "test", ]

# 4. Dos modelos del desenlace ---------------------------------------------
# (a) el modelo de la media: la misma predicción para todos
pred_avg  <- mean(train$delta_creatinine_48h)
# (b) una recta en función del SOFA (el Capítulo 4 explica lm en detalle)
line_sofa <- lm(delta_creatinine_48h ~ sofa, data = train)
pred_line <- predict(line_sofa, newdata = test)

mae <- function(y, yhat) mean(abs(y - yhat))
c(average = mae(test$delta_creatinine_48h, pred_avg),
  sofa_line = mae(test$delta_creatinine_48h, pred_line))

# 5. Desplegar: predicciones para los pacientes de esta noche -----------------
data.frame(name = tonight$name,
           sofa = tonight$sofa,
           predicted_change = round(predict(line_sofa, newdata = tonight), 2))

# 6. Gráfico ------------------------------------------------------------------
plot(jitter(icu$sofa), icu$delta_creatinine_48h, pch = 16, col = "#1f6f8b40",
     xlab = "Puntuación SOFA", ylab = "Cambio de la creatinina en 48 h (mg/dL)")
abline(line_sofa, lwd = 3)
abline(h = 0.3, lty = 2)
