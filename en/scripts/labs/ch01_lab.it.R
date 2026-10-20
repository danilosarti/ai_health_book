# Laboratorio del capitolo 1
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Caricare i dati -------------------------------------------------------
icu     <- read.csv("data/icu_aki.csv")
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
dim(icu)
str(icu)

# 2. Descrivere -------------------------------------------------------------
summary(icu$delta_creatinine_48h)
icu$aki <- icu$delta_creatinine_48h >= 0.3          # criterio KDIGO
mean(icu$aki)                                        # rischio di base 'senza modello'
round(tapply(icu$aki, icu$site, mean), 2)            # varia da un ospedale all'altro?

# 3. Suddivisione addestramento / test: quella fissa del libro (1.000 / 250) --
split <- read.csv("data/splits/ward7_split.csv")
stopifnot(identical(split$patient_id, icu$patient_id))
train <- icu[split$set == "train", ]
test  <- icu[split$set == "test", ]

# 4. Due modelli dell'esito -------------------------------------------------
# (a) il modello della media: la stessa previsione per tutti
pred_avg  <- mean(train$delta_creatinine_48h)
# (b) una retta in funzione del SOFA (il Capitolo 4 spiega lm in dettaglio)
line_sofa <- lm(delta_creatinine_48h ~ sofa, data = train)
pred_line <- predict(line_sofa, newdata = test)

mae <- function(y, yhat) mean(abs(y - yhat))
c(average = mae(test$delta_creatinine_48h, pred_avg),
  sofa_line = mae(test$delta_creatinine_48h, pred_line))

# 5. Impiego: previsioni per i pazienti di stasera --------------------------
data.frame(name = tonight$name,
           sofa = tonight$sofa,
           predicted_change = round(predict(line_sofa, newdata = tonight), 2))

# 6. Grafico ------------------------------------------------------------------
plot(jitter(icu$sofa), icu$delta_creatinine_48h, pch = 16, col = "#1f6f8b40",
     xlab = "Punteggio SOFA", ylab = "Variazione della creatinina in 48 h (mg/dL)")
abline(line_sofa, lwd = 3)
abline(h = 0.3, lty = 2)
