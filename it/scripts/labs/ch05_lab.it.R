# Laboratorio del capitolo 5
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Dati ed esito binario ---------------------------------------------------
icu     <- read.csv("data/icu_aki.csv")
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
icu$aki <- as.integer(icu$delta_creatinine_48h >= 0.3)     # criterio KDIGO
mean(icu$aki)                                               # prevalenza

# 2. Suddivisione addestramento / test: quella condivisa del libro (Capitolo 3)
split <- read.csv("data/splits/ward7_split.csv")
icu   <- merge(icu, split, by = "patient_id")
train <- icu[icu$set == "train", ]
test  <- icu[icu$set == "test", ]

# 3. Regressione logistica e odds ratio ---------------------------------------
fit <- glm(aki ~ sofa + lactate + map_min + baseline_creat + protocol_adherence,
           family = binomial, data = train)
summary(fit)
round(exp(cbind(OR = coef(fit), confint.default(fit))), 2)  # IC 95% di Wald

# 4. Probabilità previste sull'insieme di test -------------------------------
p <- predict(fit, newdata = test, type = "response")
y <- test$aki

# 5. Matrici di confusione a due soglie ---------------------------------------
confusion <- function(y, p, t) {
  table(flagged = factor(p >= t, c(TRUE, FALSE), c("sì", "no")),
        aki     = factor(y == 1, c(TRUE, FALSE), c("sì", "no")))
}
confusion(y, p, 0.50)
confusion(y, p, 0.25)

# 6. Curva ROC e AUC (formula basata sui ranghi) -------------------------------
auc <- function(y, p) {
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
auc(y, p)
ths <- c(Inf, sort(unique(p), decreasing = TRUE))
roc <- cbind(fpr = sapply(ths, function(t) mean(p[y == 0] >= t)),
             tpr = sapply(ths, function(t) mean(p[y == 1] >= t)))
plot(roc, type = "l", lwd = 2, xlab = "1 - specificità", ylab = "Sensibilità")
abline(0, 1, lty = 2)

# 7. Calibrazione ---------------------------------------------------------------
groups <- cut(p, quantile(p, 0:10 / 10), include.lowest = TRUE)
cal <- data.frame(predicted = tapply(p, groups, mean), observed = tapply(y, groups, mean))
round(cal, 2)
plot(cal, xlim = c(0, 1), ylim = c(0, 1), pch = 16); abline(0, 1, lty = 2)
mean((p - y)^2)                                             # punteggio di Brier

# 8. Beneficio netto (curva decisionale ad alcune soglie) -----------------------
net_benefit <- function(y, p, t) {
  n <- length(y)
  sum(p >= t & y == 1) / n - sum(p >= t & y == 0) / n * t / (1 - t)
}
thresholds <- c(0.10, 0.25, 0.50)
round(data.frame(t = thresholds,
                 model = sapply(thresholds, function(t) net_benefit(y, p, t)),
                 treat_all = mean(y) - (1 - mean(y)) * thresholds / (1 - thresholds)), 3)

# 9. I pazienti di stasera --------------------------------------------------------
data.frame(name = tonight$name,
           risk = round(predict(fit, newdata = tonight, type = "response"), 2),
           bundle_at_0.25 = predict(fit, newdata = tonight, type = "response") >= 0.25)
