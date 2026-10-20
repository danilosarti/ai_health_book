# Laboratorio del capitolo 13
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Reparto 7: stima ingenua contro stima aggiustata (verità -0.28) --------
icu <- read.csv("data/icu_aki.csv")
naive    <- lm(delta_creatinine_48h ~ protocol_adherence, data = icu)
adjusted <- lm(delta_creatinine_48h ~ protocol_adherence + sofa, data = icu)
round(c(naive = coef(naive)[["protocol_adherence"]],
        adjusted = coef(adjusted)[["protocol_adherence"]], truth = -0.28), 3)

# 2. Reparto 7, versione estesa: un insieme valido, un mediatore, un collider
dag <- read.csv("data/ward7_dag.csv")
sets <- c(valid    = "sofa + baseline_creat",
          mediator = "sofa + nephrotox",
          collider = "sofa + creat_tests")
round(sapply(sets, function(s) coef(lm(as.formula(
  paste("delta_creatinine_48h ~ protocol_adherence +", s)), data = dag))[["protocol_adherence"]]), 3)

# 3. Studio ONCO-1: il riferimento randomizzato -----------------------------
onco  <- read.csv("data/onco1.csv")
trial <- with(onco, mean(response_percent[treatment == "chemo"]) -
                    mean(response_percent[treatment == "no_chemo"]))

# 4. ONCO-1 nella pratica corrente: stima ingenua e regressione -------------
obs <- read.csv("data/onco1_routine.csv")
naive_obs <- with(obs, mean(response_percent[chemo == 1]) - mean(response_percent[chemo == 0]))
reg <- coef(lm(response_percent ~ chemo + patient_age + tumor_grade + performance_score + gene_05,
               data = obs))[["chemo"]]

# 5. Punteggio di propensione e ponderazione per l'inverso della probabilità
ps <- fitted(glm(chemo ~ patient_age + tumor_grade + performance_score + gene_05,
                 family = binomial, data = obs))
w  <- ifelse(obs$chemo == 1, 1 / ps, 1 / (1 - ps))
t1 <- obs$chemo == 1
ipw <- weighted.mean(obs$response_percent[t1], w[t1]) -
       weighted.mean(obs$response_percent[!t1], w[!t1])

# 6. Appaiamento greedy 1:1 al vicino più prossimo sul logit del punteggio --
set.seed(1)
lp <- qlogis(ps); caliper <- 0.2 * sd(lp)
pool <- which(!t1); pairs <- NULL
for (i in sample(which(t1))) {
  d <- abs(lp[pool] - lp[i]); j <- which.min(d)
  if (length(pool) && d[j] <= caliper) { pairs <- rbind(pairs, c(i, pool[j])); pool <- pool[-j] }
}
matched <- mean(obs$response_percent[pairs[, 1]] - obs$response_percent[pairs[, 2]])

round(c(trial = trial, naive = naive_obs, regression = reg, ipw = ipw, matching = matched), 1)

# 7. Bilanciamento: differenze medie standardizzate prima e dopo la ponderazione
smd <- function(x, w = rep(1, length(x))) {
  (weighted.mean(x[t1], w[t1]) - weighted.mean(x[!t1], w[!t1])) /
    sqrt((var(x[t1]) + var(x[!t1])) / 2)
}
covs <- data.frame(age = obs$patient_age, G3 = as.numeric(obs$tumor_grade == "G3"),
                   performance = obs$performance_score, gene_05 = obs$gene_05)
round(rbind(before = sapply(covs, smd), after_ipw = sapply(covs, smd, w = w)), 3)

# 8. Sovrapposizione dei punteggi di propensione ----------------------------
hist(ps[t1], breaks = seq(0, 1, 0.05), col = "#1f6f8b80", border = "white",
     main = "Punteggi di propensione", xlab = "Punteggio di propensione stimato")
hist(ps[!t1], breaks = seq(0, 1, 0.05), col = "#b35c0080", border = "white", add = TRUE)
legend("top", fill = c("#1f6f8b80", "#b35c0080"), legend = c("chemioterapia", "nessuna chemioterapia"), bty = "n")
