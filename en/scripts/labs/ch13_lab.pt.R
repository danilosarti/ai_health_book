# Laboratório do capítulo 13
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. UTI Ala 7: ingênuo versus ajustado (verdade -0.28) ---------------------
icu <- read.csv("data/icu_aki.csv")
naive    <- lm(delta_creatinine_48h ~ protocol_adherence, data = icu)
adjusted <- lm(delta_creatinine_48h ~ protocol_adherence + sofa, data = icu)
round(c(naive = coef(naive)[["protocol_adherence"]],
        adjusted = coef(adjusted)[["protocol_adherence"]], truth = -0.28), 3)

# 2. UTI Ala 7, versão ampliada: um conjunto válido, um mediador e um colisor
dag <- read.csv("data/ward7_dag.csv")
sets <- c(valid    = "sofa + baseline_creat",
          mediator = "sofa + nephrotox",
          collider = "sofa + creat_tests")
round(sapply(sets, function(s) coef(lm(as.formula(
  paste("delta_creatinine_48h ~ protocol_adherence +", s)), data = dag))[["protocol_adherence"]]), 3)

# 3. Ensaio ONCO-1: a referência randomizada --------------------------------
onco  <- read.csv("data/onco1.csv")
trial <- with(onco, mean(response_percent[treatment == "chemo"]) -
                    mean(response_percent[treatment == "no_chemo"]))

# 4. ONCO-1 na prática de rotina: ingênuo e ajuste por regressão ------------
obs <- read.csv("data/onco1_routine.csv")
naive_obs <- with(obs, mean(response_percent[chemo == 1]) - mean(response_percent[chemo == 0]))
reg <- coef(lm(response_percent ~ chemo + patient_age + tumor_grade + performance_score + gene_05,
               data = obs))[["chemo"]]

# 5. Escore de propensão e ponderação pelo inverso da probabilidade ---------
ps <- fitted(glm(chemo ~ patient_age + tumor_grade + performance_score + gene_05,
                 family = binomial, data = obs))
w  <- ifelse(obs$chemo == 1, 1 / ps, 1 / (1 - ps))
t1 <- obs$chemo == 1
ipw <- weighted.mean(obs$response_percent[t1], w[t1]) -
       weighted.mean(obs$response_percent[!t1], w[!t1])

# 6. Pareamento guloso 1:1 pelo vizinho mais próximo no logito do escore ----
set.seed(1)
lp <- qlogis(ps); caliper <- 0.2 * sd(lp)
pool <- which(!t1); pairs <- NULL
for (i in sample(which(t1))) {
  d <- abs(lp[pool] - lp[i]); j <- which.min(d)
  if (length(pool) && d[j] <= caliper) { pairs <- rbind(pairs, c(i, pool[j])); pool <- pool[-j] }
}
matched <- mean(obs$response_percent[pairs[, 1]] - obs$response_percent[pairs[, 2]])

round(c(trial = trial, naive = naive_obs, regression = reg, ipw = ipw, matching = matched), 1)

# 7. Equilíbrio: diferenças médias padronizadas antes e depois da ponderação
smd <- function(x, w = rep(1, length(x))) {
  (weighted.mean(x[t1], w[t1]) - weighted.mean(x[!t1], w[!t1])) /
    sqrt((var(x[t1]) + var(x[!t1])) / 2)
}
covs <- data.frame(age = obs$patient_age, G3 = as.numeric(obs$tumor_grade == "G3"),
                   performance = obs$performance_score, gene_05 = obs$gene_05)
round(rbind(before = sapply(covs, smd), after_ipw = sapply(covs, smd, w = w)), 3)

# 8. Sobreposição dos escores de propensão ----------------------------------
hist(ps[t1], breaks = seq(0, 1, 0.05), col = "#1f6f8b80", border = "white",
     main = "Escores de propensão", xlab = "Escore de propensão estimado")
hist(ps[!t1], breaks = seq(0, 1, 0.05), col = "#b35c0080", border = "white", add = TRUE)
legend("top", fill = c("#1f6f8b80", "#b35c0080"), legend = c("quimioterapia", "sem quimioterapia"), bty = "n")
