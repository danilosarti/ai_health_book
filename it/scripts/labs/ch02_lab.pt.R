# Laboratório do capítulo 2
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Carregar os dados ------------------------------------------------------
icu <- read.csv("data/icu_aki.csv")
hb  <- read.csv("data/homebeat.csv")
icu$aki <- icu$delta_creatinine_48h >= 0.3       # critério KDIGO: um desfecho de Bernoulli

# 2. Bernoulli e binomial ---------------------------------------------------
theta <- mean(icu$aki)                            # EMV da probabilidade de LRA
theta
dbinom(0:20, size = 20, prob = theta)             # distribuição dos casos de LRA em uma ala de 20
1 - pbinom(14, size = 20, prob = theta)           # P(15 casos ou mais)

# 3. Modelo normal para a frequência cardíaca de repouso --------------------
mu <- mean(hb$resting_hr); s <- sd(hb$resting_hr)
c(model = 1 - pnorm(80, mu, s), data = mean(hb$resting_hr > 80))   # P(FC > 80)
hist(hb$resting_hr, breaks = 30, freq = FALSE, main = "", xlab = "Frequência cardíaca de repouso (bpm)")
curve(dnorm(x, mu, s), add = TRUE, lwd = 3)

# 4. Verossimilhança e EMV em uma grade -------------------------------------
theta_grid <- seq(0.01, 0.99, by = 0.001)
loglik <- sapply(theta_grid, function(th) sum(dbinom(icu$aki, 1, th, log = TRUE)))
theta_grid[which.max(loglik)]                    # igual a mean(icu$aki), a menos da grade
plot(theta_grid, exp(loglik - max(loglik)), type = "l", lwd = 2,
     xlab = "theta", ylab = "Verossimilhança (pico = 1)")

# 5. Bayes por contagem: fragilidade dada uma FC alta (HomeBeat) ------------
frail <- hb$frailty >= 0.5; fast <- hb$resting_hr > 85
table(frail, fast)
mean(frail[fast])                                 # probabilidade a posteriori, contada diretamente
pF <- mean(frail); a <- mean(fast[frail]); b <- mean(fast[!frail])
a * pF / (a * pF + b * (1 - pF))                  # probabilidade a posteriori, pelo teorema de Bayes

# 6. Um teste diagnóstico para os pacientes desta noite ---------------------
se <- 0.85; sp <- 0.80
prior <- c(Tomas = 0.07, Aoife = 0.49, Priya = 0.99)   # riscos do modelo do Capítulo 5
p_pos    <- prior * se + (1 - prior) * (1 - sp)
post_pos <- prior * se / p_pos
post_neg <- prior * (1 - se) / (1 - p_pos)

# 7. Perda esperada, limiar e valor da informação ---------------------------
C_FP <- 10; C_FN <- 30
t_star <- C_FP / (C_FP + C_FN)
best <- function(q) pmin((1 - q) * C_FP, q * C_FN)  # perda esperada da melhor ação
loss_no_test   <- best(prior)
loss_with_test <- p_pos * best(post_pos) + (1 - p_pos) * best(post_neg)
round(data.frame(prior, post_pos, post_neg, loss_no_test, loss_with_test,
                 value_of_test = loss_no_test - loss_with_test), 3)

# 8. Entropia ----------------------------------------------------------------
H <- function(q) -q * log2(q) - (1 - q) * log2(1 - q)
round(H(prior), 3)
