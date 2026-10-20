# Laboratorio del capitolo 2
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Caricare i dati --------------------------------------------------------
icu <- read.csv("data/icu_aki.csv")
hb  <- read.csv("data/homebeat.csv")
icu$aki <- icu$delta_creatinine_48h >= 0.3       # criterio KDIGO: un esito di Bernoulli

# 2. Bernoulli e binomiale --------------------------------------------------
theta <- mean(icu$aki)                            # stima di massima verosimiglianza della probabilità di AKI
theta
dbinom(0:20, size = 20, prob = theta)             # distribuzione dei casi di AKI in un reparto di 20
1 - pbinom(14, size = 20, prob = theta)           # P(15 casi o più)

# 3. Modello normale per la frequenza cardiaca a riposo ---------------------
mu <- mean(hb$resting_hr); s <- sd(hb$resting_hr)
c(model = 1 - pnorm(80, mu, s), data = mean(hb$resting_hr > 80))   # P(FC > 80)
hist(hb$resting_hr, breaks = 30, freq = FALSE, main = "", xlab = "Frequenza cardiaca a riposo (bpm)")
curve(dnorm(x, mu, s), add = TRUE, lwd = 3)

# 4. Verosimiglianza e stima di massima verosimiglianza su una griglia ------
theta_grid <- seq(0.01, 0.99, by = 0.001)
loglik <- sapply(theta_grid, function(th) sum(dbinom(icu$aki, 1, th, log = TRUE)))
theta_grid[which.max(loglik)]                    # uguale a mean(icu$aki), a meno della griglia
plot(theta_grid, exp(loglik - max(loglik)), type = "l", lwd = 2,
     xlab = "theta", ylab = "Verosimiglianza (picco = 1)")

# 5. Bayes contando: fragilità data una FC elevata (HomeBeat) ---------------
frail <- hb$frailty >= 0.5; fast <- hb$resting_hr > 85
table(frail, fast)
mean(frail[fast])                                 # probabilità a posteriori, contata direttamente
pF <- mean(frail); a <- mean(fast[frail]); b <- mean(fast[!frail])
a * pF / (a * pF + b * (1 - pF))                  # probabilità a posteriori, con il teorema di Bayes

# 6. Un test diagnostico per i pazienti di stasera --------------------------
se <- 0.85; sp <- 0.80
prior <- c(Tomas = 0.07, Aoife = 0.49, Priya = 0.99)   # rischi del modello del Capitolo 5
p_pos    <- prior * se + (1 - prior) * (1 - sp)
post_pos <- prior * se / p_pos
post_neg <- prior * (1 - se) / (1 - p_pos)

# 7. Perdita attesa, soglia e valore dell'informazione ----------------------
C_FP <- 10; C_FN <- 30
t_star <- C_FP / (C_FP + C_FN)
best <- function(q) pmin((1 - q) * C_FP, q * C_FN)  # perdita attesa dell'azione migliore
loss_no_test   <- best(prior)
loss_with_test <- p_pos * best(post_pos) + (1 - p_pos) * best(post_neg)
round(data.frame(prior, post_pos, post_neg, loss_no_test, loss_with_test,
                 value_of_test = loss_no_test - loss_with_test), 3)

# 8. Entropia ----------------------------------------------------------------
H <- function(q) -q * log2(q) - (1 - q) * log2(1 - q)
round(H(prior), 3)
