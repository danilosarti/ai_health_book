# Laboratorio del capítulo 2
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. Cargar los datos ---------------------------------------------------------
icu <- read.csv("data/icu_aki.csv")
hb  <- read.csv("data/homebeat.csv")
icu$aki <- icu$delta_creatinine_48h >= 0.3       # criterio KDIGO: un desenlace de Bernoulli

# 2. Bernoulli y binomial ---------------------------------------------------
theta <- mean(icu$aki)                            # EMV de la probabilidad de LRA
theta
dbinom(0:20, size = 20, prob = theta)             # distribución de los casos de LRA en una unidad de 20
1 - pbinom(14, size = 20, prob = theta)           # P(15 casos o más)

# 3. Modelo normal para la frecuencia cardíaca en reposo --------------------
mu <- mean(hb$resting_hr); s <- sd(hb$resting_hr)
c(model = 1 - pnorm(80, mu, s), data = mean(hb$resting_hr > 80))   # P(FC > 80)
hist(hb$resting_hr, breaks = 30, freq = FALSE, main = "", xlab = "Frecuencia cardíaca en reposo (lpm)")
curve(dnorm(x, mu, s), add = TRUE, lwd = 3)

# 4. Verosimilitud y EMV en una rejilla -------------------------------------
theta_grid <- seq(0.01, 0.99, by = 0.001)
loglik <- sapply(theta_grid, function(th) sum(dbinom(icu$aki, 1, th, log = TRUE)))
theta_grid[which.max(loglik)]                    # igual a mean(icu$aki), salvo por la rejilla
plot(theta_grid, exp(loglik - max(loglik)), type = "l", lwd = 2,
     xlab = "theta", ylab = "Verosimilitud (máximo = 1)")

# 5. Bayes contando: fragilidad dada una FC alta (HomeBeat) -----------------
frail <- hb$frailty >= 0.5; fast <- hb$resting_hr > 85
table(frail, fast)
mean(frail[fast])                                 # probabilidad a posteriori, contada directamente
pF <- mean(frail); a <- mean(fast[frail]); b <- mean(fast[!frail])
a * pF / (a * pF + b * (1 - pF))                  # probabilidad a posteriori, con el teorema de Bayes

# 6. Una prueba diagnóstica para los pacientes de esta noche ----------------
se <- 0.85; sp <- 0.80
prior <- c(Tomas = 0.07, Aoife = 0.49, Priya = 0.99)   # riesgos del modelo del Capítulo 5
p_pos    <- prior * se + (1 - prior) * (1 - sp)
post_pos <- prior * se / p_pos
post_neg <- prior * (1 - se) / (1 - p_pos)

# 7. Pérdida esperada, umbral y valor de la información ---------------------
C_FP <- 10; C_FN <- 30
t_star <- C_FP / (C_FP + C_FN)
best <- function(q) pmin((1 - q) * C_FP, q * C_FN)  # pérdida esperada de la mejor acción
loss_no_test   <- best(prior)
loss_with_test <- p_pos * best(post_pos) + (1 - p_pos) * best(post_neg)
round(data.frame(prior, post_pos, post_neg, loss_no_test, loss_with_test,
                 value_of_test = loss_no_test - loss_with_test), 3)

# 8. Entropía ----------------------------------------------------------------
H <- function(q) -q * log2(q) - (1 - q) * log2(1 - q)
round(H(prior), 3)
