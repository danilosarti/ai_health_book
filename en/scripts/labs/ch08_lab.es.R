# Laboratorio del capítulo 8
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. Aprendizaje conjugado de la tasa de LRA de ICU_D ------------------------------------
icu <- read.csv("data/icu_aki.csv")
akiD <- icu$delta_creatinine_48h[icu$site == "ICU_D"] >= 0.3
a0 <- 55; b0 <- 45                                    # a priori tomada de los otros hospitales
for (n in c(0, 10, 50, 313)) {
  k <- sum(akiD[seq_len(n)])
  ci <- qbeta(c(0.025, 0.975), a0 + k, b0 + n - k)
  cat(sprintf("n = %3d  k = %3d  media a posteriori %.3f  ICr 95%% %.3f-%.3f\n",
              n, k, (a0 + k) / (a0 + b0 + n), ci[1], ci[2]))
}

# 2. Distribución a posteriori en rejilla de la pendiente de la fragilidad (intercepto y sigma fijos) -
hb <- read.csv("data/homebeat.csv")
x <- hb$frailty - mean(hb$frailty); y <- hb$resting_hr
grid <- seq(20, 70, by = 0.05)
logpost <- sapply(grid, function(b1) sum(dnorm(y, mean(y) + b1 * x, 11.2, log = TRUE))) +
  dnorm(grid, 0, 20, log = TRUE)
post <- exp(logpost - max(logpost)); post <- post / sum(post)
c(media_a_posteriori = sum(grid * post), minimos_cuadrados = unname(coef(lm(y ~ x))[2]))

# 3. Muestras de Stan: resúmenes y la lección de la confusión -------------------------------
simple   <- read.csv("data/precomputed/ch08/homebeat_simple_draws.csv")
adjusted <- read.csv("data/precomputed/ch08/homebeat_adjusted_draws.csv")
round(tapply(simple$b_frailty, simple$chain, mean), 2)    # ¿coinciden las dos cadenas?
rbind(simple   = c(quantile(simple$b_frailty, c(0.025, 0.5, 0.975)), p_pos = mean(simple$b_frailty > 0)),
      ajustado = c(quantile(adjusted$b_frailty, c(0.025, 0.5, 0.975)), p_pos = mean(adjusted$b_frailty > 0)))

# 4. Lena: intervalo de credibilidad de la media frente a intervalo de predicción -----------------
mu <- with(adjusted, b0 + b_frailty * 0.55 + b_age * 72 + b_infl * 4.1)
set.seed(8)
y_new <- rnorm(length(mu), mu, adjusted$sigma)
rbind(media_de_personas_como_Lena = quantile(mu, c(0.025, 0.975)),
      proxima_lectura_de_Lena     = quantile(y_new, c(0.025, 0.975)))
mean(y_new > 88)                                       # probabilidad de una lectura por encima de 88 lpm

# 5. Esta noche en la UCI Sala 7: riesgo de LRA con incertidumbre --------------------------------
risk <- read.csv("data/precomputed/ch08/ward7_tonight_risk_draws.csv")
summ <- function(r, t) c(media = mean(r), quantile(r, c(0.025, 0.975)), p_por_encima_de_t = mean(r > t))
round(t(sapply(risk, summ, t = 0.25)), 3)
round(t(sapply(risk, summ, t = 0.09)), 3)

# 6. BART frente a bosques y boosting (errores honestos en el conjunto de prueba) ---------------------
# BART y los demás modelos se ajustaron una vez, fuera del navegador, en las particiones comunes;
# sus errores de prueba se leen del registro de modelos del libro.
hbl <- read.csv("data/precomputed/ledger/homebeat.csv")
hbl[hbl$role %in% c("reference", "canonical"), c("model", "test_MAE", "test_RMSE", "test_R2")]
w7 <- read.csv("data/precomputed/ledger/ward7_reg.csv")
w7[w7$chapter <= 8 & w7$role %in% c("reference", "canonical", "teaching"),
   c("chapter", "model", "test_MAE", "test_RMSE")]
read.csv("data/precomputed/ch08/bart_comparison.csv")[, c("data", "model", "train_RMSE", "coverage95")]
