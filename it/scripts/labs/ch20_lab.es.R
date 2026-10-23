# Laboratorio del capítulo 20
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. Reparametrización y la penalización KL -----------------------------------
kl_normal <- function(mu, sigma) 0.5 * (mu^2 + sigma^2 - log(sigma^2) - 1)
kl_normal(0, 1)        # 0: idéntica a la normal estándar
kl_normal(2, 1)        # código lejos del origen
kl_normal(0, 0.1)      # código encogido hasta un punto
set.seed(19)
z <- 1.2 + 0.5 * rnorm(10000)          # z = mu + sigma * eps
c(mean = mean(z), sd = sd(z))

# 2. Difusión directa sobre una radiografía -----------------------------------
px   <- read.csv("data/pneumonia_subset.csv")
x0   <- as.numeric(px[px$label == 1, grep("^px", names(px))][1, ]) / 127.5 - 1
beta <- seq(1e-4, 0.02, length.out = 1000)
abar <- cumprod(1 - beta)
which(abar < 0.5)[1]                   # primer paso con más ruido que señal
set.seed(19)
eps   <- rnorm(length(x0))
steps <- c(0, 50, 100, 250, 500, 1000)
op <- par(mfrow = c(1, 6), mar = c(0.5, 0.5, 2, 0.5))
for (t in steps) {
  a  <- if (t == 0) 1 else abar[t]
  xt <- sqrt(a) * x0 + sqrt(1 - a) * eps
  image(matrix(pmin(pmax(xt, -1), 1), 28, 28)[, 28:1], col = gray.colors(64, 0, 1),
        axes = FALSE, main = paste("t =", t), asp = 1)
}
par(op)

# 3. Una cópula gaussiana para HomeBeat, desde cero ---------------------------
hb   <- read.csv("data/homebeat.csv")
vars <- c("age", "baseline_inflammation", "frailty", "resting_hr", "hrv",
          "skin_temp", "activity_index", "stress_score")
spl  <- read.csv("data/splits/homebeat_split.csv")   # la partición fija de HomeBeat del libro (como en el Capítulo 8)
spl  <- spl[order(spl$perm_order), ]
real <- hb[match(spl$participant_id[spl$set == "train"], hb$participant_id), vars]  # el generador ve estos
hold <- hb[match(spl$participant_id[spl$set == "test"], hb$participant_id), vars]   # personas reales nuevas, nunca vistas
set.seed(2026)
n <- nrow(real)
scores <- sapply(real, function(v) qnorm(rank(v, ties.method = "average") / (n + 1)))  # paso 1
R <- cor(scores)                                                                       # paso 2
Z <- matrix(rnorm(400 * length(vars)), ncol = length(vars)) %*% chol(R)                # paso 3
U <- pnorm(Z)                                                                          # paso 4
fake <- as.data.frame(sapply(seq_along(vars), function(j)
  quantile(real[[j]], U[, j], type = 8, names = FALSE)))
names(fake) <- vars

# 4. Fidelidad: marginales y relaciones ---------------------------------------
round(rbind(real = sapply(real, mean), synthetic = sapply(fake, mean)), 2)
round(c(real = cor(real$resting_hr, real$hrv), synthetic = cor(fake$resting_hr, fake$hrv)), 2)
gap <- cor(fake) - cor(real)
mean(abs(gap[upper.tri(gap)]))         # diferencia media de correlación

# 5. Privacidad: distancia al registro real de entrenamiento más cercano ------
mu <- colMeans(real); s <- apply(real, 2, sd)
std <- function(d) scale(d, center = mu, scale = s)
dcr <- function(a, b) {                # para cada fila de a, distancia a la fila más cercana de b
  A <- std(a); B <- std(b)
  apply(A, 1, function(r) sqrt(min(colSums((t(B) - r)^2))))
}
d_hold <- dcr(hold, real)
d_fake <- dcr(fake, real)
thr <- quantile(d_hold, 0.05)
c(median_new_real = median(d_hold), median_synthetic = median(d_fake),
  share_too_close = mean(d_fake < thr))

# 6. Utilidad: entrenar con lo sintético, probar con lo real ------------------
f <- stress_score ~ age + baseline_inflammation + frailty + resting_hr + hrv + skin_temp + activity_index
rmse <- function(m) sqrt(mean((hold$stress_score - predict(m, hold))^2))
c(train_on_real = rmse(lm(f, real)), train_on_synthetic = rmse(lm(f, fake)))
