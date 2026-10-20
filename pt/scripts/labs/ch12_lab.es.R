# Laboratorio del capítulo 12
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 0. Paquetes y datos ------------------------------------------------------
library(randomForest)
lung  <- read.csv("data/lung_sim_complete.csv")
lung$male <- as.integer(lung$sex == "male")
lsp   <- read.csv("data/splits/lung_split.csv")
ltr   <- lung[lsp$set == "train", ];  lte <- lung[lsp$set == "test", ]
feats <- c("age", "male", "ph.ecog", "ph.karno", "wt.loss")
mae   <- function(y, yhat) mean(abs(y - yhat))

# 1. Dos modelos de los días de seguimiento (recuerda: se ignora la censura) --
lin <- lm(time ~ age + male + ph.ecog + ph.karno + wt.loss, data = ltr)
set.seed(2026)
rf  <- randomForest(time ~ age + male + ph.ecog + ph.karno + wt.loss, data = ltr, ntree = 300)
c(linear = mae(lte$time, predict(lin, lte)), forest = mae(lte$time, predict(rf, lte)))

# 2. Importancia por permutación para cualquier modelo -----------------------
perm_importance <- function(predict_fun, data, y, features, n_shuffles = 10) {
  base <- mae(y, predict_fun(data))
  sapply(features, function(f) {
    mean(replicate(n_shuffles, {
      d <- data; d[[f]] <- sample(d[[f]])
      mae(y, predict_fun(d))
    })) - base                                    # barajado menos original
  })
}
set.seed(2026)
round(perm_importance(function(d) predict(lin, d), lte, lte$time, feats), 1)
round(perm_importance(function(d) predict(rf, d), lte, lte$time, feats), 1)

# 3. Curvas ICE y dependencia parcial ------------------------------------------
ice_curves <- function(predict_fun, data, feature, grid) {
  sapply(grid, function(g) { d <- data; d[[feature]] <- g; predict_fun(d) })  # pacientes x rejilla
}
grid <- seq(1, 25, by = 1)
ice  <- ice_curves(function(d) predict(rf, d), ltr, "wt.loss", grid)
matplot(grid, t(ice - ice[, 1]), type = "l", lty = 1, col = "#00000020",
        xlab = "Pérdida de peso (kg)", ylab = "Cambio de los días predichos (ICE centrada)")
lines(grid, colMeans(ice - ice[, 1]), lwd = 4, col = "#1f6f8b")

# 4. H de Friedman para un par (en 60 pacientes de entrenamiento sorteados) ---
pd_at <- function(predict_fun, data, cols, points) {
  apply(points[, cols, drop = FALSE], 1, function(v) {
    d <- data; for (k in seq_along(cols)) d[[cols[k]]] <- v[k]; mean(predict_fun(d))
  })
}
h_stat <- function(predict_fun, data, a, b) {
  fa <- pd_at(predict_fun, data, a, data);  fa <- fa - mean(fa)
  fb <- pd_at(predict_fun, data, b, data);  fb <- fb - mean(fb)
  fab <- pd_at(predict_fun, data, c(a, b), data); fab <- fab - mean(fab)
  c(H = sqrt(sum((fab - fa - fb)^2) / sum(fab^2)), H_days = sqrt(mean((fab - fa - fb)^2)))
}
set.seed(2026)
sub <- ltr[sample(nrow(ltr), 60), ]
round(h_stat(function(d) predict(rf, d), sub, "ph.ecog", "wt.loss"), 3)
round(h_stat(function(d) predict(rf, d), sub, "age", "male"), 3)

# 5. Valores de Shapley exactos para pacientes de la UCI Sala 7: el modelo OLS del Capítulo 4
#    (P10: los diez predictores de los Capítulos 4 y 6; el hospital cuenta como un solo jugador)
icu   <- read.csv("data/icu_aki.csv")
split <- read.csv("data/splits/ward7_split.csv")
train <- icu[split$set == "train", ];  test <- icu[split$set == "test", ]
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
f10 <- c("site", "age", "sofa", "lactate", "map_min", "baseline_creat",
         "protocol_adherence", "bun", "crp", "albumin")
ols7 <- lm(reformulate(f10, "delta_creatinine_48h"), data = train)
round(mae(test$delta_creatinine_48h, predict(ols7, test)), 4)   # 0,1413, el valor del registro de modelos (Capítulo 4)
shapley_exact <- function(predict_fun, x, background, features) {
  p <- length(features)
  masks <- as.matrix(expand.grid(rep(list(0:1), p)))
  v <- apply(masks, 1, function(m) {
    d <- background
    for (j in which(m == 1)) d[[features[j]]] <- x[[features[j]]]
    mean(predict_fun(d))
  })
  key <- apply(masks, 1, paste, collapse = "")
  phi <- setNames(numeric(p), features)
  for (r in seq_len(nrow(masks))) {
    m <- masks[r, ]; s <- sum(m)
    for (j in which(m == 0)) {
      m2 <- m; m2[j] <- 1
      phi[j] <- phi[j] + factorial(s) * factorial(p - s - 1) / factorial(p) *
        (v[match(paste(m2, collapse = ""), key)] - v[r])
    }
  }
  c(baseline = v[1], phi, prediction = v[nrow(masks)])
}
set.seed(2026)
background <- train[sample(nrow(train), 100), ]
sh <- sapply(1:3, function(i) shapley_exact(function(d) predict(ols7, d), tonight[i, ], background, f10))
colnames(sh) <- tonight$name
round(sh, 3)     # cada columna: referencia + contribuciones = predicción

# 6. El bosque aleatorio del Capítulo 6 (500 árboles, scikit-learn) ---------------
#    R no puede reajustar exactamente el bosque de scikit-learn, y sus valores de Shapley
#    exactos requieren cerca de un millón de predicciones por paciente: lee los calculados
#    de antemano por scripts/precompute/ch12_explain.py (referencia: 200 pacientes de entrenamiento).
shf <- read.csv("data/precomputed/ch12/ward7_shapley.csv", encoding = "UTF-8")
shf <- shf[shf$model == "Random forest", ]
round(xtabs(phi ~ feature + name, data = shf), 3)
unique(shf[, c("name", "baseline", "prediction")])   # predicciones del bosque 0,29 / -0,09 / 0,83, como en el registro de modelos
led <- read.csv("data/precomputed/ledger/ward7_reg.csv", encoding = "UTF-8")
led[led$model_id %in% c("ols", "rf500"), c("model", "test_MAE", "test_RMSE", "test_R2")]   # resultados de prueba de ambos modelos, del registro de modelos

# 7. Opcional, en tu equipo: los análisis de vivid y bartMan del capítulo -------
#    install.packages(c("vivid", "bartMan", "dbarts"))  (estos paquetes no se ejecutan en el navegador)
if (requireNamespace("vivid", quietly = TRUE) && requireNamespace("bartMan", quietly = TRUE) &&
    requireNamespace("dbarts", quietly = TRUE)) {
  library(vivid); library(dbarts); library(bartMan)
  trn <- ltr[, c(feats, "time")]                     # vivi() usa todas las demás columnas como características
  set.seed(2026)
  viv <- vivi(data = trn, fit = rf, response = "time", gridSize = 10,
              importanceType = "agnostic", numPerm = 4)
  print(round(viv, 2))                               # importancia (diagonal), H no normalizado (días)
  print(viviHeatmap(mat = viv, angle = 45))
  print(viviNetwork(mat = viv))
  print(pdpPairs(data = trn, fit = rf, response = "time",
                 vars = c("ph.ecog", "ph.karno", "wt.loss"), gridSize = 10, nIce = 50))
  set.seed(2026)
  bart_fit <- bart(x.train = as.matrix(ltr[feats]), y.train = ltr$time,
                   x.test = as.matrix(lte[feats]), ntree = 80L, nskip = 200L,
                   ndpost = 200L, keeptrees = TRUE, verbose = FALSE, seed = 2026)
  print(round(c(BART_test_MAE = mae(lte$time, colMeans(bart_fit$yhat.test))), 1))
  trees <- extractTreeData(model = bart_fit, data = data.frame(ltr[feats], y = ltr$time))  # uno o dos minutos
  vsupMat <- viviBartMatrix(trees = trees, type = "vsup", metric = "propMean", metricError = "CV")
  print(lapply(vsupMat, round, 3))                   # proporciones de inclusión y su CV
  print(viviBartPlot(vsupMat, label = "CV"))
} else {
  message("El paso 7 necesita los paquetes de R vivid, bartMan y dbarts: ejecútalo en tu equipo.")
}
