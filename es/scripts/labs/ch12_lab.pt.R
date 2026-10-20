# Laboratório do capítulo 12
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 0. Pacotes e dados -------------------------------------------------------
library(randomForest)
lung  <- read.csv("data/lung_sim_complete.csv")
lung$male <- as.integer(lung$sex == "male")
lsp   <- read.csv("data/splits/lung_split.csv")
ltr   <- lung[lsp$set == "train", ];  lte <- lung[lsp$set == "test", ]
feats <- c("age", "male", "ph.ecog", "ph.karno", "wt.loss")
mae   <- function(y, yhat) mean(abs(y - yhat))

# 1. Dois modelos dos dias de seguimento (lembre-se: a censura é ignorada) ---
lin <- lm(time ~ age + male + ph.ecog + ph.karno + wt.loss, data = ltr)
set.seed(2026)
rf  <- randomForest(time ~ age + male + ph.ecog + ph.karno + wt.loss, data = ltr, ntree = 300)
c(linear = mae(lte$time, predict(lin, lte)), forest = mae(lte$time, predict(rf, lte)))

# 2. Importância por permutação para qualquer modelo -------------------------
perm_importance <- function(predict_fun, data, y, features, n_shuffles = 10) {
  base <- mae(y, predict_fun(data))
  sapply(features, function(f) {
    mean(replicate(n_shuffles, {
      d <- data; d[[f]] <- sample(d[[f]])
      mae(y, predict_fun(d))
    })) - base                                    # embaralhado menos original
  })
}
set.seed(2026)
round(perm_importance(function(d) predict(lin, d), lte, lte$time, feats), 1)
round(perm_importance(function(d) predict(rf, d), lte, lte$time, feats), 1)

# 3. Curvas ICE e dependência parcial ------------------------------------------
ice_curves <- function(predict_fun, data, feature, grid) {
  sapply(grid, function(g) { d <- data; d[[feature]] <- g; predict_fun(d) })  # pacientes x grade
}
grid <- seq(1, 25, by = 1)
ice  <- ice_curves(function(d) predict(rf, d), ltr, "wt.loss", grid)
matplot(grid, t(ice - ice[, 1]), type = "l", lty = 1, col = "#00000020",
        xlab = "Perda de peso (kg)", ylab = "Variação dos dias previstos (ICE centralizada)")
lines(grid, colMeans(ice - ice[, 1]), lwd = 4, col = "#1f6f8b")

# 4. H de Friedman para um par (em 60 pacientes de treino sorteados) ----------
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

# 5. Valores de Shapley exatos para pacientes da UTI Ala 7: o modelo OLS do Capítulo 4
#    (P10: os dez preditores dos Capítulos 4 e 6; o local conta como um único jogador)
icu   <- read.csv("data/icu_aki.csv")
split <- read.csv("data/splits/ward7_split.csv")
train <- icu[split$set == "train", ];  test <- icu[split$set == "test", ]
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
f10 <- c("site", "age", "sofa", "lactate", "map_min", "baseline_creat",
         "protocol_adherence", "bun", "crp", "albumin")
ols7 <- lm(reformulate(f10, "delta_creatinine_48h"), data = train)
round(mae(test$delta_creatinine_48h, predict(ols7, test)), 4)   # 0,1413, o valor do registro de modelos (Capítulo 4)
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
round(sh, 3)     # cada coluna: referência + contribuições = predição

# 6. A floresta aleatória do Capítulo 6 (500 árvores, scikit-learn) ---------------
#    O R não consegue reajustar exatamente a floresta do scikit-learn, e seus valores
#    de Shapley exatos exigem cerca de um milhão de predições por paciente: leia os
#    calculados antes por scripts/precompute/ch12_explain.py (referência: 200 pacientes de treino).
shf <- read.csv("data/precomputed/ch12/ward7_shapley.csv", encoding = "UTF-8")
shf <- shf[shf$model == "Random forest", ]
round(xtabs(phi ~ feature + name, data = shf), 3)
unique(shf[, c("name", "baseline", "prediction")])   # predições da floresta 0,29 / -0,09 / 0,83, como no registro de modelos
led <- read.csv("data/precomputed/ledger/ward7_reg.csv", encoding = "UTF-8")
led[led$model_id %in% c("ols", "rf500"), c("model", "test_MAE", "test_RMSE", "test_R2")]   # resultados de teste dos dois modelos, do registro de modelos

# 7. Opcional, no seu computador: as análises do vivid e do bartMan do capítulo -------
#    install.packages(c("vivid", "bartMan", "dbarts"))  (esses pacotes não rodam no navegador)
if (requireNamespace("vivid", quietly = TRUE) && requireNamespace("bartMan", quietly = TRUE) &&
    requireNamespace("dbarts", quietly = TRUE)) {
  library(vivid); library(dbarts); library(bartMan)
  trn <- ltr[, c(feats, "time")]                     # vivi() usa todas as outras colunas como atributos
  set.seed(2026)
  viv <- vivi(data = trn, fit = rf, response = "time", gridSize = 10,
              importanceType = "agnostic", numPerm = 4)
  print(round(viv, 2))                               # importância (diagonal), H não normalizado (dias)
  print(viviHeatmap(mat = viv, angle = 45))
  print(viviNetwork(mat = viv))
  print(pdpPairs(data = trn, fit = rf, response = "time",
                 vars = c("ph.ecog", "ph.karno", "wt.loss"), gridSize = 10, nIce = 50))
  set.seed(2026)
  bart_fit <- bart(x.train = as.matrix(ltr[feats]), y.train = ltr$time,
                   x.test = as.matrix(lte[feats]), ntree = 80L, nskip = 200L,
                   ndpost = 200L, keeptrees = TRUE, verbose = FALSE, seed = 2026)
  print(round(c(BART_test_MAE = mae(lte$time, colMeans(bart_fit$yhat.test))), 1))
  trees <- extractTreeData(model = bart_fit, data = data.frame(ltr[feats], y = ltr$time))  # um ou dois minutos
  vsupMat <- viviBartMatrix(trees = trees, type = "vsup", metric = "propMean", metricError = "CV")
  print(lapply(vsupMat, round, 3))                   # proporções de inclusão e o seu CV
  print(viviBartPlot(vsupMat, label = "CV"))
} else {
  message("O passo 7 precisa dos pacotes de R vivid, bartMan e dbarts: rode-o no seu computador.")
}
