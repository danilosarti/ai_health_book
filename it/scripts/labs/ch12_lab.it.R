# Laboratorio del capitolo 12
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 0. Pacchetti e dati ------------------------------------------------------
library(randomForest)
lung  <- read.csv("data/lung_sim_complete.csv")
lung$male <- as.integer(lung$sex == "male")
lsp   <- read.csv("data/splits/lung_split.csv")
ltr   <- lung[lsp$set == "train", ];  lte <- lung[lsp$set == "test", ]
feats <- c("age", "male", "ph.ecog", "ph.karno", "wt.loss")
mae   <- function(y, yhat) mean(abs(y - yhat))

# 1. Due modelli dei giorni di follow-up (ricorda: la censura è ignorata) ---
lin <- lm(time ~ age + male + ph.ecog + ph.karno + wt.loss, data = ltr)
set.seed(2026)
rf  <- randomForest(time ~ age + male + ph.ecog + ph.karno + wt.loss, data = ltr, ntree = 300)
c(linear = mae(lte$time, predict(lin, lte)), forest = mae(lte$time, predict(rf, lte)))

# 2. Importanza per permutazione per qualsiasi modello -----------------------
perm_importance <- function(predict_fun, data, y, features, n_shuffles = 10) {
  base <- mae(y, predict_fun(data))
  sapply(features, function(f) {
    mean(replicate(n_shuffles, {
      d <- data; d[[f]] <- sample(d[[f]])
      mae(y, predict_fun(d))
    })) - base                                    # rimescolato meno originale
  })
}
set.seed(2026)
round(perm_importance(function(d) predict(lin, d), lte, lte$time, feats), 1)
round(perm_importance(function(d) predict(rf, d), lte, lte$time, feats), 1)

# 3. Curve ICE e dipendenza parziale -------------------------------------------
ice_curves <- function(predict_fun, data, feature, grid) {
  sapply(grid, function(g) { d <- data; d[[feature]] <- g; predict_fun(d) })  # pazienti x griglia
}
grid <- seq(1, 25, by = 1)
ice  <- ice_curves(function(d) predict(rf, d), ltr, "wt.loss", grid)
matplot(grid, t(ice - ice[, 1]), type = "l", lty = 1, col = "#00000020",
        xlab = "Calo ponderale (kg)", ylab = "Variazione dei giorni previsti (ICE centrata)")
lines(grid, colMeans(ice - ice[, 1]), lwd = 4, col = "#1f6f8b")

# 4. H di Friedman per una coppia (su 60 pazienti di addestramento estratti) -
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

# 5. Valori di Shapley esatti per pazienti del Reparto 7: il modello OLS del Capitolo 4
#    (P10: i dieci predittori dei Capitoli 4 e 6; il sito conta come un solo giocatore)
icu   <- read.csv("data/icu_aki.csv")
split <- read.csv("data/splits/ward7_split.csv")
train <- icu[split$set == "train", ];  test <- icu[split$set == "test", ]
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
f10 <- c("site", "age", "sofa", "lactate", "map_min", "baseline_creat",
         "protocol_adherence", "bun", "crp", "albumin")
ols7 <- lm(reformulate(f10, "delta_creatinine_48h"), data = train)
round(mae(test$delta_creatinine_48h, predict(ols7, test)), 4)   # 0,1413, il valore del registro dei modelli (Capitolo 4)
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
round(sh, 3)     # ogni colonna: riferimento + contributi = previsione

# 6. La foresta casuale del Capitolo 6 (500 alberi, scikit-learn) ------------------
#    R non può riadattare esattamente la foresta di scikit-learn, e i suoi valori di
#    Shapley esatti richiedono circa un milione di previsioni per paziente: si leggono
#    quelli calcolati in anticipo da scripts/precompute/ch12_explain.py (riferimento: 200 pazienti di addestramento).
shf <- read.csv("data/precomputed/ch12/ward7_shapley.csv", encoding = "UTF-8")
shf <- shf[shf$model == "Random forest", ]
round(xtabs(phi ~ feature + name, data = shf), 3)
unique(shf[, c("name", "baseline", "prediction")])   # previsioni della foresta 0,29 / -0,09 / 0,83, come nel registro dei modelli
led <- read.csv("data/precomputed/ledger/ward7_reg.csv", encoding = "UTF-8")
led[led$model_id %in% c("ols", "rf500"), c("model", "test_MAE", "test_RMSE", "test_R2")]   # risultati di test dei due modelli, dal registro dei modelli

# 7. Facoltativo, sul tuo computer: le analisi con vivid e bartMan del capitolo -------
#    install.packages(c("vivid", "bartMan", "dbarts"))  (questi pacchetti non funzionano nel browser)
if (requireNamespace("vivid", quietly = TRUE) && requireNamespace("bartMan", quietly = TRUE) &&
    requireNamespace("dbarts", quietly = TRUE)) {
  library(vivid); library(dbarts); library(bartMan)
  trn <- ltr[, c(feats, "time")]                     # vivi() usa ogni altra colonna come variabile
  set.seed(2026)
  viv <- vivi(data = trn, fit = rf, response = "time", gridSize = 10,
              importanceType = "agnostic", numPerm = 4)
  print(round(viv, 2))                               # importanza (diagonale), H non normalizzato (giorni)
  print(viviHeatmap(mat = viv, angle = 45))
  print(viviNetwork(mat = viv))
  print(pdpPairs(data = trn, fit = rf, response = "time",
                 vars = c("ph.ecog", "ph.karno", "wt.loss"), gridSize = 10, nIce = 50))
  set.seed(2026)
  bart_fit <- bart(x.train = as.matrix(ltr[feats]), y.train = ltr$time,
                   x.test = as.matrix(lte[feats]), ntree = 80L, nskip = 200L,
                   ndpost = 200L, keeptrees = TRUE, verbose = FALSE, seed = 2026)
  print(round(c(BART_test_MAE = mae(lte$time, colMeans(bart_fit$yhat.test))), 1))
  trees <- extractTreeData(model = bart_fit, data = data.frame(ltr[feats], y = ltr$time))  # un paio di minuti
  vsupMat <- viviBartMatrix(trees = trees, type = "vsup", metric = "propMean", metricError = "CV")
  print(lapply(vsupMat, round, 3))                   # proporzioni di inclusione e il loro CV
  print(viviBartPlot(vsupMat, label = "CV"))
} else {
  message("Il passo 7 richiede i pacchetti R vivid, bartMan e dbarts: eseguilo sul tuo computer.")
}
