# Laboratorio del capítulo 15
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. Cargar el corpus --------------------------------------------------------------
notes <- read.csv("data/clinical_notes.csv", na.strings = "", stringsAsFactors = FALSE)
table(notes$source, notes$set)

# 2. Expresiones regulares: dosis y fechas ------------------------------------------
w7 <- notes[notes$source == "ward7_discharge" & notes$set != "tonight", ]
doses <- unlist(regmatches(w7$text, gregexpr("[0-9]+ mg", w7$text)))
head(sort(table(doses), decreasing = TRUE))
c(iso_dates   = sum(grepl("\\b[0-9]{4}-[0-9]{2}-[0-9]{2}\\b", w7$text)),
  slash_dates = sum(grepl("\\b[0-9]{2}/[0-9]{2}/[0-9]{4}\\b", w7$text)))

# 3. NegEx-lite ------------------------------------------------------------------------
negex <- function(sentence, finding, window = 8,
                  pre = c("no evidence of", "negative for", "absence of", "rule out", "denies",
                          "denied", "without", "not", "no"),
                  post = c("was ruled out", "ruled out", "is absent", "excluded"),
                  pseudo = c("no increase", "no change", "not only", "no further"),
                  term = c("but", "however", "although", "except", ";")) {
  tok <- regmatches(tolower(sentence), gregexpr("[a-z0-9]+|;", tolower(sentence)))[[1]]
  at <- function(phrase, i) {                     # ¿empieza `phrase` en la posición i?
    p <- strsplit(phrase, " ")[[1]]
    i + length(p) - 1 <= length(tok) && all(tok[i:(i + length(p) - 1)] == p)
  }
  blocked <- function(a, b) b >= a && any(tok[a:b] %in% term)
  ts <- which(vapply(seq_along(tok), function(i) at(finding, i), logical(1)))[1]
  if (is.na(ts)) return(NA)
  te <- ts + length(strsplit(finding, " ")[[1]]) - 1
  i <- 1
  while (i < ts) {
    if (any(vapply(pseudo, at, logical(1), i = i))) { i <- i + 2; next }
    hit <- pre[vapply(pre, at, logical(1), i = i)][1]
    if (!is.na(hit)) {
      e <- i + length(strsplit(hit, " ")[[1]]) - 1
      if (e < ts && ts - e <= window && !blocked(e + 1, ts - 1)) return(TRUE)
      i <- e + 1; next
    }
    i <- i + 1
  }
  for (t in post) {
    s <- which(vapply(seq_along(tok), function(j) j > te && at(t, j), logical(1)))[1]
    if (!is.na(s) && s - te <= window && !blocked(te + 1, s - 1)) return(TRUE)
  }
  FALSE
}
neg <- read.csv("data/negation_sentences.csv", stringsAsFactors = FALSE)
neg$pred <- mapply(negex, neg$sentence, neg$finding)
mean(neg$pred == (neg$negated == 1))               # exactitud con una ventana de 8
neg[neg$pred != (neg$negated == 1), c("sentence", "finding", "negated")]

# 4. Tokens y una matriz TF-IDF (vocabulario de entrenamiento) ---------------------------
lab <- notes[notes$set %in% c("train", "test"), ]
train <- lab$set == "train"
tokenize <- function(x) regmatches(tolower(x), gregexpr("[a-z0-9]{2,}", tolower(x)))
toks <- tokenize(lab$text)
df_train <- table(unlist(lapply(toks[train], unique)))
vocab <- names(df_train)[df_train >= 2]             # palabras en al menos dos notas de entrenamiento
length(vocab)
dtm <- t(vapply(toks, function(t) tabulate(match(t, vocab), nbins = length(vocab)),
                numeric(length(vocab))))
idf <- log((1 + sum(train)) / (1 + colSums(dtm[train, ] > 0))) + 1   # idf suavizado de scikit-learn
X <- sweep(dtm, 2, idf, "*")
X <- X / sqrt(pmax(rowSums(X^2), 1e-12))            # filas a longitud 1

# 5. Regresión logística ridge, escrita paso a paso ---------------------------------------
y <- lab$adr; C <- 10                                # C = 1 / (2 * lambda), como en scikit-learn
loss <- function(b) { z <- b[1] + X[train, ] %*% b[-1]
  sum(log1p(exp(-abs(z))) + pmax(z, 0) - y[train] * z) + sum(b[-1]^2) / (2 * C) }
grad <- function(b) { z <- b[1] + X[train, ] %*% b[-1]; r <- plogis(z) - y[train]
  c(sum(r), crossprod(X[train, ], r) + b[-1] / C) }
fit <- optim(rep(0, ncol(X) + 1), loss, grad, method = "BFGS", control = list(maxit = 1000))
p <- as.vector(plogis(fit$par[1] + X[!train, ] %*% fit$par[-1]))
auc <- function(y, p) { r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
auc(y[!train], p)
beta <- setNames(fit$par[-1], vocab)
head(sort(beta, decreasing = TRUE), 8)              # palabras que suben la probabilidad

# 6. Una cola de revisión -----------------------------------------------------------------
yt <- y[!train]; t_low <- 0.10; t_high <- 0.65
c(cleared = sum(p < t_low), missed = sum(p < t_low & yt == 1),
  review = sum(p >= t_low & p < t_high),
  flagged = sum(p >= t_high), false_reports = sum(p >= t_high & yt == 0))
