# Laboratorio del capítulo 17
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. Tokens, embeddings y posiciones -----------------------------------------
set.seed(2026)
vocab  <- c("<bos>", "the", "patient", "reports", "denies", "chest", "pain", "fever")
tokens <- c("<bos>", "the", "patient", "reports", "chest", "pain")
ids    <- match(tokens, vocab)
d <- 8; h <- 2; d_k <- d / h; d_ff <- 4 * d; n_tok <- length(ids)
E <- matrix(rnorm(length(vocab) * d, sd = 0.5), ncol = d)        # matriz de embeddings |V| x d
pos_enc <- function(T, d) {                                       # codificaciones sinusoidales
  P <- matrix(0, T, d)
  for (t in 1:T) for (r in 0:(d / 2 - 1)) {
    P[t, 2 * r + 1] <- sin((t - 1) / 10000^(2 * r / d))
    P[t, 2 * r + 2] <- cos((t - 1) / 10000^(2 * r / d))
  }
  P
}
X <- E[ids, ] + pos_enc(n_tok, d)                                 # entrada n_tok x d
dim(X)

# 2. Piezas de construcción ----------------------------------------------------
softmax <- function(x) { e <- exp(x - max(x)); e / sum(e) }
layer_norm <- function(X) t(apply(X, 1, function(x) (x - mean(x)) / sqrt(mean((x - mean(x))^2) + 1e-5)))
gelu <- function(x) 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
new_block <- function() list(
  W_Q = matrix(rnorm(d * d, sd = 0.3), d), W_K = matrix(rnorm(d * d, sd = 0.3), d),
  W_V = matrix(rnorm(d * d, sd = 0.3), d), W_O = matrix(rnorm(d * d, sd = 0.3), d),
  W_1 = matrix(rnorm(d * d_ff, sd = 0.3), d), W_2 = matrix(rnorm(d_ff * d, sd = 0.3), d_ff))
causal <- matrix(0, n_tok, n_tok); causal[upper.tri(causal)] <- -Inf   # la máscara causal

mha <- function(X, B) {                                           # atención multicabeza
  Q <- X %*% B$W_Q; K <- X %*% B$W_K; V <- X %*% B$W_V
  heads <- lapply(1:h, function(m) {
    cols <- ((m - 1) * d_k + 1):(m * d_k)                          # la porción de esta cabeza
    S <- Q[, cols] %*% t(K[, cols]) / sqrt(d_k) + causal
    A <- t(apply(S, 1, softmax))
    list(A = A, out = A %*% V[, cols])
  })
  list(out = do.call(cbind, lapply(heads, `[[`, "out")) %*% B$W_O,
       A = lapply(heads, `[[`, "A"))
}
block <- function(X, B) {                                         # bloque pre-LN (sin sesgos ni ganancias)
  att <- mha(layer_norm(X), B)
  X1  <- X + att$out                                              # X' = X + MHA(LN(X))
  Y   <- X1 + gelu(layer_norm(X1) %*% B$W_1) %*% B$W_2              # Y  = X' + FFN(LN(X'))
  list(Y = Y, A = att$A)
}

# 3. Dos bloques apilados ------------------------------------------------------
B1 <- new_block(); B2 <- new_block()
out1 <- block(X, B1); out2 <- block(out1$Y, B2)
dim(out2$Y)                                                       # sigue siendo n_tok x d
round(out1$A[[1]], 2)                                             # cabeza 1, capa 1: el triángulo superior es 0
rowSums(out1$A[[1]])

# 4. Probabilidades del siguiente token (pesos compartidos) y entropía cruzada --
H <- layer_norm(out2$Y)
logits <- H %*% t(E)                                              # n_tok x |V|
probs  <- t(apply(logits, 1, softmax))
next_ids <- ids[-1]                                               # el token que viene realmente después
nll <- -log(probs[cbind(1:(n_tok - 1), next_ids)])
c(mean_cross_entropy = mean(nll), uniform_guess = log(length(vocab)))

# 5. Pesos de un bloque (con sesgos y ganancias de la normalización de capa) ----
c(attention = 4 * (d * d + d), ffn = d * d_ff + d_ff + d_ff * d + d, layernorm = 4 * d)

# 6. Atención del pequeño transformer entrenado --------------------------------
att  <- read.csv("data/precomputed/ch17/attention.csv")
sent <- read.csv("data/precomputed/ch17/sentences.csv")
tok1 <- sent$token[sent$sentence == 1]
a <- att[att$sentence == 1 & att$layer == 1 & att$head == 1, ]
A_trained <- matrix(0, length(tok1), length(tok1), dimnames = list(tok1, tok1))
A_trained[cbind(a$query + 1, a$key + 1)] <- a$weight
last_row <- A_trained[length(tok1), ]
head(sort(last_row, decreasing = TRUE), 3)                         # adónde mira la última posición
image(t(A_trained[nrow(A_trained):1, ]), axes = FALSE, col = hcl.colors(30, "Blues", rev = TRUE),
      main = "Modelo entrenado, capa 1, cabeza 1")
