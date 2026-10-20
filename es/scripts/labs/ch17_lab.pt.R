# Laboratório do capítulo 17
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Tokens, embeddings e posições -------------------------------------------
set.seed(2026)
vocab  <- c("<bos>", "the", "patient", "reports", "denies", "chest", "pain", "fever")
tokens <- c("<bos>", "the", "patient", "reports", "chest", "pain")
ids    <- match(tokens, vocab)
d <- 8; h <- 2; d_k <- d / h; d_ff <- 4 * d; n_tok <- length(ids)
E <- matrix(rnorm(length(vocab) * d, sd = 0.5), ncol = d)        # matriz de embeddings |V| x d
pos_enc <- function(T, d) {                                       # codificações senoidais
  P <- matrix(0, T, d)
  for (t in 1:T) for (r in 0:(d / 2 - 1)) {
    P[t, 2 * r + 1] <- sin((t - 1) / 10000^(2 * r / d))
    P[t, 2 * r + 2] <- cos((t - 1) / 10000^(2 * r / d))
  }
  P
}
X <- E[ids, ] + pos_enc(n_tok, d)                                 # entrada n_tok x d
dim(X)

# 2. Peças de construção --------------------------------------------------------
softmax <- function(x) { e <- exp(x - max(x)); e / sum(e) }
layer_norm <- function(X) t(apply(X, 1, function(x) (x - mean(x)) / sqrt(mean((x - mean(x))^2) + 1e-5)))
gelu <- function(x) 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
new_block <- function() list(
  W_Q = matrix(rnorm(d * d, sd = 0.3), d), W_K = matrix(rnorm(d * d, sd = 0.3), d),
  W_V = matrix(rnorm(d * d, sd = 0.3), d), W_O = matrix(rnorm(d * d, sd = 0.3), d),
  W_1 = matrix(rnorm(d * d_ff, sd = 0.3), d), W_2 = matrix(rnorm(d_ff * d, sd = 0.3), d_ff))
causal <- matrix(0, n_tok, n_tok); causal[upper.tri(causal)] <- -Inf   # a máscara causal

mha <- function(X, B) {                                           # atenção com múltiplas cabeças
  Q <- X %*% B$W_Q; K <- X %*% B$W_K; V <- X %*% B$W_V
  heads <- lapply(1:h, function(m) {
    cols <- ((m - 1) * d_k + 1):(m * d_k)                          # a fatia desta cabeça
    S <- Q[, cols] %*% t(K[, cols]) / sqrt(d_k) + causal
    A <- t(apply(S, 1, softmax))
    list(A = A, out = A %*% V[, cols])
  })
  list(out = do.call(cbind, lapply(heads, `[[`, "out")) %*% B$W_O,
       A = lapply(heads, `[[`, "A"))
}
block <- function(X, B) {                                         # bloco pre-LN (sem vieses e ganhos)
  att <- mha(layer_norm(X), B)
  X1  <- X + att$out                                              # X' = X + MHA(LN(X))
  Y   <- X1 + gelu(layer_norm(X1) %*% B$W_1) %*% B$W_2              # Y  = X' + FFN(LN(X'))
  list(Y = Y, A = att$A)
}

# 3. Dois blocos empilhados -----------------------------------------------------
B1 <- new_block(); B2 <- new_block()
out1 <- block(X, B1); out2 <- block(out1$Y, B2)
dim(out2$Y)                                                       # continua n_tok x d
round(out1$A[[1]], 2)                                             # cabeça 1, camada 1: triângulo superior é 0
rowSums(out1$A[[1]])

# 4. Probabilidades do próximo token (pesos compartilhados) e entropia cruzada ----
H <- layer_norm(out2$Y)
logits <- H %*% t(E)                                              # n_tok x |V|
probs  <- t(apply(logits, 1, softmax))
next_ids <- ids[-1]                                               # o token que realmente vem a seguir
nll <- -log(probs[cbind(1:(n_tok - 1), next_ids)])
c(mean_cross_entropy = mean(nll), uniform_guess = log(length(vocab)))

# 5. Pesos em um bloco (com vieses e ganhos da normalização de camada) -------------
c(attention = 4 * (d * d + d), ffn = d * d_ff + d_ff + d_ff * d + d, layernorm = 4 * d)

# 6. Atenção do pequeno transformer treinado --------------------------------------
att  <- read.csv("data/precomputed/ch17/attention.csv")
sent <- read.csv("data/precomputed/ch17/sentences.csv")
tok1 <- sent$token[sent$sentence == 1]
a <- att[att$sentence == 1 & att$layer == 1 & att$head == 1, ]
A_trained <- matrix(0, length(tok1), length(tok1), dimnames = list(tok1, tok1))
A_trained[cbind(a$query + 1, a$key + 1)] <- a$weight
last_row <- A_trained[length(tok1), ]
head(sort(last_row, decreasing = TRUE), 3)                         # para onde a última posição olha
image(t(A_trained[nrow(A_trained):1, ]), axes = FALSE, col = hcl.colors(30, "Blues", rev = TRUE),
      main = "Modelo treinado, camada 1, cabeça 1")
