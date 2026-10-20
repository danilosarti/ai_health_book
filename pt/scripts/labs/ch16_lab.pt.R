# Laboratório do capítulo 16
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Carregar e tokenizar as anotações -----------------------------------------
notes <- read.csv("data/ward7_notes.csv")
toks  <- function(s) c("<s>", strsplit(s, " ")[[1]], "</s>")
train <- lapply(notes$text[notes$set == "train"], toks)
test  <- lapply(notes$text[notes$set == "test"], toks)
vocab <- sort(unique(unlist(c(train, test))))
V     <- length(vocab) - 1                      # qualquer token exceto <s> pode vir em seguida

# 2. Um modelo de bigramas por contagem (máxima verossimilhança) ---------------
prev <- unlist(lapply(train, head, -1)); nxt <- unlist(lapply(train, tail, -1))
counts <- table(factor(prev, levels = vocab), factor(nxt, levels = vocab))
round(sort(counts["reports", ] / sum(counts["reports", ]), decreasing = TRUE)[1:5], 3)

# 3. Suavização add-delta e perplexidade nas anotações de teste ----------------
p_bigram <- function(a, b, delta) (counts[a, b] + delta) / (sum(counts[a, ]) + delta * V)
perplexity <- function(delta) {
  lp <- unlist(lapply(test, function(w) log2(mapply(p_bigram, head(w, -1), tail(w, -1), delta))))
  2^(-mean(lp))
}
sapply(c(1, 0.1, 0.01), perplexity)

# 4. Gerar anotações por amostragem, com uma temperatura -----------------------
generate <- function(tau = 1, max_len = 30) {
  w <- "<s>"; out <- character(0)
  for (i in seq_len(max_len)) {
    p <- counts[w, ] / sum(counts[w, ])
    p <- p^(1 / tau); p <- p / sum(p)          # temperatura nas probabilidades = nas log-contagens
    w <- sample(vocab, 1, prob = p)
    if (w == "</s>") break
    out <- c(out, w)
  }
  paste(out, collapse = " ")
}
set.seed(16)
for (tau in c(0.3, 1, 2)) cat("tau =", tau, ":", generate(tau), "\n")

# 5. Um pequeno modelo de linguagem neural (janela de 3 tokens), treinado do zero
idx <- setNames(seq_along(vocab), vocab)
windows <- function(docs) {
  m <- do.call(rbind, lapply(docs, function(w) {
    w <- c("<s>", "<s>", w)                    # preenche para que todo token tenha 3 antecessores
    i <- 4:length(w)
    cbind(idx[w[i - 3]], idx[w[i - 2]], idx[w[i - 1]], idx[w[i]])
  }))
  unname(m)
}
Wtr <- windows(train); Wte <- windows(test)
set.seed(2003)
nV <- length(vocab); d <- 16; H <- 64; lr <- 0.1
E  <- matrix(rnorm(nV * d, 0, 0.1), nV, d)
W1 <- matrix(rnorm(3 * d * H, 0, 1 / sqrt(3 * d)), 3 * d, H); b1 <- rep(0, H)
W2 <- matrix(rnorm(H * nV, 0, 1 / sqrt(H)), H, nV);           b2 <- rep(0, nV)
forward <- function(X) {
  x <- cbind(E[X[, 1], , drop = FALSE], E[X[, 2], , drop = FALSE], E[X[, 3], , drop = FALSE])
  h <- tanh(x %*% W1 + rep(b1, each = nrow(X)))
  z <- h %*% W2 + rep(b2, each = nrow(X))
  z <- z - apply(z, 1, max); p <- exp(z); p <- p / rowSums(p)
  list(x = x, h = h, p = p)
}
bits <- function(W) { f <- forward(W[, 1:3, drop = FALSE]); -mean(log2(f$p[cbind(seq_len(nrow(W)), W[, 4])])) }
cat("antes do treino:", round(bits(Wte), 3), "bits por token\n")
for (epoch in 1:5) {
  for (b in split(sample(nrow(Wtr)), ceiling(seq_len(nrow(Wtr)) / 64))) {
    X <- Wtr[b, 1:3, drop = FALSE]; y <- Wtr[b, 4]; n <- length(b)
    f <- forward(X)
    g <- f$p; g[cbind(seq_len(n), y)] <- g[cbind(seq_len(n), y)] - 1; g <- g / n   # dPerda/dz
    gh <- (g %*% t(W2)) * (1 - f$h^2)
    gx <- gh %*% t(W1)
    W2 <- W2 - lr * crossprod(f$h, g); b2 <- b2 - lr * colSums(g)
    W1 <- W1 - lr * crossprod(f$x, gh); b1 <- b1 - lr * colSums(gh)
    for (k in 1:3) {                            # cada uma das 3 posições atualiza suas linhas de E
      gE <- rowsum(gx[, (k - 1) * d + 1:d], X[, k])
      E[as.integer(rownames(gE)), ] <- E[as.integer(rownames(gE)), ] - lr * gE
    }
  }
  cat("época", epoch, ": entropia cruzada de teste", round(bits(Wte), 3), "bits, perplexidade", round(2^bits(Wte), 2), "\n")
}
ctx <- rbind(idx[c("started", "vancomycin", "for")])
round(sort(setNames(forward(ctx)$p[1, ], vocab), decreasing = TRUE)[1:5], 3)

# 6. Os embeddings skip-gram pré-calculados: vizinhos e uma analogia -----------
wv <- read.csv("data/precomputed/ch16/word_vectors.csv")
U  <- as.matrix(wv[, grep("^d[0-9]+$", names(wv))]); rownames(U) <- wv$word
U  <- U / sqrt(rowSums(U^2))                    # comprimento unitário: cosseno = produto escalar
nearest <- function(v, k = 5, exclude = character(0)) {
  s <- drop(U %*% (v / sqrt(sum(v^2)))); s <- s[!names(s) %in% exclude]
  round(head(sort(s, decreasing = TRUE), k), 3)
}
nearest(U["vancomycin", ], exclude = "vancomycin")
nearest(U["nephrology", ] - U["kidney", ] + U["heart", ], exclude = c("nephrology", "kidney", "heart"))
