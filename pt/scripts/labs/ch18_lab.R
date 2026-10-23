# Chapter 18 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. The book, the tokenizer and the split -----------------------------------------
lines <- readLines("data/notes_on_nursing.txt", encoding = "UTF-8")
text  <- paste0(paste(lines, collapse = "\n"), "\n")
ch    <- strsplit(text, "")[[1]]
chars <- unique(ch); chars <- chars[order(sapply(chars, utf8ToInt))]   # same order as Python
V <- length(chars)
encode <- function(s) match(strsplit(s, "")[[1]], chars) - 1
decode <- function(ids) paste(chars[ids + 1], collapse = "")
data <- match(ch, chars) - 1
n_train <- floor(0.9 * length(data))
train_ids <- data[1:n_train]; val_ids <- data[(n_train + 1):length(data)]
c(characters = length(data), vocabulary = V, train = length(train_ids), validation = length(val_ids))

# 2. A batch: inputs and targets shifted by one --------------------------------------
set.seed(1)
starts <- sample(1:(length(train_ids) - 33), 4)
X <- t(sapply(starts, function(s) train_ids[s:(s + 31)]))
Y <- t(sapply(starts, function(s) train_ids[(s + 1):(s + 32)]))
cat(decode(X[1, ]), "|", decode(Y[1, ]), "\n")

# 3. Baselines on the validation text (bits per character) ---------------------------
p_uni <- (tabulate(train_ids + 1, V) + 1) / (length(train_ids) + V)
pair  <- table(factor(head(train_ids, -1), 0:(V - 1)), factor(tail(train_ids, -1), 0:(V - 1)))
p_bi  <- (pair + 1) / (rowSums(pair) + V)
bl <- c(uniform = log2(V), unigram = -mean(log2(p_uni[val_ids[-1] + 1])),
        bigram = -mean(log2(p_bi[cbind(head(val_ids, -1) + 1, tail(val_ids, -1) + 1)])))
round(rbind(bits = bl, perplexity = 2^bl), 3)

# 4. Load the pretrained GPT: JSON description + float16 weights ---------------------
meta <- jsonlite::fromJSON("data/precomputed/ch18/gpt_nightingale.json")
n_w  <- sum(meta$tensors$size)
h16  <- readBin("data/precomputed/ch18/gpt_nightingale.bin", "integer", n = n_w, size = 2,
                signed = FALSE, endian = "little")
half_to_double <- function(h) {                       # IEEE 754 half precision -> double
  s <- ifelse(h >= 32768, -1, 1); e <- (h %/% 1024) %% 32; f <- h %% 1024
  ifelse(e == 0, s * f * 2^-24, s * (1 + f / 1024) * 2^(e - 15))
}
w_all <- half_to_double(h16)
W <- list()
for (i in seq_len(nrow(meta$tensors))) {
  tt <- meta$tensors[i, ]; v <- w_all[tt$offset + seq_len(tt$size)]
  sh <- tt$shape[[1]]
  W[[tt$name]] <- if (length(sh) == 2) matrix(v, sh[1], sh[2], byrow = TRUE) else v   # row-major
}
cfg <- meta$config; stopifnot(identical(meta$chars, paste(chars, collapse = "")))
c(weights = n_w, unlist(cfg))

# 5. The forward pass in base R ------------------------------------------------------
layer_norm <- function(X, g, b) {
  mu <- rowMeans(X); v <- rowMeans((X - mu)^2)
  sweep(sweep((X - mu) / sqrt(v + 1e-5), 2, g, "*"), 2, b, "+")
}
gelu <- function(x) 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
softmax_rows <- function(S) { E <- exp(S - apply(S, 1, max)); E / rowSums(E) }
add_bias <- function(X, b) sweep(X, 2, b, "+")
gpt_logits <- function(ids) {                          # ids: 0-based, at most block_size of them
  Tn <- length(ids); d <- cfg$d; h <- cfg$n_head; dk <- d / h
  X <- W$wte[ids + 1, , drop = FALSE] + W$wpe[1:Tn, , drop = FALSE]
  mask <- matrix(0, Tn, Tn); mask[upper.tri(mask)] <- -Inf
  for (l in 0:(cfg$n_layer - 1)) {
    P <- function(n) W[[sprintf("h%d.%s", l, n)]]
    A_in <- layer_norm(X, P("ln1.g"), P("ln1.b"))
    QKV <- add_bias(A_in %*% P("attn.w"), P("attn.b"))
    heads <- lapply(1:h, function(m) {
      cols <- ((m - 1) * dk + 1):(m * dk)
      Q <- QKV[, cols, drop = FALSE]; K <- QKV[, d + cols, drop = FALSE]; Vv <- QKV[, 2 * d + cols, drop = FALSE]
      softmax_rows(Q %*% t(K) / sqrt(dk) + mask) %*% Vv
    })
    X <- X + add_bias(do.call(cbind, heads) %*% P("proj.w"), P("proj.b"))       # residual 1
    M_in <- layer_norm(X, P("ln2.g"), P("ln2.b"))
    X <- X + add_bias(gelu(add_bias(M_in %*% P("fc.w"), P("fc.b"))) %*% P("fc2.w"), P("fc2.b"))  # residual 2
  }
  layer_norm(X, W$lnf.g, W$lnf.b) %*% t(W$wte)        # weight tying: T x V logits
}

# 6. Check against the logits computed in Python -------------------------------------
nxt <- jsonlite::fromJSON("data/precomputed/ch18/next_char.json")
z_r <- gpt_logits(encode("The nurse"))[9, ]
z_py <- nxt$contexts$logits[[which(nxt$contexts$prompt == "The nurse")]]
max(abs(z_r - z_py))                                   # tiny: same model, two languages
p <- exp(z_r - max(z_r)); p <- p / sum(p)
head(setNames(round(sort(p, decreasing = TRUE), 3), chars[order(p, decreasing = TRUE)]), 5)

# 7. Generate with temperature and top-k ----------------------------------------------
generate_r <- function(prompt, n = 80, tau = 0.8, k = 10, seed = 1) {
  set.seed(seed); ids <- encode(prompt)
  for (i in 1:n) {
    ctx <- tail(ids, cfg$block_size)
    z <- gpt_logits(ctx)[length(ctx), ] / tau
    z[z < sort(z, decreasing = TRUE)[k]] <- -Inf
    p <- exp(z - max(z)); ids <- c(ids, sample(0:(V - 1), 1, prob = p / sum(p)))
  }
  decode(ids)
}
cat(generate_r("Fresh air", n = 80), "\n")
