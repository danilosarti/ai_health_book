# Laboratório do capítulo 14
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Um bandido de três estímulos para o HomeBeat -------------------------------
p_true <- c(breathing = 0.20, walk = 0.35, sleep = 0.30)    # desconhecidas pelos agentes
run_bandit <- function(strategy, T = 500, eps = 0.1) {
  k <- length(p_true); n <- rep(0, k); s <- rep(0, k); regret <- 0
  for (t in 1:T) {
    if (t <= k) arm <- t                                     # experimenta cada estímulo uma vez
    else if (strategy == "thompson") arm <- which.max(rbeta(k, 1 + s, 1 + n - s))
    else if (strategy == "epsilon" && runif(1) < eps) arm <- sample(k, 1)
    else arm <- which.max(s / n)                             # escolha gulosa
    y <- rbinom(1, 1, p_true[arm])
    n[arm] <- n[arm] + 1; s[arm] <- s[arm] + y
    regret <- regret + max(p_true) - p_true[[arm]]
  }
  c(regret = regret, share_walk = n[2] / T)
}
set.seed(14)
round(sapply(c(greedy = "greedy", epsilon = "epsilon", thompson = "thompson"),
             function(st) rowMeans(replicate(50, run_bandit(st)))), 3)

# 2. O problema de brinquedo da insulina como processo de decisão de Markov ------
bands <- c("<4", "4-6", "6-10", "10-14", ">14")
rates <- c("off", "low", "medium", "high")
acts  <- c("decrease", "keep", "increase")
moves <- rbind(c(0.02, 0.38, 0.60),       # off:    desce, permanece, sobe
               c(0.10, 0.60, 0.30),       # low
               c(0.30, 0.60, 0.10),       # medium
               c(0.60, 0.38, 0.02))       # high
reward <- c(-10, 0, 1, 0, -1)             # por chegar a cada faixa
gamma <- 0.9
S <- 20; A <- 3
band_of <- rep(1:5, 4); rate_of <- rep(0:3, each = 5)   # estado s = velocidade * 5 + faixa
P <- array(0, c(S, A, S))
for (s in 1:S) for (a in 1:A) {
  r2 <- min(3, max(0, rate_of[s] + a - 2))
  for (m in 1:3) {
    g2 <- min(5, max(1, band_of[s] + m - 2))
    P[s, a, r2 * 5 + g2] <- P[s, a, r2 * 5 + g2] + moves[r2 + 1, m]
  }
}
R <- sapply(1:A, function(a) P[, a, ] %*% reward[band_of])   # recompensa esperada de cada (s, a)

# 3. Iteração de valor: aplicar a equação de otimalidade de Bellman até estabilizar
value_iteration <- function(R, sweeps = 500) {
  V <- rep(0, S)
  for (k in 1:sweeps) {
    Q <- R + gamma * sapply(1:A, function(a) P[, a, ] %*% V)
    V <- apply(Q, 1, max)
  }
  list(Q = Q, V = V, policy = apply(Q, 1, which.max))
}
opt <- value_iteration(R)
matrix(acts[opt$policy], 5, 4, dimnames = list(bands, rates))   # política ótima (em 'off', decrease = keep)
round(matrix(opt$V, 5, 4, dimnames = list(bands, rates)), 2)    # os seus valores

# 4. Q-learning a partir de experiência simulada ---------------------------------
set.seed(2014)
Qhat <- matrix(0, S, A); eta <- 0.1; eps <- 0.2
for (episode in 1:3000) {
  s <- sample(S, 1)                                           # estado inicial aleatório
  for (t in 1:12) {                                           # 24 horas, passos de 2 horas
    a  <- if (runif(1) < eps) sample(A, 1) else which.max(Qhat[s, ])
    s2 <- sample(S, 1, prob = P[s, a, ])
    r  <- reward[band_of[s2]]
    Qhat[s, a] <- Qhat[s, a] + eta * (r + gamma * max(Qhat[s2, ]) - Qhat[s, a])
    s <- s2
  }
}
learned <- apply(Qhat, 1, which.max)
mean(opt$Q[cbind(1:S, learned)] >= opt$V - 0.05)              # proporção de estados quase ótimos

# 5. Comparar políticas ao longo de 24 horas, de forma exata -----------------------
evaluate <- function(policy, start = 4) {                     # glicose 10-14, infusão desligada
  d <- replace(rep(0, S), start, 1); out <- c(in_range = 0, hypo = 0, hyper = 0)
  Ppi <- t(sapply(1:S, function(s) P[s, policy[s], ]))
  for (t in 1:12) {
    d <- as.vector(d %*% Ppi)
    out <- out + c(sum(d[band_of == 3]), sum(d[band_of == 1]), sum(d[band_of >= 4]))
  }
  round(out / 12, 3)
}
sliding <- ifelse(band_of >= 4, 3, ifelse(band_of <= 2, 1, 2))  # escala móvel de beira de leito
rbind(sliding_scale = evaluate(sliding), optimal = evaluate(opt$policy),
      q_learning = evaluate(learned))

# 6. Uma recompensa mal especificada: '+1 sempre que a glicose estiver abaixo de 10'
R_bad <- sapply(1:A, function(a) P[, a, ] %*% c(1, 1, 1, 0, 0)[band_of])
evaluate(value_iteration(R_bad)$policy)

# 7. Q-learning offline a partir de prontuários de cuidado com escala móvel ---------
set.seed(7)
D <- matrix(0, 500 * 12, 4); i <- 0                           # colunas: s, a, r, s'
for (episode in 1:500) {
  s <- sample(3:5, 1)                                         # internado com 6-14+, infusão desligada
  for (t in 1:12) {
    a <- sliding[s]; s2 <- sample(S, 1, prob = P[s, a, ]); i <- i + 1
    D[i, ] <- c(s, a, reward[band_of[s2]], s2); s <- s2
  }
}
Qoff <- matrix(0, S, A)
for (sweep in 1:20) for (j in sample(nrow(D))) {
  d <- D[j, ]
  Qoff[d[1], d[2]] <- Qoff[d[1], d[2]] + 0.1 * (d[3] + gamma * max(Qoff[d[4], ]) - Qoff[d[1], d[2]])
}
seen <- unique(D[, 1] + S * (D[, 2] - 1))
c(share_never_tried = 1 - length(seen) / (S * A))
rbind(nurses_in_records = evaluate(sliding), offline_agent = evaluate(apply(Qoff, 1, which.max)))
