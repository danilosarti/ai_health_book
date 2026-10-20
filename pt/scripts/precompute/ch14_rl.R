# =====================================================================
# Chapter 14 — precomputed results (sequential decisions: reinforcement learning)
# Usage (from the book folder): Rscript scripts/precompute/ch14_rl.R
# Needs base R and jsonlite. Runs in a few seconds on one core. Writes small
# CSV/JSON files to data/precomputed/ch14/.
#
# Everything here is a TOY: a stylised insulin-infusion problem with five
# glucose bands and four infusion levels, and a three-armed bandit for
# HomeBeat nudges. No number in this file describes real patients.
# =====================================================================
out <- "data/precomputed/ch14"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

# ---- 1. The toy insulin MDP -------------------------------------------------
bands <- c("<4", "4-6", "6-10", "10-14", ">14")          # glucose, mmol/L
rates <- c("off", "low", "medium", "high")                # insulin infusion level
acts  <- c("decrease", "keep", "increase")                # change of one level
# Probability that glucose moves down / stays / moves up one band during the
# next two hours, given the infusion level AFTER the action (rows).
moves <- rbind(off    = c(0.02, 0.38, 0.60),
               low    = c(0.10, 0.60, 0.30),
               medium = c(0.30, 0.60, 0.10),
               high   = c(0.60, 0.38, 0.02))
presets <- list(clinical   = c(-10, 0, 1, 0, -1),   # weighted: hypoglycaemia is very bad
                tir_only   = c(0, 0, 1, 0, 0),      # +1 for time in range only
                below_10   = c(1, 1, 1, 0, 0),      # +1 whenever glucose < 10 (misspecified)
                custom     = c(-10, 0, 1, 0, -1))
gamma <- 0.9
S <- 20; A <- 3
sid <- function(g, r) r * 5 + g                      # g = 1..5 (band), r = 0..3 (rate)
s_band <- rep(1:5, 4); s_rate <- rep(0:3, each = 5)

build <- function(rew) {
  P <- array(0, c(S, A, S)); R <- matrix(0, S, A)
  for (r in 0:3) for (g in 1:5) for (a in 1:3) {
    r2 <- min(3, max(0, r + a - 2)); s <- sid(g, r)
    for (m in 1:3) {
      g2 <- min(5, max(1, g + m - 2)); s2 <- sid(g2, r2)
      P[s, a, s2] <- P[s, a, s2] + moves[r2 + 1, m]
      R[s, a] <- R[s, a] + moves[r2 + 1, m] * rew[g2]
    }
  }
  list(P = P, R = R)
}
value_iteration <- function(m, gamma = 0.9, tol = 1e-10) {
  V <- rep(0, S)
  repeat {
    Q <- m$R + gamma * sapply(1:A, function(a) m$P[, a, ] %*% V)
    V2 <- apply(Q, 1, max)
    if (max(abs(V2 - V)) < tol) break
    V <- V2
  }
  list(V = V2, Q = Q, policy = apply(Q, 1, which.max))
}
# Exact 24-hour (12 two-hour steps) occupancy from glucose 10-14, infusion off.
evaluate <- function(m, policy, start = sid(4, 0), steps = 12) {
  d <- rep(0, S); d[start] <- 1; acc <- c(0, 0, 0)
  Ppi <- t(sapply(1:S, function(s) m$P[s, policy[s], ]))
  for (t in 1:steps) {
    d <- as.vector(d %*% Ppi)
    acc <- acc + c(sum(d[s_band == 3]), sum(d[s_band == 1]), sum(d[s_band >= 4]))
  }
  setNames(acc / steps, c("in_range", "hypo", "hyper"))
}

m0 <- build(presets$clinical)
vi <- value_iteration(m0, gamma)
vi_tab <- data.frame(state = paste(bands[s_band], rates[s_rate + 1], sep = "|"),
                     band = bands[s_band], rate = rates[s_rate + 1],
                     V = round(vi$V, 4), best = acts[vi$policy],
                     Q_decrease = round(vi$Q[, 1], 4), Q_keep = round(vi$Q[, 2], 4),
                     Q_increase = round(vi$Q[, 3], 4))
write.csv(vi_tab, file.path(out, "vi_policy.csv"), row.names = FALSE)

# Policies under each reward preset, plus two simple fixed rules for comparison
fixed_low <- rep(2, S); fixed_low[s_rate == 0] <- 3; fixed_low[s_rate >= 2] <- 1  # go to 'low' and stay
pres <- do.call(rbind, lapply(c("clinical", "tir_only", "below_10"), function(k) {
  mk <- build(presets[[k]]); v <- value_iteration(mk, gamma)
  e <- evaluate(m0, v$policy)
  data.frame(preset = k, in_range = e[["in_range"]], hypo = e[["hypo"]], hyper = e[["hyper"]],
             policy = paste(substr(acts[v$policy], 1, 1), collapse = ""))
}))
# A bedside 'sliding scale': increase above 10 mmol/L, decrease below 6, otherwise keep
sliding <- ifelse(s_band >= 4, 3, ifelse(s_band <= 2, 1, 2))
for (nm in c("fixed_low", "sliding")) {
  pol <- get(nm); e_fix <- evaluate(m0, pol)
  pres <- rbind(pres, data.frame(preset = nm, in_range = e_fix[["in_range"]],
                                 hypo = e_fix[["hypo"]], hyper = e_fix[["hyper"]],
                                 policy = paste(substr(acts[pol], 1, 1), collapse = "")))
}
pres[, 2:4] <- round(pres[, 2:4], 4)
write.csv(pres, file.path(out, "reward_presets.csv"), row.names = FALSE)

# Policy evaluation of the fixed 'low' rate on glucose alone (Mission 2 check value)
P5 <- matrix(0, 5, 5)
for (g in 1:5) for (m in 1:3) { g2 <- min(5, max(1, g + m - 2)); P5[g, g2] <- P5[g, g2] + moves[2, m] }
r5 <- as.vector(P5 %*% presets$clinical)
V5 <- solve(diag(5) - gamma * P5, r5)

# ---- 2. Q-learning on the same problem (one seeded run) -----------------------
set.seed(1414)
q_learn <- function(m, episodes = 3000, alpha = 0.1, eps = 0.2, gamma = 0.9, steps = 12,
                    every = 25, Qstar = vi$Q, Vstar = vi$V) {
  Q <- matrix(0, S, A); curve <- NULL
  near_opt <- function(Q) mean(Qstar[cbind(1:S, apply(Q, 1, which.max))] >= Vstar - 0.05)
  for (ep in 1:episodes) {
    s <- sample(S, 1)
    for (t in 1:steps) {
      a  <- if (runif(1) < eps) sample(A, 1) else which.max(Q[s, ])
      s2 <- sample(S, 1, prob = m$P[s, a, ])
      r  <- presets$clinical[s_band[s2]]
      Q[s, a] <- Q[s, a] + alpha * (r + gamma * max(Q[s2, ]) - Q[s, a])
      s <- s2
    }
    if (ep %% every == 0 || ep == 1) curve <- rbind(curve, data.frame(episode = ep, near_optimal = near_opt(Q)))
  }
  list(Q = Q, curve = curve)
}
ql <- q_learn(m0)
write.csv(ql$curve, file.path(out, "qlearning_curve.csv"), row.names = FALSE)
ql_eval <- evaluate(m0, apply(ql$Q, 1, which.max))

# ---- 2b. Offline Q-learning from logged sliding-scale care ----------------------
# The 'records' come from clinicians who always follow the sliding scale. The
# learner can only replay these transitions; it can never try another action.
set.seed(2014)
offline_q <- function(logpol, episodes = 1000, sweeps = 20, alpha = 0.1) {
  D <- matrix(0, episodes * 12, 4); i <- 0
  for (ep in 1:episodes) {
    s <- sid(sample(3:5, 1), 0)
    for (t in 1:12) {
      a <- logpol[s]; s2 <- sample(S, 1, prob = m0$P[s, a, ]); i <- i + 1
      D[i, ] <- c(s, a, presets$clinical[s_band[s2]], s2); s <- s2
    }
  }
  Q <- matrix(0, S, A)
  for (k in 1:sweeps) for (j in sample(nrow(D))) {
    d <- D[j, ]
    Q[d[1], d[2]] <- Q[d[1], d[2]] + alpha * (d[3] + gamma * max(Q[d[4], ]) - Q[d[1], d[2]])
  }
  seen <- unique(D[, 1] + S * (D[, 2] - 1))
  list(Q = Q, unseen = 1 - length(seen) / (S * A), policy = apply(Q, 1, which.max))
}
off <- offline_q(sliding)
off_eval <- evaluate(m0, off$policy)

# ---- 3. The HomeBeat nudge bandit ---------------------------------------------
nudges <- c("breathing exercise", "short walk", "sleep tip")
p_true <- c(0.20, 0.35, 0.30)
T_h <- 500; reps <- 200
run_bandit <- function(strategy, seed, eps = 0.1) {
  set.seed(seed)
  k <- length(p_true); n <- rep(0, k); s <- rep(0, k); arms <- integer(T_h)
  for (t in 1:T_h) {
    if (strategy == "equal") arm <- sample(k, 1)
    else if (t <= k) arm <- t                                   # try each nudge once
    else if (strategy == "thompson") arm <- which.max(rbeta(k, 1 + s, 1 + n - s))
    else {
      est <- s / n
      best <- which(est == max(est)); best <- if (length(best) > 1) sample(best, 1) else best
      arm <- if (strategy == "epsilon" && runif(1) < eps) sample(k, 1) else best
    }
    r <- rbinom(1, 1, p_true[arm]); n[arm] <- n[arm] + 1; s[arm] <- s[arm] + r; arms[t] <- arm
  }
  regret <- cumsum(max(p_true) - p_true[arms])
  c(regret = regret[T_h], engaged = sum(s), share_best = mean(arms == which.max(p_true)),
    wrong_lock = as.numeric(names(which.max(table(factor(tail(arms, 100), levels = 1:k)))) != which.max(p_true)))
}
strategies <- c(equal = "Equal allocation", greedy = "Greedy", epsilon = "Epsilon-greedy (0.1)",
                thompson = "Thompson sampling")
band_sum <- do.call(rbind, lapply(names(strategies), function(st) {
  res <- t(sapply(1:reps, function(i) run_bandit(st, seed = 5000 + i)))
  data.frame(strategy = st, label = strategies[[st]],
             regret = round(mean(res[, "regret"]), 2), regret_sd = round(sd(res[, "regret"]), 2),
             engaged = round(mean(res[, "engaged"]), 1),
             share_best = round(mean(res[, "share_best"]), 4),
             wrong_lock = round(mean(res[, "wrong_lock"]), 4))
}))
write.csv(band_sum, file.path(out, "bandit_summary.csv"), row.names = FALSE)

# ---- 4. Scalars used in the text and the JSON spec used by the widgets -------
first90 <- ql$curve$episode[which(ql$curve$near_optimal >= 0.9)[1]]
scal <- data.frame(key = c("ql_first90", "ql_near_opt_final", "ql_in_range", "ql_hypo", "V5_target", "V5_hypo",
                           "bandit_T", "bandit_reps", "gamma",
                           "off_unseen", "off_in_range", "off_hypo"),
                   value = c(first90, tail(ql$curve$near_optimal, 1), ql_eval[["in_range"]], ql_eval[["hypo"]],
                             V5[3], V5[1], T_h, reps, gamma,
                             off$unseen, off_eval[["in_range"]], off_eval[["hypo"]]))
write.csv(scal, file.path(out, "scalars.csv"), row.names = FALSE)

spec <- list(bands = bands, rates = rates, actions = acts,
             moves = unname(split(moves, row(moves))), presets = presets, gamma = gamma,
             start = list(band = 4, rate = 0), nudges = nudges, p_true = p_true)
writeLines(jsonlite::toJSON(spec, auto_unbox = TRUE, pretty = TRUE),
           file.path(out, "insulin_mdp.json"))
print(vi_tab[, c("state", "best", "V")]); print(pres); print(band_sum); print(scal)
