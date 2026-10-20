# =====================================================================
# Chapter 8 — Bayesian models fitted with Stan (rstanarm), precomputed.
# Usage (from the book folder): Rscript scripts/precompute/ch08_stan.R
# Writes small CSV/JSON files to data/precomputed/ch08/.
# Needs: rstanarm, posterior. Runtime: about two minutes on one core.
# MCMC is kept deliberately small (2-3 chains, 2 000 iterations).
# =====================================================================
suppressPackageStartupMessages({
  library(rstanarm)
  library(posterior)
})
options(mc.cores = 1)
SEED <- 2026
out <- "data/precomputed/ch08"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
w <- function(x, f) write.csv(x, file.path(out, f), row.names = FALSE)

summ <- function(fit, pars = NULL) {
  dr <- as_draws_df(as.array(fit))
  s <- summarise_draws(dr, mean, median, sd, ~quantile(.x, c(0.025, 0.975)), rhat, ess_bulk, ess_tail)
  s <- as.data.frame(s)
  names(s)[5:6] <- c("q2.5", "q97.5")
  s
}

# ---- 1. HomeBeat: resting heart rate ~ frailty (the v1 model) --------
hb <- read.csv("data/homebeat.csv")
fit_simple <- stan_glm(
  resting_hr ~ frailty, data = hb, family = gaussian(),
  prior_intercept = normal(60, 20, autoscale = FALSE),  # HR of a participant of average frailty
  prior           = normal(0, 20, autoscale = FALSE),   # slope, bpm per unit of frailty
  prior_aux       = normal(0, 10, autoscale = FALSE),   # sigma > 0: a half-normal(10)
  chains = 2, iter = 2000, seed = SEED, refresh = 0)
s1 <- summ(fit_simple); s1$model <- "simple"
d1 <- as.data.frame(fit_simple)
p_pos_simple <- mean(d1$frailty > 0)

# ---- 2. HomeBeat: adjusted for age and inflammation -------------------
fit_adj <- stan_glm(
  resting_hr ~ frailty + age + baseline_inflammation, data = hb, family = gaussian(),
  prior_intercept = normal(60, 20, autoscale = FALSE),
  prior           = normal(0, 20, autoscale = FALSE),
  prior_aux       = normal(0, 10, autoscale = FALSE),
  chains = 2, iter = 2000, seed = SEED, refresh = 0)
s2 <- summ(fit_adj); s2$model <- "adjusted"
d2 <- as.data.frame(fit_adj)
names(d2) <- c("b0", "b_frailty", "b_age", "b_infl", "sigma")
w(round(d2, 4), "homebeat_adjusted_draws.csv")           # 2 000 draws, used in the browser
w(rbind(s1, s2), "homebeat_summaries.csv")

# Draws of the simple model, chain by chain (for the trace plot and densities)
a1 <- as.array(fit_simple)                                 # iterations x chains x parameters
tr <- data.frame(chain = rep(1:2, each = dim(a1)[1]), iter = rep(seq_len(dim(a1)[1]), 2),
                 b0 = round(c(a1[, 1, 1], a1[, 2, 1]), 3),
                 b_frailty = round(c(a1[, 1, 2], a1[, 2, 2]), 3),
                 sigma = round(c(a1[, 1, 3], a1[, 2, 3]), 3))
w(tr, "homebeat_simple_draws.csv")

# Lena's profile (Chapter 2: resting HR 88 bpm)
lena <- data.frame(age = 72, frailty = 0.55, baseline_inflammation = 4.1)
set.seed(8)                                                # same seed as Mission 3 and the Lab
mu_lena <- d2$b0 + d2$b_frailty * lena$frailty + d2$b_age * lena$age + d2$b_infl * lena$baseline_inflammation
y_lena  <- rnorm(length(mu_lena), mu_lena, d2$sigma)
lena_s <- data.frame(quantity = c("mean_hr", "new_hr"),
                     mean = c(mean(mu_lena), mean(y_lena)),
                     q2.5 = c(quantile(mu_lena, 0.025), quantile(y_lena, 0.025)),
                     q97.5 = c(quantile(mu_lena, 0.975), quantile(y_lena, 0.975)),
                     p_above_88 = c(mean(mu_lena > 88), mean(y_lena > 88)))
w(lena_s, "homebeat_lena.csv")

# ---- 3. Ward 7: Bayesian logistic regression for AKI -----------------
icu <- read.csv("data/icu_aki.csv")
icu$aki <- as.integer(icu$delta_creatinine_48h >= 0.3)
split <- read.csv("data/splits/ward7_split.csv")           # shared split (Chapter 3)
icu <- merge(icu, split, by = "patient_id", sort = FALSE)
icu <- icu[order(icu$patient_id), ]
train <- icu[icu$set == "train", ]; test <- icu[icu$set == "test", ]
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")

f_model <- aki ~ sofa + lactate + map_min + baseline_creat + protocol_adherence   # as in Chapter 5
fit_aki <- stan_glm(f_model, data = train, family = binomial(link = "logit"),
                    prior = normal(0, 2, autoscale = FALSE),
                    prior_intercept = normal(0, 5, autoscale = FALSE),
                    chains = 2, iter = 2000, seed = SEED, refresh = 0)
s3 <- summ(fit_aki)
w(s3, "ward7_logistic_summary.csv")
risk <- posterior_epred(fit_aki, newdata = tonight)        # 2 000 x 3 draws of each patient's risk
colnames(risk) <- c("aoife", "tomas", "priya")
w(round(as.data.frame(risk), 4), "ward7_tonight_risk_draws.csv")
glm_fit <- glm(f_model, family = binomial, data = train)
pt <- posterior_epred(fit_aki, newdata = test)
test_pred <- data.frame(patient_id = test$patient_id, aki = test$aki,
                        p_bayes = round(colMeans(pt), 4),            # posterior mean per patient
                        lo = round(apply(pt, 2, quantile, 0.025), 4),
                        hi = round(apply(pt, 2, quantile, 0.975), 4),
                        p_glm = round(predict(glm_fit, test, type = "response"), 4))
w(test_pred, "ward7_test_bayes_predictions.csv")

# A small-data version: the first 40 training patients of ICU_D only
small <- head(train[train$site == "ICU_D", ], 40)
fit_small <- stan_glm(f_model, data = small, family = binomial(link = "logit"),
                      prior = normal(0, 2, autoscale = FALSE),
                      prior_intercept = normal(0, 5, autoscale = FALSE),
                      chains = 2, iter = 2000, seed = SEED, refresh = 0)
risk_small <- posterior_epred(fit_small, newdata = tonight)
colnames(risk_small) <- c("aoife", "tomas", "priya")
w(round(as.data.frame(risk_small), 4), "ward7_tonight_risk_draws_small.csv")

# ---- 4. Good and bad chains: the trace-plot quiz -----------------------
# Panel data for one parameter (the frailty slope of model 1), three chains of 500 kept draws.
X <- cbind(1, hb$frailty); y <- hb$resting_hr
logpost <- function(th) {                                  # th = (b0, b1, log sigma)
  s <- exp(th[3]); mu <- X %*% th[1:2]
  sum(dnorm(y, mu, s, log = TRUE)) + dnorm(th[1], 60, 20, log = TRUE) +
    dnorm(th[2], 0, 20, log = TRUE) + dnorm(s, 0, 10, log = TRUE) + th[3]
}
rwm <- function(start, step, n, seed) {                    # random-walk Metropolis
  set.seed(seed); th <- start; lp <- logpost(th); outm <- matrix(NA, n, 3)
  for (i in 1:n) {
    prop <- th + rnorm(3, 0, step); lpp <- logpost(prop)
    if (log(runif(1)) < lpp - lp) { th <- prop; lp <- lpp }
    outm[i, ] <- th
  }
  outm
}
ols <- coef(lm(y ~ hb$frailty)); s_ols <- log(sigma(lm(y ~ hb$frailty)))
# (a) well tuned Stan: 3 chains, 1 000 iterations, 500 warm-up
fit_q <- update(fit_simple, chains = 3, iter = 1000, seed = SEED + 1)
good <- lapply(1:3, function(k) as.array(fit_q)[, k, "frailty"])
# (b) not yet converged: dispersed starts, small steps, no warm-up removed
starts <- list(c(ols[1] + 25, ols[2] - 60, s_ols), c(ols[1] - 20, ols[2] + 55, s_ols), c(ols[1], ols[2], s_ols))
notconv <- lapply(1:3, function(k) rwm(starts[[k]], c(0.25, 0.6, 0.01), 500, SEED + 10 + k)[, 2])
# (c) sticky: proposals far too large, most rejected
sticky <- lapply(1:3, function(k) rwm(c(ols[1], ols[2], s_ols) + c(0, (k - 2) * 3, 0),
                                      c(6, 18, 0.4), 500, SEED + 20 + k)[, 2])
# (d) slow mixing: tiny steps, chains start in the right place but wander slowly
slow <- lapply(1:3, function(k) rwm(c(ols[1], ols[2] + (k - 2) * 4, s_ols),
                                    c(0.08, 0.25, 0.004), 500, SEED + 30 + k)[, 2])
panels <- list(C = good, A = notconv, D = sticky, B = slow)   # shown in the order A, B, C, D
diag_tab <- do.call(rbind, lapply(names(panels), function(p) {
  m <- do.call(cbind, panels[[p]])
  data.frame(panel = p, rhat = rhat(m), ess_bulk = ess_bulk(m))
}))
diag_tab <- diag_tab[order(diag_tab$panel), ]
w(diag_tab, "trace_quiz_diagnostics.csv")
tq <- do.call(rbind, lapply(names(panels), function(p) do.call(rbind, lapply(1:3, function(k)
  data.frame(panel = p, chain = k, iter = seq_along(panels[[p]][[k]]), value = round(panels[[p]][[k]], 2))))))
tq <- tq[order(tq$panel, tq$chain, tq$iter), ]
w(tq, "trace_quiz.csv")

# ---- 5. Versions -----------------------------------------------------------
writeLines(sprintf('{"R": "%s", "rstanarm": "%s", "rstan": "%s", "posterior": "%s", "seed": %d, "p_frailty_positive_simple": %.4f}',
                   getRversion(), packageVersion("rstanarm"), packageVersion("rstan"),
                   packageVersion("posterior"), SEED, p_pos_simple),
           file.path(out, "stan_versions.json"))
cat("Done.\n")
