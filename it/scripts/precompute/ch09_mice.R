# Precomputed results for Chapter 9 (When data are missing).
# Runs mice locally so the book renders fast and the browser needs only base R / pandas.
# Usage (from the book root): Rscript scripts/precompute/ch09_mice.R
# Output: data/precomputed/ch09/*.csv  (about 1 minute on one CPU)

suppressPackageStartupMessages(library(mice))
out <- "data/precomputed/ch09"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
wcsv <- function(x, f) write.csv(x, file.path(out, f), row.names = FALSE)

hb <- read.csv("data/homebeat.csv")            # complete data (known only in a simulation)
hm <- read.csv("data/homebeat_missing.csv")    # what the trial team actually has
f  <- stress_score ~ frailty + hrv + resting_hr
terms <- c("frailty", "hrv", "resting_hr")

# Variables given to the imputation model: everything the team observes.
# participant_id is an identifier and event_risk is the simulator's true risk
# (never observed), so both are left out. The outcome stress_score IS included.
imp_vars <- c("age", "baseline_inflammation", "frailty", "resting_hr", "hrv",
              "skin_temp", "activity_index", "stress_score", "event_occurred")

# ---------------------------------------------------------------------------
# 1. Four analyses of the same question --------------------------------------
fit_full <- lm(f, data = hb)
fit_cc   <- lm(f, data = hm)                                # listwise deletion
hm_mean  <- hm
for (v in c("resting_hr", "hrv", "skin_temp", "activity_index"))
  hm_mean[[v]][is.na(hm_mean[[v]])] <- mean(hm[[v]], na.rm = TRUE)
fit_mean <- lm(f, data = hm_mean)                          # single mean imputation

imp50 <- mice(hm[, imp_vars], m = 50, method = "pmm", maxit = 10,
              seed = 2025, printFlag = FALSE)
fits50 <- with(imp50, lm(stress_score ~ frailty + hrv + resting_hr))
pool50 <- summary(pool(fits50), conf.int = TRUE)

tab <- function(fit, label) {
  s <- summary(fit)$coefficients[terms, , drop = FALSE]
  ci <- confint(fit)[terms, , drop = FALSE]
  data.frame(analysis = label, term = terms, estimate = s[, 1], se = s[, 2],
             lower = ci[, 1], upper = ci[, 2], n_used = nobs(fit))
}
mi_row <- pool50[pool50$term %in% terms, ]
models <- rbind(
  tab(fit_full, "Complete data (truth in this simulation)"),
  tab(fit_cc,   "Complete-case analysis"),
  tab(fit_mean, "Mean imputation"),
  data.frame(analysis = "Multiple imputation (M = 50)", term = as.character(mi_row$term),
             estimate = mi_row$estimate, se = mi_row$std.error,
             lower = mi_row$`2.5 %`, upper = mi_row$`97.5 %`, n_used = nrow(hm))
)
wcsv(models, "hb_models.csv")

# 2. Per-imputation estimates (for the pooling widget and the Rubin mission) --
per <- do.call(rbind, lapply(seq_along(fits50$analyses), function(m) {
  s <- summary(fits50$analyses[[m]])$coefficients[terms, ]
  data.frame(m = m, term = terms, estimate = s[, 1], se = s[, 2])
}))
wcsv(per, "mi_estimates.csv")

# 3. Diagnostics of a small run (M = 5): trace of chain means and imputed values
imp5 <- mice(hm[, imp_vars], m = 5, method = "pmm", maxit = 10,
             seed = 2025, printFlag = FALSE)
cm <- imp5$chainMean                       # variables x iterations x chains
trace <- do.call(rbind, lapply(c("hrv", "resting_hr"), function(v)
  do.call(rbind, lapply(1:5, function(ch)
    data.frame(variable = v, chain = ch, iteration = 1:10, mean = cm[v, , ch])))))
wcsv(trace, "trace.csv")
imputed <- do.call(rbind, lapply(1:5, function(m)
  data.frame(m = m, hrv = imp5$imp$hrv[, m])))
wcsv(imputed, "imputed_hrv.csv")
fits5 <- with(imp5, lm(stress_score ~ frailty + hrv + resting_hr))
pool5 <- summary(pool(fits5))
wcsv(data.frame(term = as.character(pool5$term), estimate = pool5$estimate,
                se = pool5$std.error, df = pool5$df), "pool_m5.csv")

# 4. Bias or bad luck? Repeat the deletion many times ------------------------
# Scenario A: the HomeBeat mechanisms for hrv (older participants) and
#             resting_hr (frailer participants), as in sim_homebeat.R.
# Scenario B: hrv is missing more often when the stress score (the outcome) is high.
set.seed(99)
R <- 200; n <- nrow(hb)
zs <- as.numeric(scale(hb$stress_score))
sims <- list()
for (scen in c("A", "B")) for (r in 1:R) {
  d <- hb[, imp_vars]
  if (scen == "A") {
    d$hrv[d$age > 60 & runif(n) < 0.28] <- NA
    d$resting_hr[d$frailty > quantile(d$frailty, 0.8) & runif(n) < 0.30] <- NA
  } else {
    d$hrv[runif(n) < plogis(-1.2 + 1.5 * zs)] <- NA
  }
  cc <- coef(lm(f, data = d))[terms]
  imp <- mice(d, m = 5, method = "pmm", maxit = 5, printFlag = FALSE, seed = r)
  mi <- summary(pool(with(imp, lm(stress_score ~ frailty + hrv + resting_hr))))
  mi <- setNames(mi$estimate, mi$term)[terms]
  sims[[length(sims) + 1]] <- data.frame(scenario = scen, rep = r, term = terms,
                                         cc = cc, mi = mi,
                                         p_incomplete = mean(!complete.cases(d[, c("hrv", "resting_hr")])))
}
sims <- do.call(rbind, sims)
full <- coef(fit_full)[terms]
simsum <- do.call(rbind, lapply(split(sims, list(sims$scenario, sims$term)), function(s)
  data.frame(scenario = s$scenario[1], term = s$term[1], truth = full[s$term[1]],
             cc_mean = mean(s$cc), cc_sd = sd(s$cc), mi_mean = mean(s$mi), mi_sd = sd(s$mi),
             p_incomplete = mean(s$p_incomplete), reps = nrow(s))))
wcsv(simsum, "bias_or_luck.csv")

# 5. MNAR: the didactic hydration_level variable of the first edition ---------
# Same code and seeds as the first edition: low hydration -> more likely missing.
set.seed(2027)
hydr_true <- rnorm(nrow(hm), mean = 50, sd = 10)
hm_h <- hm[, imp_vars]
hm_h$hydration_level <- hydr_true
p_miss <- plogis(-(hydr_true - 40) / 5)
set.seed(2028)
hm_h$hydration_level[runif(nrow(hm_h)) < p_miss] <- NA
wcsv(data.frame(hydration_true = round(hydr_true, 3),
                hydration_observed = round(hm_h$hydration_level, 3),
                p_missing = round(p_miss, 4)), "hydration.csv")

# Imputation under MAR, then delta-adjustment: imputed values are shifted by delta
# (mice 'post' argument) to represent "missing values are lower than similar observed ones".
deltas <- seq(0, -15, by = -1)
dgrid <- do.call(rbind, lapply(deltas, function(dl) {
  post <- make.post(hm_h)
  post["hydration_level"] <- sprintf("imp[[j]][, i] <- imp[[j]][, i] + (%s)", dl)
  imp <- mice(hm_h, m = 20, method = "pmm", maxit = 5, post = post,
              seed = 2028, printFlag = FALSE)
  s <- summary(pool(with(imp, lm(hydration_level ~ 1))), conf.int = TRUE)
  data.frame(delta = dl, estimate = s$estimate, lower = s$`2.5 %`, upper = s$`97.5 %`)
}))
dgrid$true_mean <- mean(hydr_true)
dgrid$observed_mean <- mean(hm_h$hydration_level, na.rm = TRUE)
wcsv(dgrid, "hydration_delta.csv")

cat("done\n")
