# =====================================================================
# Chapter 13: precomputed results (from prediction to causation)
# Usage (from the book folder): Rscript scripts/precompute/ch13_precompute.R
# Needs base R only. Runs in a few seconds. Writes small CSV/JSON files to
# data/precomputed/ch13/ and (re)creates data/ward7_dag.csv and
# data/onco1_routine.csv.
# =====================================================================
out <- "data/precomputed/ch13"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
source("scripts/simulate/ward7_scm.R")
source("scripts/simulate/sim_ch13_onco1_routine.R")
system("Rscript scripts/simulate/sim_ch13_ward7_dag.R")
system("Rscript scripts/simulate/sim_ch13_onco1_routine.R")

tojson <- function(x) {             # minimal JSON writer for data frames / named lists
  if (is.data.frame(x)) {
    rows <- apply(x, 1, function(r) paste0("{", paste0('"', names(x), '":',
             ifelse(sapply(x, is.numeric), r, paste0('"', r, '"')), collapse = ","), "}"))
    return(paste0("[", paste(gsub(" ", "", rows), collapse = ",\n"), "]"))
  }
  if (!is.list(x)) return(if (is.numeric(x)) as.character(x) else paste0('"', x, '"'))
  paste0("{", paste0('"', names(x), '":', sapply(x, tojson), collapse = ",\n"), "}")
}

# ---- 1. Ward 7: the adherence coefficient under different adjustment sets ----
icu <- read.csv("data/icu_aki.csv")
est <- function(rhs, d = icu) {
  m <- lm(as.formula(paste("delta_creatinine_48h ~ protocol_adherence", rhs)), data = d)
  s <- summary(m)$coefficients["protocol_adherence", ]
  ci <- confint(m)["protocol_adherence", ]
  c(estimate = s[[1]], se = s[[2]], lo = ci[[1]], hi = ci[[2]])
}
others <- "+ age + lactate + map_min + baseline_creat + bun + crp + albumin + site"
sets <- list(
  c("Nothing (naive)", ""),
  c("SOFA", "+ sofa"),
  c("SOFA and every other recorded variable", paste("+ sofa", others)),
  c("Every recorded variable except SOFA", others),
  c("Lowest blood pressure only (a proxy of SOFA)", "+ map_min"),
  c("Lactate and baseline creatinine (causes of the outcome only)", "+ lactate + baseline_creat"))
adj <- do.call(rbind, lapply(sets, function(s) data.frame(set = s[1], t(est(s[2])))))
write.csv(transform(adj, estimate = round(estimate, 4), se = round(se, 4),
                    lo = round(lo, 4), hi = round(hi, 4)),
          file.path(out, "ward7_adjustment.csv"), row.names = FALSE)

# ---- 2. Ward 7: what the simulator knows ------------------------------------
y_obs <- ward7_scm()$delta_creatinine_48h
stopifnot(isTRUE(all.equal(y_obs, icu$delta_creatinine_48h)))
policy <- function(a) pmax(a, 0.9)                      # "at least 90 % adherence for everyone"
y_pol <- ward7_scm(do = list(protocol_adherence = policy))$delta_creatinine_48h
y_a1  <- ward7_scm(do = list(protocol_adherence = 1))$delta_creatinine_48h
y_a0  <- ward7_scm(do = list(protocol_adherence = 0))$delta_creatinine_48h
y_s1  <- ward7_scm(do = list(sofa = function(s) s + 1))$delta_creatinine_48h
incr  <- policy(icu$protocol_adherence) - icu$protocol_adherence
b_naive <- adj$estimate[1]; b_adj <- adj$estimate[2]
full <- lm(delta_creatinine_48h ~ sofa + age + lactate + map_min + baseline_creat +
             protocol_adherence + bun + crp + albumin + site, data = icu)
truth <- data.frame(key = c(
  "ate_per_unit", "policy_mean_increase", "policy_true_change", "policy_naive_change",
  "policy_adj_change", "aki_observed", "aki_policy_true", "aki_policy_naive", "aki_policy_adj",
  "sofa_total_effect", "sofa_direct_coef_full_model", "sofa_simulation_direct"),
  value = c(mean(y_a1 - y_a0), mean(incr), mean(y_pol - y_obs), b_naive * mean(incr),
            b_adj * mean(incr), mean(y_obs >= 0.3), mean(y_pol >= 0.3),
            mean(y_obs + b_naive * incr >= 0.3), mean(y_obs + b_adj * incr >= 0.3),
            mean(y_s1 - y_obs), coef(full)[["sofa"]], 0.04))
truth$value <- round(truth$value, 5)
write.csv(truth, file.path(out, "ward7_truth.csv"), row.names = FALSE)

# potential outcomes of six patients (only a simulation can fill both columns)
y_50 <- ward7_scm(do = list(protocol_adherence = 0.5))$delta_creatinine_48h
y_90 <- ward7_scm(do = list(protocol_adherence = 0.9))$delta_creatinine_48h
ids <- c(2, 11, 25, 48, 103, 777)
po <- data.frame(patient_id = icu$patient_id[ids], sofa = icu$sofa[ids],
                 adherence = icu$protocol_adherence[ids], observed = icu$delta_creatinine_48h[ids],
                 y_at_50 = round(y_50[ids], 3), y_at_90 = round(y_90[ids], 3))
write.csv(po, file.path(out, "ward7_potential_outcomes.csv"), row.names = FALSE)

# ---- 3. Ward 7: good predictor, wrong effect (shared split) -----------------
split <- read.csv("data/splits/ward7_split.csv")
icu_s <- merge(icu, split, by = "patient_id")
tr <- icu_s[icu_s$set == "train", ]; te <- icu_s[icu_s$set == "test", ]
pm <- function(label, rhs) {
  m <- lm(as.formula(paste("delta_creatinine_48h ~ protocol_adherence", rhs)), data = tr)
  data.frame(model = label, test_mae = mean(abs(te$delta_creatinine_48h - predict(m, te))),
             adherence_coef = coef(m)[["protocol_adherence"]])
}
pvc <- rbind(pm("Adherence only", ""),
             pm("Adherence + SOFA", "+ sofa"),
             pm("Adherence + every variable except SOFA", others),
             pm("Adherence + every variable", paste("+ sofa", others)))
pvc$test_mae <- round(pvc$test_mae, 4); pvc$adherence_coef <- round(pvc$adherence_coef, 4)
write.csv(pvc, file.path(out, "ward7_prediction_vs_causation.csv"), row.names = FALSE)

# ---- 4. DAG widget: every adjustment set in "Ward 7, extended" --------------
dag <- read.csv("data/ward7_dag.csv")
cand <- c("sofa", "baseline_creat", "staffing", "nephrotox", "creat_tests")
ext_a1 <- ward7_scm(seed = 1313, extended = TRUE, do = list(protocol_adherence = 1))
ext_a0 <- ward7_scm(seed = 1313, extended = TRUE, do = list(protocol_adherence = 0))
ext_truth <- mean(ext_a1$delta_creatinine_48h - ext_a0$delta_creatinine_48h)
rows <- list()
for (code in 0:31) {
  use <- cand[bitwAnd(code, 2^(0:4)) > 0]
  rhs <- if (length(use)) paste("+", paste(use, collapse = " + ")) else ""
  e <- est(rhs, dag)
  rows[[length(rows) + 1]] <- data.frame(code = code, set = paste(use, collapse = "|"),
                                         estimate = round(e[["estimate"]], 4), se = round(e[["se"]], 4),
                                         lo = round(e[["lo"]], 4), hi = round(e[["hi"]], 4))
}
dagw <- do.call(rbind, rows)
writeLines(tojson(list(truth = round(ext_truth, 4), sets = dagw)),
           file.path(out, "dag_widget.json"))

# ---- 5. ONCO-1: randomised benchmark and the routine-practice twin ---------
onco <- read.csv("data/onco1.csv")
onco$chemo <- as.integer(onco$treatment == "chemo")
trial_ate <- with(onco, mean(response_percent[chemo == 1]) - mean(response_percent[chemo == 0]))
covs <- "patient_age + tumor_grade + performance_score + gene_05"

ps_fit <- function(d) fitted(glm(as.formula(paste("chemo ~", covs)), family = binomial, data = d))
ipw <- function(d, ps) {
  w <- ifelse(d$chemo == 1, 1 / ps, 1 / (1 - ps))
  weighted.mean(d$response_percent[d$chemo == 1], w[d$chemo == 1]) -
    weighted.mean(d$response_percent[d$chemo == 0], w[d$chemo == 0])
}
match_nn <- function(d, ps, caliper = 0.2, seed = 1) {   # greedy 1:1, no replacement
  set.seed(seed)
  lp <- qlogis(ps); cal <- caliper * sd(lp)
  tr <- sample(which(d$chemo == 1)); pool <- which(d$chemo == 0); pairs <- NULL
  for (i in tr) {
    if (!length(pool)) break
    dist <- abs(lp[pool] - lp[i]); j <- which.min(dist)
    if (dist[j] <= cal) { pairs <- rbind(pairs, c(i, pool[j])); pool <- pool[-j] }
  }
  pairs
}
estimators <- function(d) {
  ps <- ps_fit(d); pr <- match_nn(d, ps)
  c(naive = mean(d$response_percent[d$chemo == 1]) - mean(d$response_percent[d$chemo == 0]),
    regression = coef(lm(as.formula(paste("response_percent ~ chemo +", covs)), data = d))[["chemo"]],
    ipw = ipw(d, ps),
    matching = mean(d$response_percent[pr[, 1]] - d$response_percent[pr[, 2]]))
}
smd <- function(x, t, w = rep(1, length(x))) {
  m1 <- weighted.mean(x[t == 1], w[t == 1]); m0 <- weighted.mean(x[t == 0], w[t == 0])
  (m1 - m0) / sqrt((var(x[t == 1]) + var(x[t == 0])) / 2)
}

# 5a. the static dataset used in the text (k = 1)
rt <- read.csv("data/onco1_routine.csv")
ps <- ps_fit(rt); pr <- match_nn(rt, ps)
att_bench <- with(onco, {                                  # trial estimate of the ATT
  e <- plogis(onco1_score(onco))
  weighted.mean(response_percent[chemo == 1], e[chemo == 1]) -
    weighted.mean(response_percent[chemo == 0], e[chemo == 0])
})
e1 <- estimators(rt)
est_tab <- data.frame(key = c("trial_ate", "trial_att", "n_routine", "n_treated", "n_pairs",
                              names(e1), "ps_min", "ps_max"),
                      value = round(c(trial_ate, att_bench, nrow(rt), sum(rt$chemo), nrow(pr),
                                      e1, min(ps), max(ps)), 4))
write.csv(est_tab, file.path(out, "onco_routine_estimates.csv"), row.names = FALSE)

X <- data.frame(age = rt$patient_age, grade_G3 = as.integer(rt$tumor_grade == "G3"),
                grade_G1 = as.integer(rt$tumor_grade == "G1"),
                performance = rt$performance_score, gene_05 = rt$gene_05)
w <- ifelse(rt$chemo == 1, 1 / ps, 1 / (1 - ps))
inm <- seq_len(nrow(rt)) %in% c(pr)
bal <- data.frame(variable = names(X),
                  smd_before = sapply(X, smd, t = rt$chemo),
                  smd_ipw = sapply(X, smd, t = rt$chemo, w = w),
                  smd_matched = sapply(X, function(x) {
                    xx <- x[inm]; tt <- rt$chemo[inm]
                    (mean(xx[tt == 1]) - mean(xx[tt == 0])) / sqrt((var(x[rt$chemo == 1]) + var(x[rt$chemo == 0])) / 2)
                  }))
bal[-1] <- round(bal[-1], 3)
write.csv(bal, file.path(out, "onco_routine_balance.csv"), row.names = FALSE)
write.csv(data.frame(chemo = rt$chemo, ps = round(ps, 4)),
          file.path(out, "onco_routine_ps.csv"), row.names = FALSE)

# 5b. widget: how strong is the oncologists' selection? (100 replicates per k)
ks <- c(0, 0.5, 1, 1.5, 2, 2.5, 3)
wid <- do.call(rbind, lapply(ks, function(k) {
  r <- t(sapply(1:100, function(s) estimators(make_routine(onco, k = k, seed = 5000 + s))))
  do.call(rbind, lapply(colnames(r), function(m)
    data.frame(k = k, method = m, mean = round(mean(r[, m]), 3),
               lo = round(quantile(r[, m], 0.05)[[1]], 3), hi = round(quantile(r[, m], 0.95)[[1]], 3))))
}))
psd <- do.call(rbind, lapply(ks, function(k) {
  d <- make_routine(onco, k = k, seed = 1313); p <- ps_fit(d)
  data.frame(k = k, chemo = d$chemo, ps = round(p, 3))
}))
writeLines(tojson(list(trial = round(trial_ate, 3), summary = wid, ps = psd)),
           file.path(out, "onco_selection_widget.json"))
cat("done\n")
