# =====================================================================
# Chapter 13 — "ONCO-1 in routine practice": an observational twin of
# the randomised ONCO-1 trial (data/onco1.csv).
#
# Idea: in ONCO-1 a coin decided who received chemotherapy. Here we
# imagine that, instead, oncologists chose: they offered chemotherapy more
# often to younger and fitter patients, to higher-grade tumours and to
# patients with low expression of gene_05 (a gene a previous study had
# linked with poor response). We mimic this by keeping each trial patient
# with a probability that depends on the arm and on these covariates:
#     treated patients are kept with probability e(X),
#     control patients with probability 1 - e(X),
# where e(X) = plogis(k * score(X)). Because the trial was 1:1, the kept
# patients have P(chemo | X) = e(X) — the propensity score is known by
# construction — and the same covariate distribution as the trial, so the
# trial's difference in means is a fair benchmark for the average effect.
# k = 0 is "no selection" (a smaller randomised trial); larger k means
# stronger selection by the oncologists.
#
# Usage (from the book folder): Rscript scripts/simulate/sim_ch13_onco1_routine.R
# Writes data/onco1_routine.csv (k = 1, seed 1313).
# =====================================================================
onco1_score <- function(d) {
  z05 <- (d$gene_05 - 7.5) / 0.5                    # gene_05 on a standard-deviation scale
  0.2 - 0.4 * (d$patient_age - 55) / 10 +
    c(G1 = -0.5, G2 = 0, G3 = 0.8)[as.character(d$tumor_grade)] -
    0.7 * d$performance_score - 1.0 * z05
}

make_routine <- function(onco, k = 1, seed = 1313) {
  set.seed(seed)
  e <- plogis(k * onco1_score(onco))
  chemo <- as.integer(onco$treatment == "chemo")
  keep <- runif(nrow(onco)) < ifelse(chemo == 1, e, 1 - e)
  out <- data.frame(patient_id = onco$patient_id, chemo = chemo,
                    patient_age = onco$patient_age, tumor_grade = onco$tumor_grade,
                    performance_score = onco$performance_score,
                    gene_05 = onco$gene_05, dose_intensity = onco$dose_intensity,
                    response_percent = onco$response_percent,
                    true_ps = round(e, 4))[keep, ]
  rownames(out) <- NULL
  out
}

if (sys.nframe() == 0) {
  onco <- read.csv("data/onco1.csv")
  routine <- make_routine(onco, k = 1, seed = 1313)
  routine$true_ps <- NULL                            # in real life nobody knows it
  write.csv(routine, "data/onco1_routine.csv", row.names = FALSE)
  cat("onco1_routine.csv:", nrow(routine), "patients,", sum(routine$chemo), "treated\n")
}
