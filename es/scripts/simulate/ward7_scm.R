# =====================================================================
# Ward 7 as a structural causal model (used by Chapter 13)
#
# ward7_scm() re-implements, line by line and in the same random-number
# order, the generating code of scripts/simulate/sim_ward7_icu_aki.R.
# With seed = 1002 and no intervention it reproduces data/icu_aki.csv
# exactly. Its extra argument `do` sets a variable to a chosen value for
# every patient *after* its random draw, so that all other random draws
# (each patient's "noise") stay the same: the result is each patient's
# potential outcome under that intervention.
#
#   do = list(protocol_adherence = 0.9)        # everyone at 90 % adherence
#   do = list(protocol_adherence = function(a) pmax(a, 0.9))   # a policy
#   do = list(sofa = function(s) s + 1)        # one more SOFA point
#
# extended = TRUE adds three variables used only by the Chapter 13 DAG
# widget (see sim_ch13_ward7_dag.R): nurse staffing (affects adherence),
# a nephrotoxic-drug exposure score (a mediator) and the number of
# creatinine tests ordered in 48 h (a collider). In that version the effect of adherence is split into a
# direct part (-0.18) and a part through the mediator (-2.0 x 0.05 = -0.10),
# so that its total effect is still -0.28 per unit, as in Ward 7.
# =====================================================================
ward7_scm <- function(n = 1250, seed = 1002, do = list(), extended = FALSE) {
  set.seed(seed)
  apply_do <- function(name, x) {
    if (!is.null(do[[name]])) {
      v <- do[[name]]
      x <- if (is.function(v)) v(x) else rep(v, length.out = length(x))
    }
    x
  }
  if (extended) {
    staffing <- round(rnorm(n, mean = 0, sd = 1), 2)        # standardised nurse staffing
    staffing <- apply_do("staffing", staffing)
  }
  patient_id <- sprintf("ICU_%04d", 1:n)
  site <- sample(c("ICU_A", "ICU_B", "ICU_C", "ICU_D"), size = n, replace = TRUE,
                 prob = c(0.24, 0.26, 0.25, 0.25))
  age <- round(pmin(pmax(rnorm(n, mean = 64, sd = 15), 18), 95))
  sofa <- pmin(pmax(round(rnorm(n, mean = 7, sd = 3)), 0), 18)
  sofa <- apply_do("sofa", sofa)
  lactate <- round(pmax(rlnorm(n, meanlog = log(1.8), sdlog = 0.45), 0.3), 2)
  map_min <- round(rnorm(n, mean = 68, sd = 10) - 0.9 * pmax(sofa - 7, 0), 1)
  map_min <- pmax(map_min, 35)
  baseline_creat <- round(pmax(rlnorm(n, meanlog = log(1.0), sdlog = 0.35) +
                                 0.01 * (age - 64), 0.4), 2)
  u_adh <- rnorm(n, mean = 1.1, sd = 0.9)
  protocol_adherence <- plogis(u_adh - 0.12 * pmax(sofa - 7, 0) +
                                 (if (extended) 0.6 * staffing else 0))
  protocol_adherence <- round(protocol_adherence, 3)
  protocol_adherence <- apply_do("protocol_adherence", protocol_adherence)
  bun <- round(pmax(8 + 9 * baseline_creat + 1.8 * sofa + rnorm(n, 0, 8), 4), 1)
  crp <- round(pmax(rlnorm(n, meanlog = log(35), sdlog = 0.7) + 2.2 * sofa, 1), 1)
  albumin <- round(pmin(pmax(rnorm(n, mean = 3.4, sd = 0.45) - 0.05 * sofa -
                               0.08 * log(crp), 1.5), 5.2), 2)
  site_effect <- c(ICU_A = -0.03, ICU_B = 0.00, ICU_C = 0.06, ICU_D = 0.10)
  if (extended) {
    nephrotox <- round(pmax(2.6 - 2.0 * protocol_adherence + rnorm(n, 0, 0.3), 0), 2)
    nephrotox <- apply_do("nephrotox", nephrotox)
    b_adh <- -0.18; b_nep <- 0.05
  } else {
    nephrotox <- 0; b_adh <- -0.28; b_nep <- 0
  }
  eps <- rnorm(n, mean = 0, sd = 0.18)
  delta <- 0.04 * sofa + 0.10 * log(lactate + 0.1) - 0.006 * (map_min - 65) +
    0.22 * baseline_creat + b_adh * protocol_adherence + b_nep * nephrotox +
    0.0035 * bun + 0.0012 * crp - 0.08 * albumin + unname(site_effect[site]) + eps
  delta <- round(delta, 3)
  out <- data.frame(patient_id, site, age, sofa, lactate, map_min, baseline_creat,
                    protocol_adherence, bun, crp, albumin,
                    delta_creatinine_48h = delta)
  if (extended) {
    # creatinine tests ordered in 48 h: more with a rising creatinine (worried
    # clinicians re-test) AND with high adherence (the protocol asks for
    # creatinine every few hours) -> a collider between adherence and outcome
    creat_tests <- pmax(round(1 + 4 * protocol_adherence + 6 * pmax(delta, 0) +
                                rnorm(n, 0, 0.8)), 0)
    out$staffing <- staffing
    out$nephrotox <- nephrotox
    out$creat_tests <- creat_tests
  }
  out
}
