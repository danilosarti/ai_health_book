# =====================================================================
# HomeBeat digital (decentralised) trial with wearables — simulation
# (Running example 3). Source: v1 book, introd_bayesian.qmd (author:
# D. A. Sarti), unchanged except for the output path at the end.
# =====================================================================
## -----------------------------------------------------------
## Digital Clinical Trial (DCT) Wearable Dataset Simulation
## Complete dataset (no missing) + Amputated dataset (with missing)
## -----------------------------------------------------------

set.seed(2025)

# Number of patients
n <- 500

# -----------------------------------------------------------
# Baseline characteristics
# -----------------------------------------------------------

age <- round(runif(n, 18, 80))

baseline_inflammation <- rnorm(
  n,
  mean = 3 + 0.03 * (age - 50),
  sd = 1
)

frailty <- rnorm(
  n,
  mean = 0.4 + 0.004 * (age - 50),
  sd = 0.12
)

# -----------------------------------------------------------
# Wearable physiological signals (no missing)
# -----------------------------------------------------------

resting_hr <- rnorm(
  n,
  mean = 60 +
         0.5 * (age - 50) +
         4 * frailty +
         3 * baseline_inflammation,
  sd = 5
)

hrv <- rnorm(
  n,
  mean = 70 -
         0.6 * (age - 50) -
         10 * frailty -
         8 * baseline_inflammation,
  sd = 10
)

skin_temp <- rnorm(
  n,
  mean = 33.0 +
         0.015 * (age - 50) +
         0.5 * baseline_inflammation,
  sd = 0.35
)

activity_index <- rnorm(
  n,
  mean = 60 -
         15 * frailty -
         5 * baseline_inflammation,
  sd = 8
)

# -----------------------------------------------------------
# Digital biomarker endpoint: Stress Score
# -----------------------------------------------------------

stress_score <- 40 +
  0.35 * resting_hr +
  -0.30 * hrv +
  0.15 * age +
  6 * baseline_inflammation +
  12 * frailty +
  rnorm(n, 0, 6)

# -----------------------------------------------------------
# Clinical event risk
# -----------------------------------------------------------

linpred <- -5 +
  0.04 * stress_score +
  0.03 * age +
  0.8  * baseline_inflammation +
  1.5  * frailty

event_risk <- 1 / (1 + exp(-linpred))
event_occurred <- rbinom(n, 1, event_risk)

# -----------------------------------------------------------
# COMPLETE DATASET (no missing)
# -----------------------------------------------------------

dct_wearable_complete <- data.frame(
  age,
  baseline_inflammation,
  frailty,
  resting_hr,
  hrv,
  skin_temp,
  activity_index,
  stress_score,
  event_risk,
  event_occurred
)


# -----------------------------------------------------------
# AMPUTATED DATASET (MAR + MCAR)
# -----------------------------------------------------------

# Copy complete dataset
dct_wearable_amputated <- dct_wearable_complete

# MCAR: random dropout for activity
set.seed(2026)
drop_act <- sample(1:n, 45)
dct_wearable_amputated$activity_index[drop_act] <- NA

# MAR: older → HRV missing
dct_wearable_amputated$hrv[
  age > 60 & runif(n) < 0.28
] <- NA

# MAR: high stress → skin temp missing
dct_wearable_amputated$skin_temp[
  stress_score > quantile(stress_score, 0.75) &
    runif(n) < 0.25
] <- NA

# MAR: frail → resting HR missing
dct_wearable_amputated$resting_hr[
  frailty > quantile(frailty, 0.8) &
    runif(n) < 0.30
] <- NA

# -----------------------------------------------------------
# Save datasets
# -----------------------------------------------------------

out_dir <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1] else "data"
dir.create(out_dir, showWarnings = FALSE)
write.csv(dct_wearable_complete,  file.path(out_dir, "homebeat.csv"), row.names = FALSE)
write.csv(dct_wearable_amputated, file.path(out_dir, "homebeat_missing.csv"), row.names = FALSE)



