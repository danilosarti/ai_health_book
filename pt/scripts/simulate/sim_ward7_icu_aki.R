# =========================================================
# DATASET 2: ICU Acute Kidney Injury (AKI) Risk and Early Trajectory
# =========================================================

# Rationale:
# This simulation keeps:
# - 1250 ICU patients
# - one site variable
# - clinically interpretable ICU predictors
# - continuous renal outcome: delta_creatinine_48h
#
# The outcome is driven by illness severity, perfusion, renal baseline status,
# inflammation, and process-of-care quality. Site effects are included to mimic
# institutional variation.

set.seed(1002)

out_dir <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1] else "data"; dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

n <- 1250

patient_id <- sprintf("ICU_%04d", 1:n)
site <- sample(
  c("ICU_A", "ICU_B", "ICU_C", "ICU_D"),
  size = n, replace = TRUE, prob = c(0.24, 0.26, 0.25, 0.25)
)

age <- round(pmin(pmax(rnorm(n, mean = 64, sd = 15), 18), 95))

sofa <- pmin(pmax(round(rnorm(n, mean = 7, sd = 3)), 0), 18)

lactate <- round(pmax(rlnorm(n, meanlog = log(1.8), sdlog = 0.45), 0.3), 2)

map_min <- round(rnorm(n, mean = 68, sd = 10) - 0.9 * pmax(sofa - 7, 0), 1)
map_min <- pmax(map_min, 35)

baseline_creat <- round(
  pmax(
    rlnorm(n, meanlog = log(1.0), sdlog = 0.35) + 0.01 * (age - 64),
    0.4
  ),
  2
)

protocol_adherence <- plogis(
  rnorm(n, mean = 1.1, sd = 0.9) - 0.12 * pmax(sofa - 7, 0)
)
protocol_adherence <- round(protocol_adherence, 3)

bun <- round(pmax(8 + 9 * baseline_creat + 1.8 * sofa + rnorm(n, 0, 8), 4), 1)

crp <- round(pmax(rlnorm(n, meanlog = log(35), sdlog = 0.7) + 2.2 * sofa, 1), 1)

albumin <- round(
  pmin(
    pmax(
      rnorm(n, mean = 3.4, sd = 0.45) - 0.05 * sofa - 0.08 * log(crp),
      1.5
    ),
    5.2
  ),
  2
)

site_effect <- c(ICU_A = -0.03, ICU_B = 0.00, ICU_C = 0.06, ICU_D = 0.10)

# Outcome logic:
# kidney deterioration tends to worsen with severity, lactate, low MAP,
# higher baseline creatinine, inflammation, and lower albumin.
# Better protocol adherence is mildly protective.
delta_creatinine_48h <-
  0.04 * sofa +
  0.10 * log(lactate + 0.1) -
  0.006 * (map_min - 65) +
  0.22 * baseline_creat -
  0.28 * protocol_adherence +
  0.0035 * bun +
  0.0012 * crp -
  0.08 * albumin +
  unname(site_effect[site]) +
  rnorm(n, mean = 0, sd = 0.18)

delta_creatinine_48h <- round(delta_creatinine_48h, 3)

ds02_icu_aki_FULL <- data.frame(
  patient_id = patient_id,
  site = site,
  age = age,
  sofa = sofa,
  lactate = lactate,
  map_min = map_min,
  baseline_creat = baseline_creat,
  protocol_adherence = protocol_adherence,
  bun = bun,
  crp = crp,
  albumin = albumin,
  delta_creatinine_48h = delta_creatinine_48h,
  check.names = FALSE
)

write.csv(
  ds02_icu_aki_FULL,
  file = file.path(out_dir, "icu_aki.csv"),
  row.names = FALSE
)

# --- VERIFICATION BLOCK ---

cat("Dataset 2 dimensions:", dim(ds02_icu_aki_FULL), "\n")

expected_cols <- c(
  "patient_id", "site", "age", "sofa", "lactate", "map_min",
  "baseline_creat", "protocol_adherence", "bun", "crp",
  "albumin", "delta_creatinine_48h"
)

missing_cols <- setdiff(expected_cols, colnames(ds02_icu_aki_FULL))
cat("Missing columns:", length(missing_cols), "\n")

all_present <- all(expected_cols %in% colnames(ds02_icu_aki_FULL))
cat("All required features present:", all_present, "\n")

str(ds02_icu_aki_FULL)