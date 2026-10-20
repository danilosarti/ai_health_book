# =========================================================
# DATASET 5 — Cell & Gene Therapy Manufacturing
# =========================================================

# Rationale:
# This simulation preserves:
# - 420 manufacturing batches
# - one facility variable
# - process-control and batch-quality variables
# - potency_score as the main outcome
#
# Potency is generated from process conditions and intermediate quality signals:
# transduction efficiency and viability help potency, while process deviations
# and shear stress can reduce it. Facility effects introduce site-level structure.

set.seed(1005)

out_dir <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1] else "data"; dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

n <- 420

batch_id <- sprintf("BATCH_%04d", 1:n)
facility <- sample(
  c("FAC_A", "FAC_B", "FAC_C"),
  size = n, replace = TRUE, prob = c(0.34, 0.33, 0.33)
)

facility_shift <- c(FAC_A = 0.0, FAC_B = 0.2, FAC_C = -0.2)

culture_days <- sample(
  10:14, size = n, replace = TRUE,
  prob = c(0.12, 0.22, 0.30, 0.22, 0.14)
)

temp_dev <- round(rnorm(n, mean = unname(facility_shift[facility]) * 0.4, sd = 0.7), 2)

ph_dev <- round(rnorm(n, mean = unname(facility_shift[facility]) * 0.05, sd = 0.12), 3)

transduction_eff <- plogis(
  1.2 - 0.45 * abs(temp_dev) - 1.0 * abs(ph_dev) + rnorm(n, 0, 0.7)
)
transduction_eff <- round(transduction_eff, 3)

shear_stress <- round(
  pmax(rnorm(n, mean = 6.5, sd = 1.8) + 0.35 * (culture_days - 12), 0.5),
  2
)

viability <-
  88 +
  8 * transduction_eff -
  3.2 * abs(temp_dev) -
  18 * abs(ph_dev) -
  0.9 * shear_stress +
  rnorm(n, 0, 3.5)

viability <- round(pmin(pmax(viability, 45), 99), 1)

yield_millions <-
  55 +
  5.5 * culture_days +
  18 * transduction_eff -
  1.1 * shear_stress +
  rnorm(n, 0, 8)

yield_millions <- round(pmax(yield_millions, 5), 1)

# Outcome logic:
# final potency depends on process quality and batch quality.
# Some facility effect remains even after measured variables.
facility_potency <- c(FAC_A = 0.0, FAC_B = 2.0, FAC_C = -1.5)

potency_score <-
  35 +
  8.5 * transduction_eff +
  0.35 * viability +
  0.08 * yield_millions -
  2.4 * abs(temp_dev) -
  14 * abs(ph_dev) -
  0.9 * shear_stress +
  0.6 * (culture_days - 12) +
  unname(facility_potency[facility]) +
  rnorm(n, 0, 4.5)

potency_score <- round(potency_score, 1)

ds05_cell_FULL <- data.frame(
  batch_id = batch_id,
  facility = facility,
  culture_days = culture_days,
  temp_dev = temp_dev,
  ph_dev = ph_dev,
  transduction_eff = transduction_eff,
  shear_stress = shear_stress,
  viability = viability,
  yield_millions = yield_millions,
  potency_score = potency_score,
  check.names = FALSE
)

write.csv(
  ds05_cell_FULL,
  file = file.path(out_dir, "cell.csv"),
  row.names = FALSE
)

# --- VERIFICATION BLOCK ---

cat("Dataset 5 dimensions:", dim(ds05_cell_FULL), "\n")

expected_cols <- c(
  "batch_id", "facility", "culture_days", "temp_dev", "ph_dev",
  "transduction_eff", "shear_stress", "viability",
  "yield_millions", "potency_score"
)

missing_cols <- setdiff(expected_cols, colnames(ds05_cell_FULL))
cat("Missing columns:", length(missing_cols), "\n")

all_present <- all(expected_cols %in% colnames(ds05_cell_FULL))
cat("All required features present:", all_present, "\n")

str(ds05_cell_FULL)