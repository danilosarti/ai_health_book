# =====================================================================
# Ward 7 ICU with missing values (Running example 1, Chapter 9)
#
# Derived from data/icu_aki.csv (made by sim_ward7_icu_aki.R) by deleting
# values on purpose ("amputation") with three documented mechanisms.
# The outcome and all other variables stay complete; rows and order are
# unchanged, so patient_id still matches icu_aki.csv and the shared split
# data/splits/ward7_split.csv.
#
#   albumin         MCAR  a batch of albumin results was lost when the
#                         laboratory analyser failed: every patient has the
#                         same 8 % chance of a missing value.
#   lactate         MAR   lactate is ordered less often for patients who look
#                         well: the probability of a missing value falls as
#                         SOFA rises (it depends only on an observed variable).
#   baseline_creat  MNAR  a baseline creatinine exists only if the patient had
#                         a blood test before admission. Younger and healthier
#                         people (lower creatinine) rarely have one, so the
#                         probability of a missing value depends on age
#                         (observed) AND on the unseen creatinine value itself.
#
# The older att_ai_ml "_MISSING" ICU file came from a different simulation
# and does not match icu_aki.csv; this script replaces it.
# Usage: Rscript scripts/simulate/sim_ward7_missing.R   (from the book root)
# =====================================================================

set.seed(909)

icu <- read.csv("data/icu_aki.csv", check.names = FALSE)
n   <- nrow(icu)
out <- icu

# 1. albumin: MCAR, 8 % --------------------------------------------------
p_alb <- rep(0.08, n)
out$albumin[runif(n) < p_alb] <- NA

# 2. lactate: MAR on SOFA -------------------------------------------------
# about 45 % missing at SOFA 2, about 5 % at SOFA 12
p_lac <- plogis(-1.55 - 0.40 * (icu$sofa - 7))
out$lactate[runif(n) < p_lac] <- NA

# 3. baseline creatinine: MNAR (own value) + age -------------------------
p_bc <- plogis(-1.70 - 0.03 * (icu$age - 64) - 3.0 * (icu$baseline_creat - 1.0))
out$baseline_creat[runif(n) < p_bc] <- NA

write.csv(out, "data/icu_aki_missing.csv", row.names = FALSE, na = "NA")

# --- verification ----------------------------------------------------------
stopifnot(identical(out$patient_id, icu$patient_id))
cat("Missing proportion per variable:\n")
print(round(colMeans(is.na(out)), 3))
cat("Rows with at least one missing value:", sum(!complete.cases(out)), "of", n, "\n")
cat("True vs observed mean baseline creatinine:",
    round(mean(icu$baseline_creat), 3), round(mean(out$baseline_creat, na.rm = TRUE), 3), "\n")
