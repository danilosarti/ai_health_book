# =====================================================================
# Chapter 13: "Ward 7, extended": data for the DAG adjustment widget.
#
# Same generating rules as Ward 7 (scripts/simulate/sim_ward7_icu_aki.R,
# re-implemented in ward7_scm.R) with a different seed and three extra
# recorded variables whose causal role we know:
#   staffing      nurse staffing (standardised): affects adherence only
#   nephrotox     nephrotoxic-drug exposure in 48 h (dose-days): lowered by
#                 the protocol, raises creatinine -> a MEDIATOR
#   creat_tests   creatinine tests ordered in 48 h: raised by the protocol
#                 and by a rising creatinine -> a COLLIDER
# Total effect of adherence: -0.28 per unit (direct -0.18, via nephrotox
# about -0.10), as in Ward 7.
# Usage (from the book folder): Rscript scripts/simulate/sim_ch13_ward7_dag.R
# Writes data/ward7_dag.csv.
# =====================================================================
source("scripts/simulate/ward7_scm.R")
d <- ward7_scm(n = 1250, seed = 1313, extended = TRUE)
keep <- c("patient_id", "sofa", "baseline_creat", "staffing", "protocol_adherence",
          "nephrotox", "creat_tests", "delta_creatinine_48h")
d <- d[, keep]
d$patient_id <- sub("ICU_", "X7_", d$patient_id)     # not the same patients as icu_aki.csv
write.csv(d, "data/ward7_dag.csv", row.names = FALSE)
cat("ward7_dag.csv:", nrow(d), "rows\n")
