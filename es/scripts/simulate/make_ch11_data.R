# =====================================================================
# Chapter 11 (Clustering patients): browser versions of the v1 datasets
# Usage (from the book folder):
#   Rscript scripts/simulate/make_ch11_data.R [folder with the v1 .rds files]
# Writes data/wearable_phenotypes.csv (800 x 8) and data/genomic_subtypes.csv
# (60 x 302). Needs base R + MASS.
#
# 1. Wearable phenotypes. The v1 file wearable_cluster.rds was produced by the
#    simulation in v1 clustering.qmd (seed 2025, five phenotypes of 160 people,
#    MASS::mvrnorm) with the phenotype column removed. The simulation is re-run
#    below so that the true phenotype can be kept, for teaching only; if the
#    v1 file is available, the script checks that both are identical.
# 2. Genomic matrix. 60 simulated patients x 300 genes with three molecular
#    subtypes (v1 simulated_genomic_matrix.rds and simulated_genomic_groups.rds,
#    no generating script survives); converted to CSV and rounded to 3 decimals.
# =====================================================================
args   <- commandArgs(trailingOnly = TRUE)
src    <- if (length(args)) args[1] else "../src/data"
dir.create("data", showWarnings = FALSE)

# ---- 1. Wearable phenotypes ------------------------------------------
library(MASS)
set.seed(2025)
vars <- c("HR_mean", "HRV_sdnn", "steps_mean", "skin_temp", "spo2", "sleep_eff")
means <- list(
  c(60, 80, 12000, 32.5, 98.0, 0.95),   # C1 healthy active
  c(100, 40, 2000, 33.0, 96.5, 0.80),   # C2 high stress
  c(55, 60, 3000, 32.7, 97.5, 0.88),    # C3 sedentary calm
  c(75, 50, 6000, 34.8, 95.8, 0.82),    # C4 fever / infection
  c(110, 35, 1000, 33.5, 96.2, 0.75))   # C5 post-operative
cov_base <- matrix(c(
   20,  5,   500, 0.4, 0.1, 0.02,
    5, 60,   400, 0.2, 0.1, 0.02,
  500, 400, 50000, 1.0, 0.2, 0.03,
  0.4, 0.2,  1.0, 0.5, 0.2, 0.01,
  0.1, 0.1,  0.2, 0.2, 0.3, 0.01,
  0.02, 0.02, 0.03, 0.01, 0.01, 0.02), 6, 6)
sim <- do.call(rbind, lapply(means, function(m) mvrnorm(160, m, cov_base)))
colnames(sim) <- vars
labels <- c("healthy_active", "high_stress", "sedentary_calm", "fever_infection", "post_operative")

f_v1 <- file.path(src, "wearable_cluster.rds")
if (file.exists(f_v1)) {
  v1 <- as.matrix(readRDS(f_v1))
  stopifnot(max(abs(v1 - sim)) < 1e-8)
  message("wearable: re-simulated data identical to the v1 file")
}
wear <- data.frame(person_id = sprintf("W%03d", 1:800), round(sim, 3),
                   phenotype_true = rep(labels, each = 160))
wear$steps_mean <- round(wear$steps_mean)
write.csv(wear, "data/wearable_phenotypes.csv", row.names = FALSE)

# ---- 2. Genomic matrix -------------------------------------------------
G  <- readRDS(file.path(src, "simulated_genomic_matrix.rds"))
gr <- readRDS(file.path(src, "simulated_genomic_groups.rds"))
gen <- data.frame(patient_id = sprintf("P%02d", seq_len(nrow(G))), subtype = gr,
                  round(G, 3), check.names = FALSE)
names(gen)[-(1:2)] <- sprintf("gene_%03d", seq_len(ncol(G)))
write.csv(gen, "data/genomic_subtypes.csv", row.names = FALSE)
message("written: data/wearable_phenotypes.csv, data/genomic_subtypes.csv")
