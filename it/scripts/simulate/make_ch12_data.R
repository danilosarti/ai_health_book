# =====================================================================
# Chapter 12: datasets used to teach interpretability
# Usage (from the book folder): Rscript scripts/simulate/make_ch12_data.R
#
# 1. med_small_sim: a small simulated oncology study (800 patients, standard
#    therapy vs a new therapy, a biomarker that modifies the treatment effect,
#    120 genes). This is the simulator of the first edition, unchanged apart
#    from file paths. It writes the full data set (RDS) and a browser version
#    (CSV) with the clinical columns and the first 20 genes (genes 1-12 carry
#    signal, the rest are noise).
# 2. lung_sim_complete: a small simulated lung-cancer follow-up data set
#    (228 patients) inspired by the `lung` data of the R package survival.
#    The file is taken unchanged from the first edition; here we only add a
#    fixed 75/25 train/test split, data/splits/lung_split.csv (seed 2026).
# =====================================================================

dir.create("data/splits", recursive = TRUE, showWarnings = FALSE)

# ---- 1. med_small_sim ---------------------------------------------------
simulate_med_small <- function(n = 800, seed = 2026, sigma = 8, high_thr = 30,
                               n_genes = 120, n_gene_signal = 12) {
  set.seed(seed)
  age         <- pmin(pmax(round(rnorm(n, 62, 10)), 18), 90)
  sex         <- rbinom(n, 1, 0.48)                                   # 1 = male
  bmi         <- pmin(pmax(rnorm(n, 27, 4), 18), 45)
  stage       <- sample(1:3, n, replace = TRUE, prob = c(0.35, 0.40, 0.25))
  baseline_mm <- pmin(pmax(rlnorm(n, log(40), 0.35), 10), 120)
  crp         <- pmin(pmax(rlnorm(n, log(3), 0.6), 0.1), 60)
  egfr        <- pmin(pmax(rnorm(n, 85, 18), 15), 140)
  treatment   <- rbinom(n, 1, 0.5)                                    # 1 = new therapy (randomised)
  adherence   <- pmin(pmax(rbeta(n, 7, 2), 0.05), 1)
  biomarker   <- rnorm(n, 0, 1)

  egfr_center <- (egfr - 90) / 20
  eta <- 0.8 * treatment + 1.1 * treatment * biomarker + 0.5 * adherence -
    0.25 * (age - 60) / 10 - 0.35 * sex - 0.30 * (bmi - 27) / 5 - 0.60 * (stage - 1) -
    0.55 * log1p(crp) - 0.22 * sqrt(baseline_mm) + 0.35 * egfr_center - 0.15 * egfr_center^2
  shrink_mean <- 100 * plogis(eta - 0.5)
  tumor_shrinkage_pct <- pmin(pmax(shrink_mean + rnorm(n, 0, sigma), 0), 100)
  high_response <- as.integer(tumor_shrinkage_pct >= high_thr)

  # gene block: correlated noise; genes 1-12 also track the biomarker
  g_factor <- rnorm(n, 0, 1)
  G <- matrix(rnorm(n * n_genes, 0, 1), nrow = n, ncol = n_genes)
  G <- 0.70 * G + 0.30 * matrix(g_factor, nrow = n, ncol = n_genes)
  for (j in seq_len(n_gene_signal)) {
    G[, j] <- G[, j] + 0.55 * biomarker + 0.45 * (treatment * biomarker) +
      0.15 * (stage - 1) + rnorm(n, 0, 0.15)
  }
  colnames(G) <- sprintf("gene_%03d", seq_len(n_genes))

  df <- data.frame(
    patient_id = sprintf("P%04d", seq_len(n)),
    tumor_shrinkage_pct = tumor_shrinkage_pct, high_response = high_response,
    age = age, sex = factor(sex, levels = c(0, 1), labels = c("female", "male")),
    bmi = bmi, stage = factor(stage, levels = 1:3, labels = c("I", "II", "III")),
    baseline_mm = baseline_mm, crp = crp, egfr = egfr,
    treatment = factor(treatment, levels = c(0, 1), labels = c("standard", "new_therapy")),
    adherence = adherence, biomarker = biomarker)
  cbind(df, as.data.frame(G))
}

med <- simulate_med_small()
saveRDS(med, "data/med_small_sim.rds")
keep <- c(names(med)[1:13], sprintf("gene_%03d", 1:20))
med_csv <- med[, keep]
num <- vapply(med_csv, is.numeric, logical(1))
med_csv[num] <- lapply(med_csv[num], function(v) round(v, 3))
write.csv(med_csv, "data/med_small_sim.csv", row.names = FALSE)

# ---- 2. lung_sim_complete: shared split ---------------------------------
lung <- read.csv("data/lung_sim_complete.csv")
set.seed(2026)
test_id <- sample(lung$patient_id, size = round(0.25 * nrow(lung)))
write.csv(data.frame(patient_id = lung$patient_id,
                     set = ifelse(lung$patient_id %in% test_id, "test", "train")),
          "data/splits/lung_split.csv", row.names = FALSE)
cat("med_small_sim:", dim(med), " | lung split:", table(ifelse(lung$patient_id %in% test_id, "test", "train")), "\n")
