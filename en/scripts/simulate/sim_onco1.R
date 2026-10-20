# =====================================================================
# ONCO-1 chemotherapy trial — simulation (Running example 2)
# Source: v1 book, supervised_regression.qmd (author: D. A. Sarti), unchanged
# except for the output path. Seed 2025 reproduces the v1 dataset.
# NOTE (kept on purpose, used as a lesson): genes 08, 14 and 19 are
# overwritten with noisy copies of the outcome ("response-linked"); the
# truly causal genes are 01, 05 and 12.
# Usage: Rscript sim_onco1.R [output_dir]
# =====================================================================
#| include: false
#| echo: false
#| message: false
#| warning: false
# Clinical trial (tumor patients only): chemo vs no_chemo (1:1).
# Gene expression is continuous (log2-CPM-like), not ternary.
# Outcomes:
#   - response_percent = 100 * (baseline - post) / baseline   (continuous shrinkage)
#   - high_response    = 1 if response_percent >= 30          (RECIST-like)
# Chemo must have larger mean shrinkage than no_chemo.
# 
set.seed(2025)

# ---- SETTINGS ----
n <- 10000                     # total sample size (all are tumor patients in this trial)
p <- 2000                      # number of genes
gene_names <- paste0("gene_", sprintf("%02d", 1:p))

# Randomization 1:1
treatment <- factor(sample(c("chemo","no_chemo"), n, replace = TRUE),
                    levels = c("no_chemo","chemo"))

# Clinical covariates
patient_age       <- round(runif(n, 22, 82))
tumor_grade       <- factor(sample(c("G1","G2","G3"), n, TRUE, c(0.25,0.45,0.30)))
performance_score <- sample(0:2, n, TRUE, prob = c(0.55,0.35,0.10))  # ECOG-like (0 best)

# Dose intensity (only meaningful for chemo; control = 0)
dose_intensity <- ifelse(treatment == "chemo", runif(n, 0.8, 1.1), 0)

# ---- CONTINUOUS GENE EXPRESSION (simulate counts -> log2 CPM) ----
# We simulate library sizes, gene base expression, and modest correlation structure.

# Library sizes (per sample), e.g., 2–6 million reads
lib_size <- round(runif(n, 2e6, 6e6))

# Gene base expression levels (on log scale for rates)
# Some genes are higher expressed than others
gene_base_mu <- rnorm(p, mean = -6, sd = 1)  # base log-rate offsets

# Low-rank correlation structure across genes via two latent factors
f1 <- rnorm(n, 0, 1)
f2 <- rnorm(n, 0, 1)
gene_loading1 <- rnorm(p, 0.5, 0.3)
gene_loading2 <- rnorm(p, -0.4, 0.3)

# Build raw count matrix via Poisson with sample-specific library sizes
counts <- matrix(0L, n, p, dimnames = list(NULL, gene_names))
for (j in 1:p) {
  # per-sample log-rate for gene j
  log_rate <- log(lib_size) + gene_base_mu[j] + gene_loading1[j]*f1 + gene_loading2[j]*f2 + rnorm(n, 0, 0.2)
  rate <- pmax(exp(log_rate), 1e-6)
  counts[, j] <- rpois(n, lambda = rate)
}

# CPM and log2-CPM transform (simple, library-size normalized)
cpm <- counts / rowSums(counts) * 1e6
expr_log2 <- log2(cpm + 1)
colnames(expr_log2) <- gene_names

# ---- Define causal and response-linked genes on continuous expression ----
idx_true   <- c(1, 5, 12)   # truly causal genes (affect shrinkage)
idx_linked <- c(8, 14, 19)  # response-linked (non-causal but correlated)

# Standardize gene expression for effect calculation
expr_std <- scale(expr_log2)

# Causal gene effects on shrinkage proportion (small-to-moderate)
beta_genes <- rep(0, p)
beta_genes[idx_true] <- c(0.06, -0.10, 0.05)   # signs: +, -, +
gene_signal <- drop(expr_std %*% beta_genes)

# ---- Baseline tumor size (mm) ----
baseline_tumor_mm <- rlnorm(n, meanlog = log(45), sdlog = 0.35)

# ---- Chemo efficacy (Emax-like) and toxicity ----
Emax <- 0.70                              # max shrinkage at high intensity
EC50 <- 0.85                              # intensity that gives 50% of Emax
efficacy_raw <- Emax * (dose_intensity / (EC50 + pmax(dose_intensity, 1e-6)))
efficacy_raw[treatment == "no_chemo"] <- 0

# Small clinical effects
age_eff   <- 0.001 * (patient_age - 55)                      # tiny
grade_eff <- c(G1 = 0.00, G2 = 0.02, G3 = 0.05)[tumor_grade] # slightly higher in G3
perf_eff  <- c(`0` = 0.00, `1` = -0.02, `2` = -0.05)[as.character(performance_score)]

# Toxicity penalty (reduces net benefit slightly at higher intensity)
toxicity <- 0.06 * (pmax(dose_intensity - 0.95, 0))^2

# ---- Shrinkage proportion (0..1), then sizes and percent ----
shrink_prop <- pmin(pmax(efficacy_raw + gene_signal + age_eff + grade_eff + perf_eff - toxicity +
                           rnorm(n, 0, 0.03), 0), 1)

post_tumor_mm    <- baseline_tumor_mm * (1 - shrink_prop)
response_percent <- pmin(pmax(100 * (baseline_tumor_mm - post_tumor_mm) / baseline_tumor_mm, 0), 100)
high_response    <- as.integer(response_percent >= 30)  # RECIST-like threshold

# ---- Response-linked (non-causal) genes: align a few with shrinkage ----
# We add a small correlation by replacing those columns with noisy ranks of shrink_prop
linked_signal <- as.numeric(scale(shrink_prop))
linked_signal[is.na(linked_signal)] <- 0
for (j in idx_linked) {
  expr_log2[, j] <- scale(linked_signal + rnorm(n, 0, 0.2))[,1]  # overwrite with linked pattern
}

# ---- Final dataset ----
trial_ct <- data.frame(
  patient_id        = sprintf("P%03d", 1:n),
  treatment         = treatment,
  dose_intensity    = round(dose_intensity, 3),
  patient_age       = patient_age,
  tumor_grade       = tumor_grade,
  performance_score = performance_score,
  baseline_tumor_mm = round(baseline_tumor_mm, 1),
  post_tumor_mm     = round(post_tumor_mm, 1),
  response_percent  = round(response_percent, 1),
  high_response     = high_response
)
trial_ct <- cbind(trial_ct, as.data.frame(expr_log2))  # continuous gene expression

# ---- Safety net: guarantee chemo minus no_chemo >= margin ----
margin <- 18  # desired minimal difference (pp) in mean response_percent
m <- with(trial_ct, tapply(response_percent, treatment, mean))
if (all(c("chemo","no_chemo") %in% names(m))) {
  need <- (m["no_chemo"] + margin) - m["chemo"]
  if (!is.na(need) && need > 0) {
    trial_ct$response_percent <- ifelse(
      trial_ct$treatment == "chemo",
      pmin(pmax(trial_ct$response_percent + as.numeric(need), 0), 100),
      trial_ct$response_percent
    )
    trial_ct$post_tumor_mm <- ifelse(
      trial_ct$treatment == "chemo",
      round(trial_ct$baseline_tumor_mm * (1 - trial_ct$response_percent/100), 1),
      trial_ct$post_tumor_mm
    )
    trial_ct$high_response <- as.integer(trial_ct$response_percent >= 30)
  }
}

# ---- SAVE ----
out_dir <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1] else "data"
dir.create(out_dir, showWarnings = FALSE)
saveRDS(trial_ct, file.path(out_dir, "onco1_full.rds"))


