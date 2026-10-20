# =====================================================================
# Dermato-ATT: simulated multi-omics psoriasis cohort (Chapter 10)
# Source: v1 book, high_dims.qmd (author: D. A. Sarti). The simulation code
# below is the original v1 code, unchanged (seed 1234 reproduces
# Dermato_ATT_simulated.rds exactly). Only the output step is new: the full
# table (200 x 1 852) is too large for the browser, so we also write a reduced
# version with every clinical, biomarker and exposome column, the first 100
# genes and the first 50 microbiome features (ASVs).
#
# What the simulation really contains (used as a lesson in Chapter 10):
#   * genes      = latent_G1 * a_j + latent_G2 * b_j + noise, with a_j and b_j
#                  drawn from the SAME distribution N(0.7, 0.2): the two "gene
#                  signatures" are not distinguishable by design, and no IL-17
#                  (or any other named pathway) is simulated;
#   * ASVs       = latent_M1 * c_j + latent_M2 * d_j + noise, c_j, d_j ~ N(0.5, 0.1);
#   * cytokines  = 2 * latent_G1 + noise (18 columns);
#   * exposome   = independent noise, on very different scales (NO2 sd 8,
#                  humidity sd 12, the expo_ columns sd 1);
#   * outcome    = 20 - 2 latent_G1 + 1.5 latent_M1 - 0.5 NO2 + noise: NO2,
#                  a single column, carries the largest share of the signal.
#
# Usage: Rscript scripts/simulate/sim_dermato_att.R [output_dir] [full_rds_path]
#   (run from the book root; writes data/dermato.csv by default)
# =====================================================================

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "data"

# ------------------------------------------------------------
# Simulated High-Dimensional Dermatology Dataset
# Psoriasis patients treated with anti–IL-17 biologics
# ------------------------------------------------------------

set.seed(1234)

# Number of patients
n <- 200

# ------------------------------------------------------------
# 1. Genomic block (1,500 genes)
# Two latent genomic signatures:
#  - G1: inflammatory axis (IL17/IL23 pathway)
#  - G2: barrier/keratinocyte differentiation
# ------------------------------------------------------------

latent_G1 <- rnorm(n, 0, 1)
latent_G2 <- rnorm(n, 0, 1)

# gene loadings
G1_load <- matrix(rnorm(1500, 0.7, 0.2), nrow = n, ncol = 1500, byrow = TRUE)
G2_load <- matrix(rnorm(1500, 0.7, 0.2), nrow = n, ncol = 1500, byrow = TRUE)

Gene_block <- latent_G1 * G1_load + latent_G2 * G2_load +
  matrix(rnorm(n*1500, 0, 0.5), n, 1500)

colnames(Gene_block) <- paste0("gene_", 1:1500)

# ------------------------------------------------------------
# 2. Microbiome (300 ASVs collapsed)
# Two microbial communities:
#   M1 = dysbiotic signature
#   M2 = commensal-dominant signature
# ------------------------------------------------------------

latent_M1 <- rnorm(n, 0, 1)
latent_M2 <- rnorm(n, 0, 1)

ASV_load1 <- matrix(rnorm(300, 0.5, 0.1), nrow = n, ncol = 300, byrow = TRUE)
ASV_load2 <- matrix(rnorm(300, 0.5, 0.1), nrow = n, ncol = 300, byrow = TRUE)

Microbiome_block <- latent_M1 * ASV_load1 + latent_M2 * ASV_load2 +
  matrix(rnorm(n*300, 0, 0.5), n, 300)

colnames(Microbiome_block) <- paste0("ASV_", 1:300)

# ------------------------------------------------------------
# 3. Exposome (30 variables: pollution, UV, humidity, etc.)
# ------------------------------------------------------------

Exposome_block <- data.frame(
  NO2 = rnorm(n, 25, 8),
  PM10 = rnorm(n, 18, 6),
  UV_index = rnorm(n, 4, 1),
  temperature = rnorm(n, 15, 5),
  humidity = rnorm(n, 60, 12)
)

# Fill remaining variables
for(i in 6:30){
  Exposome_block[[paste0("expo_", i)]] <- rnorm(n, 0, 1)
}

# ------------------------------------------------------------
# 4. Biomarkers (20 vars: PASI baseline, DLQI, cytokines)
# ------------------------------------------------------------

Biomarkers_block <- data.frame(
  PASI_baseline = rnorm(n, 12, 3),
  DLQI = rnorm(n, 8, 4)
)

# Cytokines (IL17, IL23, TNF, IL-6 etc.)
for(i in 1:18){
  Biomarkers_block[[paste0("cytokine_", i)]] <-
    2*latent_G1 + rnorm(n, 0, 1)
}

# ------------------------------------------------------------
# 5. Outcome – PASI reduction after 12 weeks
# Influenced by:
#  - genomic inflammation (G1)
#  - microbiome dysbiosis (M1)
#  - exposome (pollution worsens response)
#  - random noise
# ------------------------------------------------------------

Outcome <-  -2*latent_G1 +
            1.5*latent_M1 -
            0.5*Exposome_block$NO2 +
            rnorm(n, 0, 1)

# Scale to "ΔPASI" reasonable range
Outcome <- 20 + Outcome + rnorm(n, 0, 3)

# ------------------------------------------------------------
# Final dataset assembly
# ------------------------------------------------------------

Dermato_ATT <- data.frame(
  patient_id = paste0("P", 1:n),
  Outcome_PASI12 = Outcome,
  Biomarkers_block,
  Exposome_block,
  Gene_block,
  Microbiome_block
)

# ------------------------------------------------------------
# Output (new): full table as .rds (for the precompute scripts) and a reduced,
# browser-sized CSV. The latent factors are kept in a separate file so that the
# chapter can reveal the truth behind the principal components.
# ------------------------------------------------------------
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
keep <- c("patient_id", "Outcome_PASI12", names(Biomarkers_block), names(Exposome_block),
          paste0("gene_", 1:100), paste0("ASV_", 1:50))
red <- Dermato_ATT[, keep]
num <- vapply(red, is.numeric, logical(1))
red[num] <- lapply(red[num], round, 3)
write.csv(red, file.path(out_dir, "dermato.csv"), row.names = FALSE)

latent <- data.frame(patient_id = Dermato_ATT$patient_id,
                     latent_G1 = round(latent_G1, 4), latent_G2 = round(latent_G2, 4),
                     latent_M1 = round(latent_M1, 4), latent_M2 = round(latent_M2, 4))
write.csv(latent, file.path(out_dir, "dermato_latent.csv"), row.names = FALSE)

# The full table is not shipped with the book; the precompute script rebuilds it
# by passing a second argument (path of the full .rds to write).
if (length(args) >= 2) saveRDS(Dermato_ATT, args[2])

# Check against the v1 file when it is available.
v1 <- Sys.getenv("DERMATO_V1_RDS", "Dermato_ATT_simulated.rds")
if (file.exists(v1)) {
  ref <- readRDS(v1)
  cat("Identical to v1 rds:", isTRUE(all.equal(ref, Dermato_ATT, check.attributes = FALSE)), "\n")
}
cat("Wrote", file.path(out_dir, "dermato.csv"), "with", nrow(red), "rows and", ncol(red), "columns\n")
