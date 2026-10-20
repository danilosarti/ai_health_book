# =====================================================================
# Digital Trial "manifold pedagogy" dataset (Chapter 10)
# Source: v1 book, high_dims.qmd (author: D. A. Sarti). The simulation code
# below is the original v1 code, unchanged (seed 2027 reproduces
# DigitalTrial_ManifoldPedagogy.rds exactly). New: the hidden geometry that the
# v1 code generated and then dropped (which shape each participant was drawn
# from, the position along it, and the three coordinates) is written to a
# separate "truth" file, so that Chapter 10 can reveal it after the reader has
# looked at the t-SNE and UMAP maps.
#
# The truth, in short:
#   * 350 participants lie on a curve that the code calls an "S-curve";
#     with x = sin t, y = t, z = cos t it is in fact a helix (a spring of one and
#     a half turns), a one-dimensional object;
#   * 150 participants lie on a Swiss roll (a rolled-up sheet, two-dimensional);
#   * every feature is a smooth or oscillating function of the three
#     coordinates plus noise; clinical_1 is 3 on the helix and 5 on the roll
#     (sd 0.3), so this single column separates the two shapes;
#   * wearable_4 = sin(10 (X1 + X2)) oscillates so fast that it behaves almost
#     like noise.
#
# Usage: Rscript scripts/simulate/sim_digital_trial.R [output_dir]
# =====================================================================

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "data"

suppressPackageStartupMessages(library(dplyr))

set.seed(2027)

# ------------------------------------------------------------
# 1. Generate two expressive manifolds
#    - 1) S-curve (main structure)
#    - 2) Swiss roll (secondary structure)
# ------------------------------------------------------------

n1 <- 350   # S-curve points
n2 <- 150   # Swiss-roll points
n  <- n1 + n2

# ---- S-curve ----
t1 <- runif(n1, -3*pi/2, 3*pi/2)
x1 <- sin(t1)
y1 <- t1
z1 <- cos(t1)

Scurve <- data.frame(
  manifold = "S",
  X1 = x1,
  X2 = y1 + rnorm(n1, 0, 0.15),
  X3 = z1
)

# ---- Swiss roll ----
t2 <- runif(n2, 1, 4*pi)
h2 <- runif(n2, -2, 2)
x2 <- t2 * cos(t2)
y2 <- h2
z2 <- t2 * sin(t2)

Swiss <- data.frame(
  manifold = "R",
  X1 = x2 / max(x2),
  X2 = y2,
  X3 = z2 / max(z2)
)

df_raw <- rbind(Scurve, Swiss)

# ------------------------------------------------------------
# 2. Create cytokine signals (smooth gradients)
# ------------------------------------------------------------
df_raw <- df_raw %>%
  mutate(
    cytokine_1 = 2 + 1.5 * X1 + rnorm(n, 0, 0.1),
    cytokine_2 = 1 + 1.2 * X2 + rnorm(n, 0, 0.1),
    cytokine_3 = 0.5 * (X1 + X2) + rnorm(n, 0, 0.1),
    cytokine_4 = sin(X2) + rnorm(n, 0, 0.1)
  )

# ------------------------------------------------------------
# 3. Wearable features (oscillatory, high-frequency)
# ------------------------------------------------------------
df_raw <- df_raw %>%
  mutate(
    wearable_1 = sin(4*X1) + rnorm(n, 0, 0.2),
    wearable_2 = cos(5*X2) + rnorm(n, 0, 0.2),
    wearable_3 = sin(6*X3) + cos(3*X1) + rnorm(n, 0, 0.2),
    wearable_4 = sin(10*(X1+X2)) + rnorm(n, 0, 0.2)
  )

# ------------------------------------------------------------
# 4. Clinical features (cluster-like behavior)
# ------------------------------------------------------------
df_raw <- df_raw %>%
  mutate(
    clinical_1 = ifelse(manifold == "S",
                        3 + rnorm(n, 0, 0.3),
                        5 + rnorm(n, 0, 0.3)),
    clinical_2 = ifelse(X2 > 0,
                        2.5 + rnorm(n,0,0.3),
                        1.2 + rnorm(n,0,0.3)),
    clinical_3 = ifelse(X1 > 0,
                        3 + 0.5*X1 + rnorm(n, 0, 0.2),
                        2 + 0.5*X1 + rnorm(n, 0, 0.2))
  )

# ------------------------------------------------------------
# 5. Outcome (ΔScore): nonlinear combination of features
# ------------------------------------------------------------
df_raw <- df_raw %>%
  mutate(
    delta_score =
      0.4*cytokine_1 +
      0.3*wearable_3 +
      0.5*clinical_2 +
      0.25*sin(X1 + X2) +
      rnorm(n, 0, 0.3)
  )

# ------------------------------------------------------------
# 6. Responder category (clear split)
# ------------------------------------------------------------
df_raw <- df_raw %>%
  mutate(
    responder = ifelse(delta_score > median(delta_score),
                       "High response", "Low response")
  )

# ------------------------------------------------------------
# 7. Add patient_id and reorder columns
# ------------------------------------------------------------
df_final <- df_raw %>%
  dplyr::mutate(patient_id = 1:n) %>%
  dplyr::select(patient_id, everything()) %>%
  dplyr::select(-X1, -X2, -X3, -manifold)

# ------------------------------------------------------------
# 8. Output (new): browser CSV + truth file
# ------------------------------------------------------------
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out <- df_final
num <- vapply(out, is.numeric, logical(1)) & names(out) != "patient_id"
out[num] <- lapply(out[num], round, 4)
write.csv(out, file.path(out_dir, "digital_trial.csv"), row.names = FALSE)

truth <- data.frame(
  patient_id = 1:n,
  shape      = ifelse(df_raw$manifold == "S", "helix", "swiss_roll"),
  position   = round(c(t1, t2), 4),       # parameter t along the helix / the roll
  X1 = round(df_raw$X1, 4), X2 = round(df_raw$X2, 4), X3 = round(df_raw$X3, 4)
)
write.csv(truth, file.path(out_dir, "digital_trial_truth.csv"), row.names = FALSE)

v1 <- Sys.getenv("DIGITAL_V1_RDS", "DigitalTrial_ManifoldPedagogy.rds")
if (file.exists(v1)) {
  ref <- readRDS(v1)
  cat("Identical to v1 rds:", isTRUE(all.equal(as.data.frame(ref), as.data.frame(df_final),
                                               check.attributes = FALSE)), "\n")
}
cat("Wrote digital_trial.csv and digital_trial_truth.csv to", out_dir, "\n")
