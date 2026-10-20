# =====================================================================
# Chapter 10: PCA of the full Dermato-ATT cohort (200 patients x 1 850
# variables) with FactoMineR, plus the distance and scaling summaries quoted
# in the text. The browser copy of the data (data/dermato.csv) keeps only 200
# of the 1 850 variables, so the full-data results are precomputed here.
# Usage (from the book root): Rscript scripts/precompute/ch10_pca.R
# Needs: FactoMineR, jsonlite. Runs in a few seconds.
# =====================================================================
suppressPackageStartupMessages({ library(FactoMineR); library(jsonlite) })
out <- "data/precomputed/ch10"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

# Rebuild the full table with the original simulator (identical to the v1 .rds)
tmp <- tempfile(); full <- tempfile(fileext = ".rds")
system2("Rscript", c("scripts/simulate/sim_dermato_att.R", tmp, full), stdout = FALSE)
df  <- readRDS(full)
lat <- read.csv(file.path(tmp, "dermato_latent.csv"))
y   <- df$Outcome_PASI12
X   <- df[, setdiff(names(df), c("patient_id", "Outcome_PASI12"))]

block_of <- function(v) ifelse(grepl("^gene_", v), "genes",
                        ifelse(grepl("^ASV_", v), "microbiome",
                        ifelse(grepl("^cytokine_", v), "cytokines",
                        ifelse(v %in% c("PASI_baseline", "DLQI"), "clinical", "exposome"))))

# ---- PCA on standardised variables (FactoMineR default scale.unit = TRUE) ----
res <- PCA(X, scale.unit = TRUE, ncp = 10, graph = FALSE)
eig <- data.frame(component = 1:20, eigenvalue = res$eig[1:20, 1],
                  pve = res$eig[1:20, 2], cumulative = res$eig[1:20, 3])
write.csv(round(eig, 4), file.path(out, "dermato_eig.csv"), row.names = FALSE)

ct <- res$var$contrib; co <- res$var$coord; c2 <- res$var$cos2
blk <- block_of(rownames(ct))
blocks <- do.call(rbind, lapply(split(seq_along(blk), blk), function(ix)
  data.frame(block = blk[ix[1]], n_vars = length(ix),
             t(round(colSums(ct[ix, 1:5, drop = FALSE]), 2)))))
names(blocks) <- c("block", "n_vars", paste0("PC", 1:5))
write.csv(blocks, file.path(out, "dermato_blocks.csv"), row.names = FALSE)

top <- do.call(rbind, lapply(1:4, function(k) {
  o <- order(-ct[, k])[1:10]
  data.frame(pc = k, variable = rownames(ct)[o], block = blk[o],
             coord = round(co[o, k], 3), cos2 = round(c2[o, k], 3), contrib = round(ct[o, k], 4))
}))
write.csv(top, file.path(out, "dermato_topvars.csv"), row.names = FALSE)

coords <- data.frame(variable = rownames(co), block = blk,
                     Dim1 = round(co[, 1], 3), Dim2 = round(co[, 2], 3), Dim3 = round(co[, 3], 3),
                     contrib1 = round(ct[, 1], 4), contrib2 = round(ct[, 2], 4), contrib3 = round(ct[, 3], 4))
write.csv(coords, file.path(out, "dermato_var_coords.csv"), row.names = FALSE)

S <- res$ind$coord
scores <- data.frame(patient_id = df$patient_id, round(S[, 1:5], 3), outcome = round(y, 3),
                     NO2 = round(df$NO2, 3), lat[, -1])
names(scores)[2:6] <- paste0("PC", 1:5)
write.csv(scores, file.path(out, "dermato_scores.csv"), row.names = FALSE)

cors <- data.frame(component = 1:10, round(cor(S[, 1:10], cbind(lat[, -1], outcome = y, NO2 = df$NO2)), 3),
                   NO2_loading = round(co["NO2", 1:10], 3))
write.csv(cors, file.path(out, "dermato_pc_cor.csv"), row.names = FALSE)

r2 <- function(...) summary(lm(y ~ ...))$r.squared

# ---- Scaling: the exposome block on its own --------------------------------
E  <- X[, block_of(names(X)) == "exposome"]
pu <- prcomp(E, scale. = FALSE); ps <- prcomp(E, scale. = TRUE)
pve_u <- pu$sdev^2 / sum(pu$sdev^2); pve_s <- ps$sdev^2 / sum(ps$sdev^2)
pu_all <- prcomp(X, scale. = FALSE); pve_all_u <- pu_all$sdev^2 / sum(pu_all$sdev^2)

# ---- Distances: nearest / farthest pair --------------------------------------
ratio <- function(M) { d <- dist(M); min(d) / max(d) }
set.seed(10)
iid <- matrix(rnorm(nrow(X) * ncol(X)), nrow(X))

summ <- list(
  n = nrow(X), p = ncol(X),
  mean_contrib = 100 / ncol(X),
  pc1_genes_share = blocks$PC1[blocks$block == "genes"],
  pc2_micro_share = blocks$PC2[blocks$block == "microbiome"],
  pc3_cyto_share  = blocks$PC3[blocks$block == "cytokines"],
  r2_no2 = r2(df$NO2), r2_g1 = r2(lat$latent_G1), r2_m1 = r2(lat$latent_M1),
  r2_pc1to3 = r2(S[, 1:3]), r2_pc1to5 = r2(S[, 1:5]), r2_pc1to10 = r2(S[, 1:10]),
  cor_no2_y = cor(df$NO2, y),
  no2_contrib_pc1 = ct["NO2", 1],
  no2_max_pc = unname(which.max(abs(co["NO2", 1:10]))),
  no2_max_loading = unname(co["NO2", which.max(abs(co["NO2", 1:10]))]),
  expo_unscaled_pc1 = pve_u[1], expo_unscaled_top = names(which.max(abs(pu$rotation[, 1]))),
  expo_scaled_pc1 = pve_s[1],
  all_unscaled_pc1 = pve_all_u[1], all_unscaled_pc2 = pve_all_u[2],
  all_unscaled_pc2_top = names(which.max(abs(pu_all$rotation[, 2]))),
  ratio_scaled_full = ratio(scale(X)), ratio_pc5 = ratio(S[, 1:5]),
  ratio_iid_same_dims = ratio(iid)
)
write_json(summ, file.path(out, "dermato_summary.json"), auto_unbox = TRUE, digits = 6, pretty = TRUE)
cat("ch10_pca.R done\n")
