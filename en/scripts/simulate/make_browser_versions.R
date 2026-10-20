# Derive small, browser-friendly versions of the running-example datasets.
# Usage: Rscript scripts/simulate/make_browser_versions.R <dir_with_full_outputs>
args <- commandArgs(TRUE); src <- if (length(args)) args[1] else "data"
set.seed(2026)
onco <- readRDS(file.path(src, "onco1_full.rds"))
keep_genes <- sprintf("gene_%02d", 1:50)          # includes causal 01/05/12 and leaky 08/14/19
clin <- c("patient_id", "treatment", "dose_intensity", "patient_age", "tumor_grade",
          "performance_score", "baseline_tumor_mm", "post_tumor_mm",
          "response_percent", "high_response")
idx <- sort(sample(nrow(onco), 1000))
onco_small <- onco[idx, c(clin, keep_genes)]
onco_small[keep_genes] <- lapply(onco_small[keep_genes], round, 3)
write.csv(onco_small, "data/onco1.csv", row.names = FALSE)

hb <- read.csv(file.path(src, "homebeat.csv"))
hb <- cbind(participant_id = sprintf("HB%03d", seq_len(nrow(hb))), hb)
hb[] <- lapply(hb, function(x) if (is.numeric(x)) round(x, 3) else x)
write.csv(hb, "data/homebeat.csv", row.names = FALSE)
hbm <- read.csv(file.path(src, "homebeat_missing.csv"))
hbm <- cbind(participant_id = sprintf("HB%03d", seq_len(nrow(hbm))), hbm)
hbm[] <- lapply(hbm, function(x) if (is.numeric(x)) round(x, 3) else x)
write.csv(hbm, "data/homebeat_missing.csv", row.names = FALSE)
cat("done\n")
