# =====================================================================
# Helpers for chapters: read numbers from the model ledger
# (data/precomputed/ledger/, written by scripts/precompute/ledger.R).
#
#   source("scripts/ledger_helpers.R")
#   ledger("ward7_reg", "ols", "test_MAE")            # 0.14132  (model_id or exact model name)
#   ledger("ward7_aki", "logit_p5", "test_AUC", 3)    # 0.855    (optional rounding)
#   ledger_table("ward7_reg", upto_chapter = 6)       # cumulative comparison table
#   ledger_tonight("ward7_aki", "logit_p5", "Aoife")  # 0.48587  (Δ creatinine for ward7_reg)
#   ledger_patient("onco1_class", "logit_7", "#0042") # P(high response) for patient #0042
#   ledger_lena("bayes_lin")                          # Lena's predicted resting HR
#
# example: "ward7_reg", "ward7_aki", "onco1_reg", "onco1_class", "homebeat".
# Chapters are rendered from the book folder; set options(ledger.dir = ...) otherwise.
# Base R only (works in knitr and in webR).
# =====================================================================
.ledger_env <- new.env()

.ledger_read <- function(file) {
  dir <- getOption("ledger.dir", "data/precomputed/ledger")
  key <- file.path(dir, file)
  if (is.null(.ledger_env[[key]])) {
    .ledger_env[[key]] <- utils::read.csv(key, stringsAsFactors = FALSE, encoding = "UTF-8",
                                          check.names = FALSE)
  }
  .ledger_env[[key]]
}

.ledger_row <- function(tab, model, what) {
  i <- which(tab$model_id == model)
  if (!length(i)) i <- which(tab$model == model)
  if (length(i) != 1)
    stop(sprintf("ledger: model '%s' not found (or not unique) in %s; known model_id: %s",
                 model, what, paste(unique(tab$model_id), collapse = ", ")), call. = FALSE)
  i
}

# One number. metric = any column of the example's table, e.g.
#   regression:      cv_MAE, test_MAE, test_RMSE, test_R2, n_train, n_test
#   classification:  cv_logloss, test_AUC, test_Brier, cal_slope, cal_intercept, n_train, n_test
ledger <- function(example, model, metric, digits = NULL) {
  tab <- .ledger_read(paste0(example, ".csv"))
  if (!metric %in% names(tab))
    stop(sprintf("ledger: metric '%s' not in %s (columns: %s)", metric, example,
                 paste(names(tab), collapse = ", ")), call. = FALSE)
  x <- tab[[metric]][.ledger_row(tab, model, example)]
  if (!is.null(digits) && is.numeric(x)) round(x, digits) else x
}

# The cumulative comparison table: every model introduced up to (and including) a chapter, in the
# order the chapters introduce them. By default variants (old implementations kept for labelling),
# the leakage warning and the separate HomeBeat stress-score task are left out.
ledger_table <- function(example, upto_chapter = Inf,
                         roles = c("reference", "canonical", "teaching", "second predictor set"),
                         columns = NULL, digits = NULL) {
  tab <- .ledger_read(paste0(example, ".csv"))
  tab <- tab[tab$chapter <= upto_chapter & tab$role %in% roles, , drop = FALSE]
  tab <- tab[order(tab$order), , drop = FALSE]
  if (is.null(columns)) {
    columns <- if ("test_AUC" %in% names(tab))
      c("chapter", "model", "predictor_set", "n_test", "cv_logloss", "test_AUC", "test_Brier", "cal_slope", "cal_intercept")
    else c("chapter", "model", "predictor_set", "n_test", "cv_MAE", "test_MAE", "test_RMSE", "test_R2")
  }
  tab <- tab[, columns, drop = FALSE]
  if (!is.null(digits)) tab[] <- lapply(tab, function(x) if (is.numeric(x)) round(x, digits) else x)
  rownames(tab) <- NULL
  tab
}

# Tonight's patients on Ward 7. example "ward7_reg" gives Δ creatinine (mg/dL), "ward7_aki" P(AKI).
# patient: "Aoife", "Tomas" (or "Tomás"), "Priya".
ledger_tonight <- function(example, model, patient, digits = NULL) {
  tab <- .ledger_read("ward7_tonight.csv")
  tab <- tab[tab$example == example, , drop = FALSE]
  col <- c(aoife = "Aoife", tomas = "Tomas", "tomás" = "Tomas", priya = "Priya")[[tolower(patient)]]
  x <- tab[[col]][.ledger_row(tab, model, paste("ward7_tonight /", example))]
  if (!is.null(digits)) round(x, digits) else x
}

# ONCO-1 patients "#0042" and "#0917": predicted response % (onco1_reg) or P(high response) (onco1_class)
ledger_patient <- function(example, model, label, digits = NULL) {
  tab <- .ledger_read("onco1_patients.csv")
  tab <- tab[tab$example == example & tab$label == label, , drop = FALSE]
  x <- tab$prediction[.ledger_row(tab, model, paste("onco1_patients /", example, label))]
  if (!is.null(digits)) round(x, digits) else x
}

# Lena (HomeBeat): predicted mean resting heart rate (bpm)
ledger_lena <- function(model, digits = NULL) {
  tab <- .ledger_read("homebeat_lena.csv")
  x <- tab$lena_mean_hr[.ledger_row(tab, model, "homebeat_lena")]
  if (!is.null(digits)) round(x, digits) else x
}
