# =====================================================================
# Fixed train/test splits and cross-validation folds, book-wide.
# Every chapter of Part II uses these files, so that models fitted in
# different chapters are compared on exactly the same patients.
#   data/splits/ward7_split.csv : 1 000 training + 250 test patients,
#                                 training patients assigned to 5 folds
#   data/splits/onco1_split.csv : 800 training + 200 test patients,
#                                 training patients assigned to 5 folds
# Columns: id, set ("train"/"test"), fold (1-5 for training rows, empty for test)
# Usage (from the book folder): Rscript scripts/simulate/make_splits.R
#
#   data/splits/homebeat_split.csv : 400 training + 100 test participants, 5 folds of 80.
#     Written by the companion script scripts/simulate/make_homebeat_split.py (Python),
#     because it must reproduce the NumPy permutation default_rng(2026) that Chapters 8
#     and 19 already use; R's sample() cannot. See that script for the columns.
# =====================================================================
make_split <- function(ids, n_test, V = 5, seed) {
  set.seed(seed)
  n <- length(ids)
  test <- sort(sample(n, n_test))
  set <- rep("train", n); set[test] <- "test"
  fold <- rep(NA_integer_, n)
  train <- which(set == "train")
  fold[train] <- sample(rep(seq_len(V), length.out = length(train)))
  data.frame(id = ids, set = set, fold = fold)
}

dir.create("data/splits", showWarnings = FALSE, recursive = TRUE)

icu <- read.csv("data/icu_aki.csv")
w7 <- make_split(icu$patient_id, n_test = 250, seed = 2026)
names(w7)[1] <- "patient_id"
write.csv(w7, "data/splits/ward7_split.csv", row.names = FALSE, na = "")

onco <- read.csv("data/onco1.csv")
o1 <- make_split(onco$patient_id, n_test = 200, seed = 2027)
names(o1)[1] <- "patient_id"
write.csv(o1, "data/splits/onco1_split.csv", row.names = FALSE, na = "")

# --- checks ---
stopifnot(sum(w7$set == "test") == 250, all(table(w7$fold) == 200))
stopifnot(sum(o1$set == "test") == 200, all(table(o1$fold) == 160))
cat("Ward 7: ", table(w7$set), "\nONCO-1: ", table(o1$set), "\n")
