# Chapter 5 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Data and binary outcome ----------------------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
icu = pd.read_csv("data/icu_aki.csv")
tonight = pd.read_csv("data/ward7_tonight.csv")
icu["aki"] = (icu["delta_creatinine_48h"] >= 0.3).astype(int)   # KDIGO criterion
print("prevalence:", icu["aki"].mean())

# 2. Train / test split: the book's shared split (Chapter 3) ---------------
split = pd.read_csv("data/splits/ward7_split.csv")
icu = icu.merge(split, on="patient_id")
train, test = icu[icu["set"] == "train"], icu[icu["set"] == "test"]
features = ["sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence"]

# 3. Logistic regression and odds ratios --------------------------------------
model = LogisticRegression(penalty=None, max_iter=5000).fit(train[features], train["aki"])
print(pd.Series(np.round(np.exp(model.coef_[0]), 2), index=features))

# 4. Predicted probabilities on the test set -------------------------------
p = model.predict_proba(test[features])[:, 1]
y = test["aki"].to_numpy()

# 5. Confusion matrices at two thresholds ([[TN, FP], [FN, TP]]) -------------
for t in (0.50, 0.25):
    print("t =", t, confusion_matrix(y, (p >= t).astype(int)).tolist())

# 6. ROC AUC ---------------------------------------------------------------------
print("AUC:", round(roc_auc_score(y, p), 3))
fpr, tpr, _ = roc_curve(y, p)

# 7. Calibration -----------------------------------------------------------------
observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
print(pd.DataFrame({"predicted": predicted.round(2), "observed": observed.round(2)}))
print("Brier score:", round(brier_score_loss(y, p), 3))

# 8. Net benefit --------------------------------------------------------------------
def net_benefit(y, p, t):
    n = len(y)
    return ((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t)
for t in (0.10, 0.25, 0.50):
    print(t, round(net_benefit(y, p, t), 3), round(y.mean() - (1 - y.mean()) * t / (1 - t), 3))

# 9. Tonight's patients -------------------------------------------------------------
print(tonight.assign(risk=model.predict_proba(tonight[features])[:, 1].round(2))[["name", "risk"]])
