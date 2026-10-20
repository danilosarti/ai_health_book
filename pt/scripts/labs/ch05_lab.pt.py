# Laboratório do capítulo 5
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Dados e desfecho binário ------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
icu = pd.read_csv("data/icu_aki.csv")
tonight = pd.read_csv("data/ward7_tonight.csv")
icu["aki"] = (icu["delta_creatinine_48h"] >= 0.3).astype(int)   # critério KDIGO
print("prevalência:", icu["aki"].mean())

# 2. Divisão treino / teste: a divisão compartilhada do livro (Capítulo 3) ---
split = pd.read_csv("data/splits/ward7_split.csv")
icu = icu.merge(split, on="patient_id")
train, test = icu[icu["set"] == "train"], icu[icu["set"] == "test"]
features = ["sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence"]

# 3. Regressão logística e razões de chances ---------------------------------
model = LogisticRegression(penalty=None, max_iter=5000).fit(train[features], train["aki"])
print(pd.Series(np.round(np.exp(model.coef_[0]), 2), index=features))

# 4. Probabilidades previstas no conjunto de teste ---------------------------
p = model.predict_proba(test[features])[:, 1]
y = test["aki"].to_numpy()

# 5. Matrizes de confusão em dois limiares ([[TN, FP], [FN, TP]]) ------------
for t in (0.50, 0.25):
    print("t =", t, confusion_matrix(y, (p >= t).astype(int)).tolist())

# 6. AUC ROC ------------------------------------------------------------------
print("AUC:", round(roc_auc_score(y, p), 3))
fpr, tpr, _ = roc_curve(y, p)

# 7. Calibração ---------------------------------------------------------------
observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
print(pd.DataFrame({"predicted": predicted.round(2), "observed": observed.round(2)}))
print("Escore de Brier:", round(brier_score_loss(y, p), 3))

# 8. Benefício líquido ---------------------------------------------------------
def net_benefit(y, p, t):
    n = len(y)
    return ((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t)
for t in (0.10, 0.25, 0.50):
    print(t, round(net_benefit(y, p, t), 3), round(y.mean() - (1 - y.mean()) * t / (1 - t), 3))

# 9. Os pacientes desta noite --------------------------------------------------
print(tonight.assign(risk=model.predict_proba(tonight[features])[:, 1].round(2))[["name", "risk"]])
