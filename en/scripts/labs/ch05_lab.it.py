# Laboratorio del capitolo 5
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Dati ed esito binario ---------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
icu = pd.read_csv("data/icu_aki.csv")
tonight = pd.read_csv("data/ward7_tonight.csv")
icu["aki"] = (icu["delta_creatinine_48h"] >= 0.3).astype(int)   # criterio KDIGO
print("prevalenza:", icu["aki"].mean())

# 2. Suddivisione addestramento / test: quella condivisa del libro (Capitolo 3)
split = pd.read_csv("data/splits/ward7_split.csv")
icu = icu.merge(split, on="patient_id")
train, test = icu[icu["set"] == "train"], icu[icu["set"] == "test"]
features = ["sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence"]

# 3. Regressione logistica e odds ratio ---------------------------------------
model = LogisticRegression(penalty=None, max_iter=5000).fit(train[features], train["aki"])
print(pd.Series(np.round(np.exp(model.coef_[0]), 2), index=features))

# 4. Probabilità previste sull'insieme di test -------------------------------
p = model.predict_proba(test[features])[:, 1]
y = test["aki"].to_numpy()

# 5. Matrici di confusione a due soglie ([[TN, FP], [FN, TP]]) ---------------
for t in (0.50, 0.25):
    print("t =", t, confusion_matrix(y, (p >= t).astype(int)).tolist())

# 6. AUC ROC ---------------------------------------------------------------------
print("AUC:", round(roc_auc_score(y, p), 3))
fpr, tpr, _ = roc_curve(y, p)

# 7. Calibrazione ---------------------------------------------------------------
observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
print(pd.DataFrame({"predicted": predicted.round(2), "observed": observed.round(2)}))
print("Punteggio di Brier:", round(brier_score_loss(y, p), 3))

# 8. Beneficio netto --------------------------------------------------------------
def net_benefit(y, p, t):
    n = len(y)
    return ((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t)
for t in (0.10, 0.25, 0.50):
    print(t, round(net_benefit(y, p, t), 3), round(y.mean() - (1 - y.mean()) * t / (1 - t), 3))

# 9. I pazienti di stasera --------------------------------------------------------
print(tonight.assign(risk=model.predict_proba(tonight[features])[:, 1].round(2))[["name", "risk"]])
