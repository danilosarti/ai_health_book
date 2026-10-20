# Laboratorio del capitolo 1
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Caricare i dati -------------------------------------------------------
import pandas as pd
import numpy as np
icu = pd.read_csv("data/icu_aki.csv")
tonight = pd.read_csv("data/ward7_tonight.csv")
print(icu.shape)
print(icu.dtypes)

# 2. Descrivere -------------------------------------------------------------
print(icu["delta_creatinine_48h"].describe())
icu["aki"] = icu["delta_creatinine_48h"] >= 0.3      # criterio KDIGO
print("rischio di base:", icu["aki"].mean())
print(icu.groupby("site")["aki"].mean().round(2))

# 3. Suddivisione addestramento / test: quella fissa del libro (1.000 / 250) --
split = pd.read_csv("data/splits/ward7_split.csv")
assert (split["patient_id"] == icu["patient_id"]).all()
train, test = icu[split["set"] == "train"], icu[split["set"] == "test"]

# 4. Due modelli dell'esito -------------------------------------------------
pred_avg = train["delta_creatinine_48h"].mean()                 # (a) modello della media
b, a = np.polyfit(train["sofa"], train["delta_creatinine_48h"], deg=1)   # (b) retta
pred_line = a + b * test["sofa"]

def mae(y, yhat):
    return np.mean(np.abs(y - yhat))
print({"average": round(float(mae(test["delta_creatinine_48h"], pred_avg)), 3),
       "sofa_line": round(float(mae(test["delta_creatinine_48h"], pred_line)), 3)})

# 5. Impiego: previsioni per i pazienti di stasera --------------------------
print(tonight.assign(predicted_change=(a + b * tonight["sofa"]).round(2))[["name", "sofa", "predicted_change"]])
