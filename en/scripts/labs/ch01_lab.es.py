# Laboratorio del capítulo 1
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. Cargar los datos -----------------------------------------------------
import pandas as pd
import numpy as np
icu = pd.read_csv("data/icu_aki.csv")
tonight = pd.read_csv("data/ward7_tonight.csv")
print(icu.shape)
print(icu.dtypes)

# 2. Describir --------------------------------------------------------------
print(icu["delta_creatinine_48h"].describe())
icu["aki"] = icu["delta_creatinine_48h"] >= 0.3      # criterio KDIGO
print("riesgo de referencia:", icu["aki"].mean())
print(icu.groupby("site")["aki"].mean().round(2))

# 3. División entrenamiento / prueba: la división fija del libro (1000 / 250) --
split = pd.read_csv("data/splits/ward7_split.csv")
assert (split["patient_id"] == icu["patient_id"]).all()
train, test = icu[split["set"] == "train"], icu[split["set"] == "test"]

# 4. Dos modelos del desenlace ---------------------------------------------
pred_avg = train["delta_creatinine_48h"].mean()                 # (a) modelo de la media
b, a = np.polyfit(train["sofa"], train["delta_creatinine_48h"], deg=1)   # (b) recta
pred_line = a + b * test["sofa"]

def mae(y, yhat):
    return np.mean(np.abs(y - yhat))
print({"average": round(float(mae(test["delta_creatinine_48h"], pred_avg)), 3),
       "sofa_line": round(float(mae(test["delta_creatinine_48h"], pred_line)), 3)})

# 5. Desplegar: predicciones para los pacientes de esta noche -----------------
print(tonight.assign(predicted_change=(a + b * tonight["sofa"]).round(2))[["name", "sofa", "predicted_change"]])
