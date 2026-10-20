# Laboratório do capítulo 1
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Carregar os dados ---------------------------------------------------
import pandas as pd
import numpy as np
icu = pd.read_csv("data/icu_aki.csv")
tonight = pd.read_csv("data/ward7_tonight.csv")
print(icu.shape)
print(icu.dtypes)

# 2. Descrever --------------------------------------------------------------
print(icu["delta_creatinine_48h"].describe())
icu["aki"] = icu["delta_creatinine_48h"] >= 0.3      # critério KDIGO
print("risco de referência:", icu["aki"].mean())
print(icu.groupby("site")["aki"].mean().round(2))

# 3. Divisão treino / teste: a divisão fixa do livro (1.000 / 250) ------------
split = pd.read_csv("data/splits/ward7_split.csv")
assert (split["patient_id"] == icu["patient_id"]).all()
train, test = icu[split["set"] == "train"], icu[split["set"] == "test"]

# 4. Dois modelos do desfecho ----------------------------------------------
pred_avg = train["delta_creatinine_48h"].mean()                 # (a) modelo da média
b, a = np.polyfit(train["sofa"], train["delta_creatinine_48h"], deg=1)   # (b) reta
pred_line = a + b * test["sofa"]

def mae(y, yhat):
    return np.mean(np.abs(y - yhat))
print({"average": round(float(mae(test["delta_creatinine_48h"], pred_avg)), 3),
       "sofa_line": round(float(mae(test["delta_creatinine_48h"], pred_line)), 3)})

# 5. Implantar: predições para os pacientes desta noite -----------------------
print(tonight.assign(predicted_change=(a + b * tonight["sofa"]).round(2))[["name", "sofa", "predicted_change"]])
