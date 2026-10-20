# Chapter 1 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Load the data -------------------------------------------------------
import pandas as pd
import numpy as np
icu = pd.read_csv("data/icu_aki.csv")
tonight = pd.read_csv("data/ward7_tonight.csv")
print(icu.shape)
print(icu.dtypes)

# 2. Describe ---------------------------------------------------------------
print(icu["delta_creatinine_48h"].describe())
icu["aki"] = icu["delta_creatinine_48h"] >= 0.3      # KDIGO criterion
print("baseline risk:", icu["aki"].mean())
print(icu.groupby("site")["aki"].mean().round(2))

# 3. Train / test split: the book's fixed split (1 000 / 250) -----------------
split = pd.read_csv("data/splits/ward7_split.csv")
assert (split["patient_id"] == icu["patient_id"]).all()
train, test = icu[split["set"] == "train"], icu[split["set"] == "test"]

# 4. Two models of the outcome ---------------------------------------------
pred_avg = train["delta_creatinine_48h"].mean()                 # (a) average model
b, a = np.polyfit(train["sofa"], train["delta_creatinine_48h"], deg=1)   # (b) straight line
pred_line = a + b * test["sofa"]

def mae(y, yhat):
    return np.mean(np.abs(y - yhat))
print({"average": round(float(mae(test["delta_creatinine_48h"], pred_avg)), 3),
       "sofa_line": round(float(mae(test["delta_creatinine_48h"], pred_line)), 3)})

# 5. Deploy: predictions for tonight's patients -------------------------------
print(tonight.assign(predicted_change=(a + b * tonight["sofa"]).round(2))[["name", "sofa", "predicted_change"]])
