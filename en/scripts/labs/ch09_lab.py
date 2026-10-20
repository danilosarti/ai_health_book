# Chapter 9 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Load the data -------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

icu = pd.read_csv("data/icu_aki_missing.csv")
full = pd.read_csv("data/icu_aki.csv")
split = pd.read_csv("data/splits/ward7_split.csv")
icu = icu.merge(split[["patient_id", "set"]], on="patient_id")
full = full.merge(split[["patient_id", "set"]], on="patient_id")

# 2. How much is missing, and where? ------------------------------------
print((100 * icu.isna().mean()).round(1))
print("patients with a gap:", icu.isna().any(axis=1).sum())

# 3. Who has gaps? ------------------------------------------------------
print(icu.groupby(icu["lactate"].isna())["sofa"].mean())
print(icu.groupby(icu["baseline_creat"].isna())["age"].mean())
print("true vs recorded baseline creatinine:",
      round(full["baseline_creat"].mean(), 3), round(icu["baseline_creat"].mean(), 3))

# 4. Prediction with gaps, without leakage --------------------------------
# P10, as in Chapters 4-8: site enters as dummy columns (ICU_A is the reference)
icu = pd.get_dummies(icu, columns=["site"], drop_first=True, dtype=float)
full = pd.get_dummies(full, columns=["site"], drop_first=True, dtype=float)
preds = [c for c in icu.columns if c.startswith("site_")] + [
    "age", "sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence", "bun", "crp", "albumin"]
y = "delta_creatinine_48h"
train, test = icu[icu["set"] == "train"], icu[icu["set"] == "test"]

# complete cases: cannot predict for test patients with a gap
cc = train.dropna(subset=preds)
m_cc = LinearRegression().fit(cc[preds], cc[y])
ok = test[preds].notna().all(axis=1).to_numpy()
print("training patients without a gap:", len(cc), "| test patients:", len(test),
      "| complete-case model can predict:", int(ok.sum()))

# mean imputation + missing indicators, learned on the training rows only
m_imp = make_pipeline(SimpleImputer(strategy="mean", add_indicator=True), LinearRegression())
m_imp.fit(train[preds], train[y])

ftr, fte = full[full["set"] == "train"], full[full["set"] == "test"]
m_full = LinearRegression().fit(ftr[preds], ftr[y])

mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
y_te = test[y].to_numpy()
p_full, p_imp = m_full.predict(fte[preds]), m_imp.predict(test[preds])
p_cc = m_cc.predict(test.loc[ok, preds])
print("all 250 test patients:", {"complete_data": round(mae(y_te, p_full), 3),
                                 "impute_indicator": round(mae(y_te, p_imp), 3)})
print("only the", int(ok.sum()), "the complete-case model can predict:",
      {"complete_case": round(mae(y_te[ok], p_cc), 3), "complete_data": round(mae(y_te[ok], p_full[ok]), 3),
       "impute_indicator": round(mae(y_te[ok], p_imp[ok]), 3)})
# coefficients of the three indicators (they come last)
print({k: round(float(v), 3) for k, v in zip(["lactate_missing", "baseline_creat_missing", "albumin_missing"],
                                              m_imp[-1].coef_[-3:])})
# 5. Multiple imputation -------------------------------------------------
# The book's model ledger holds the result of mice (R, m = 20, imputation fitted on training rows).
# In Python, see sklearn's IterativeImputer
# (from sklearn.experimental import enable_iterative_imputer) with sample_posterior=True.
w7 = pd.read_csv("data/precomputed/ledger/ward7_reg.csv")
print(w7.loc[w7["model_id"].isin(["ols", "ols_imp_mean", "ols_mi"]), ["model", "n_test", "test_MAE", "test_RMSE"]].round(3))
