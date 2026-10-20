# Laboratorio del capitolo 9
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Caricare i dati -------------------------------------------------------
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

# 2. Quanto manca, e dove? ------------------------------------------------
print((100 * icu.isna().mean()).round(1))
print("pazienti con una lacuna:", icu.isna().any(axis=1).sum())

# 3. Chi ha lacune? -------------------------------------------------------
print(icu.groupby(icu["lactate"].isna())["sofa"].mean())
print(icu.groupby(icu["baseline_creat"].isna())["age"].mean())
print("creatinina di base vera e registrata:",
      round(full["baseline_creat"].mean(), 3), round(icu["baseline_creat"].mean(), 3))

# 4. Previsione con lacune, senza fughe di informazione --------------------
# P10, come nei Capitoli 4-8: l'ospedale entra come colonne indicatrici (ICU_A è il riferimento)
icu = pd.get_dummies(icu, columns=["site"], drop_first=True, dtype=float)
full = pd.get_dummies(full, columns=["site"], drop_first=True, dtype=float)
preds = [c for c in icu.columns if c.startswith("site_")] + [
    "age", "sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence", "bun", "crp", "albumin"]
y = "delta_creatinine_48h"
train, test = icu[icu["set"] == "train"], icu[icu["set"] == "test"]

# casi completi: non può prevedere per i pazienti di test con una lacuna
cc = train.dropna(subset=preds)
m_cc = LinearRegression().fit(cc[preds], cc[y])
ok = test[preds].notna().all(axis=1).to_numpy()
print("pazienti di addestramento senza lacune:", len(cc), "| pazienti di test:", len(test),
      "| il modello sui casi completi può prevedere:", int(ok.sum()))

# imputazione con la media + indicatori di dato mancante, appresi solo dalle righe di addestramento
m_imp = make_pipeline(SimpleImputer(strategy="mean", add_indicator=True), LinearRegression())
m_imp.fit(train[preds], train[y])

ftr, fte = full[full["set"] == "train"], full[full["set"] == "test"]
m_full = LinearRegression().fit(ftr[preds], ftr[y])

mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
y_te = test[y].to_numpy()
p_full, p_imp = m_full.predict(fte[preds]), m_imp.predict(test[preds])
p_cc = m_cc.predict(test.loc[ok, preds])
print("tutti i 250 pazienti di test:", {"complete_data": round(mae(y_te, p_full), 3),
                                 "impute_indicator": round(mae(y_te, p_imp), 3)})
print("solo i", int(ok.sum()), "che il modello sui casi completi può prevedere:",
      {"complete_case": round(mae(y_te[ok], p_cc), 3), "complete_data": round(mae(y_te[ok], p_full[ok]), 3),
       "impute_indicator": round(mae(y_te[ok], p_imp[ok]), 3)})
# coefficienti dei tre indicatori (vengono per ultimi)
print({k: round(float(v), 3) for k, v in zip(["lactate_missing", "baseline_creat_missing", "albumin_missing"],
                                              m_imp[-1].coef_[-3:])})
# 5. Imputazione multipla ----------------------------------------------
# Il registro dei modelli del libro contiene il risultato di mice (R, m = 20, imputazione stimata sul training).
# In Python, vedi IterativeImputer di sklearn
# (from sklearn.experimental import enable_iterative_imputer) con sample_posterior=True.
w7 = pd.read_csv("data/precomputed/ledger/ward7_reg.csv")
print(w7.loc[w7["model_id"].isin(["ols", "ols_imp_mean", "ols_mi"]), ["model", "n_test", "test_MAE", "test_RMSE"]].round(3))
