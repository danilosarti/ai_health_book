# Laboratorio del capítulo 9
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. Cargar los datos ----------------------------------------------------
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

# 2. ¿Cuánto falta, y dónde? ---------------------------------------------
print((100 * icu.isna().mean()).round(1))
print("pacientes con algún hueco:", icu.isna().any(axis=1).sum())

# 3. ¿Quién tiene huecos? ------------------------------------------------
print(icu.groupby(icu["lactate"].isna())["sofa"].mean())
print(icu.groupby(icu["baseline_creat"].isna())["age"].mean())
print("creatinina basal verdadera frente a registrada:",
      round(full["baseline_creat"].mean(), 3), round(icu["baseline_creat"].mean(), 3))

# 4. Predicción con huecos, sin fuga de información ----------------------
# P10, como en los Capítulos 4-8: el hospital entra como columnas indicadoras (ICU_A es la referencia)
icu = pd.get_dummies(icu, columns=["site"], drop_first=True, dtype=float)
full = pd.get_dummies(full, columns=["site"], drop_first=True, dtype=float)
preds = [c for c in icu.columns if c.startswith("site_")] + [
    "age", "sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence", "bun", "crp", "albumin"]
y = "delta_creatinine_48h"
train, test = icu[icu["set"] == "train"], icu[icu["set"] == "test"]

# casos completos: no puede predecir para los pacientes de prueba con algún hueco
cc = train.dropna(subset=preds)
m_cc = LinearRegression().fit(cc[preds], cc[y])
ok = test[preds].notna().all(axis=1).to_numpy()
print("pacientes de entrenamiento sin huecos:", len(cc), "| pacientes de prueba:", len(test),
      "| el modelo de casos completos puede predecir:", int(ok.sum()))

# imputación con la media + indicadores de ausencia, aprendidos solo con las filas de entrenamiento
m_imp = make_pipeline(SimpleImputer(strategy="mean", add_indicator=True), LinearRegression())
m_imp.fit(train[preds], train[y])

ftr, fte = full[full["set"] == "train"], full[full["set"] == "test"]
m_full = LinearRegression().fit(ftr[preds], ftr[y])

mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
y_te = test[y].to_numpy()
p_full, p_imp = m_full.predict(fte[preds]), m_imp.predict(test[preds])
p_cc = m_cc.predict(test.loc[ok, preds])
print("los 250 pacientes de prueba:", {"complete_data": round(mae(y_te, p_full), 3),
                                 "impute_indicator": round(mae(y_te, p_imp), 3)})
print("solo los", int(ok.sum()), "que el modelo de casos completos puede predecir:",
      {"complete_case": round(mae(y_te[ok], p_cc), 3), "complete_data": round(mae(y_te[ok], p_full[ok]), 3),
       "impute_indicator": round(mae(y_te[ok], p_imp[ok]), 3)})
# coeficientes de los tres indicadores (van al final)
print({k: round(float(v), 3) for k, v in zip(["lactate_missing", "baseline_creat_missing", "albumin_missing"],
                                              m_imp[-1].coef_[-3:])})
# 5. Imputación múltiple -------------------------------------------------
# El registro de modelos del libro guarda el resultado de mice (R, m = 20, imputación ajustada con el entrenamiento).
# En Python, consulta el IterativeImputer de sklearn
# (from sklearn.experimental import enable_iterative_imputer) con sample_posterior=True.
w7 = pd.read_csv("data/precomputed/ledger/ward7_reg.csv")
print(w7.loc[w7["model_id"].isin(["ols", "ols_imp_mean", "ols_mi"]), ["model", "n_test", "test_MAE", "test_RMSE"]].round(3))
