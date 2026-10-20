# Laboratório do capítulo 9
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Carregar os dados ---------------------------------------------------
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

# 2. Quanto está faltando, e onde? --------------------------------------
print((100 * icu.isna().mean()).round(1))
print("pacientes com lacuna:", icu.isna().any(axis=1).sum())

# 3. Quem tem lacunas? --------------------------------------------------
print(icu.groupby(icu["lactate"].isna())["sofa"].mean())
print(icu.groupby(icu["baseline_creat"].isna())["age"].mean())
print("creatinina basal verdadeira vs registrada:",
      round(full["baseline_creat"].mean(), 3), round(icu["baseline_creat"].mean(), 3))

# 4. Predição com lacunas, sem vazamento ---------------------------------
# P10, como nos Capítulos 4-8: o hospital entra como colunas indicadoras (ICU_A é a referência)
icu = pd.get_dummies(icu, columns=["site"], drop_first=True, dtype=float)
full = pd.get_dummies(full, columns=["site"], drop_first=True, dtype=float)
preds = [c for c in icu.columns if c.startswith("site_")] + [
    "age", "sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence", "bun", "crp", "albumin"]
y = "delta_creatinine_48h"
train, test = icu[icu["set"] == "train"], icu[icu["set"] == "test"]

# casos completos: não consegue prever para pacientes de teste com lacuna
cc = train.dropna(subset=preds)
m_cc = LinearRegression().fit(cc[preds], cc[y])
ok = test[preds].notna().all(axis=1).to_numpy()
print("pacientes de treino sem lacuna:", len(cc), "| pacientes de teste:", len(test),
      "| o modelo de casos completos consegue prever:", int(ok.sum()))

# imputação pela média + indicadores de ausência, aprendidos apenas com as linhas de treino
m_imp = make_pipeline(SimpleImputer(strategy="mean", add_indicator=True), LinearRegression())
m_imp.fit(train[preds], train[y])

ftr, fte = full[full["set"] == "train"], full[full["set"] == "test"]
m_full = LinearRegression().fit(ftr[preds], ftr[y])

mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
y_te = test[y].to_numpy()
p_full, p_imp = m_full.predict(fte[preds]), m_imp.predict(test[preds])
p_cc = m_cc.predict(test.loc[ok, preds])
print("todos os 250 pacientes de teste:", {"complete_data": round(mae(y_te, p_full), 3),
                                 "impute_indicator": round(mae(y_te, p_imp), 3)})
print("só os", int(ok.sum()), "que o modelo de casos completos consegue prever:",
      {"complete_case": round(mae(y_te[ok], p_cc), 3), "complete_data": round(mae(y_te[ok], p_full[ok]), 3),
       "impute_indicator": round(mae(y_te[ok], p_imp[ok]), 3)})
# coeficientes dos três indicadores (vêm por último)
print({k: round(float(v), 3) for k, v in zip(["lactate_missing", "baseline_creat_missing", "albumin_missing"],
                                              m_imp[-1].coef_[-3:])})
# 5. Imputação múltipla ------------------------------------------------
# O registro de modelos do livro guarda o resultado do mice (R, m = 20, imputação ajustada no treino).
# Em Python, veja o IterativeImputer do sklearn
# (from sklearn.experimental import enable_iterative_imputer) com sample_posterior=True.
w7 = pd.read_csv("data/precomputed/ledger/ward7_reg.csv")
print(w7.loc[w7["model_id"].isin(["ols", "ols_imp_mean", "ols_mi"]), ["model", "n_test", "test_MAE", "test_RMSE"]].round(3))
