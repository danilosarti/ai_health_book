# Laboratório do capítulo 3
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Carregar os dados e a divisão usada em todo o livro ----------------------
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

icu = pd.read_csv("data/icu_aki.csv")
split = pd.read_csv("data/splits/ward7_split.csv")
tonight = pd.read_csv("data/ward7_tonight.csv")
assert (icu["patient_id"] == split["patient_id"]).all()
train = icu[split["set"] == "train"]
test = icu[split["set"] == "test"]                      # trancado até a etapa 5
fold = split.loc[split["set"] == "train", "fold"].astype(int).to_numpy()
cv = PredefinedSplit(fold - 1)                          # as partições usadas em todo o livro
feat, y = ["sofa", "baseline_creat"], "delta_creatinine_48h"

# 2. Referências, com validação cruzada ---------------------------------------
def cv_mae(model, cols):
    return float(-cross_val_score(model, train[cols], train[y], cv=cv, scoring="neg_mean_absolute_error").mean())
print({"mean": round(cv_mae(DummyRegressor(strategy="mean"), feat), 3),
       "sofa_line": round(cv_mae(LinearRegression(), ["sofa"]), 3),
       "two_line": round(cv_mae(LinearRegression(), feat), 3)})

# 3. kNN sintonizado por validação cruzada. O padronizador fica DENTRO do pipeline,
#    então é reajustado nas partições de ajuste de cada rodada: nenhum vazamento.
ks = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150]
search = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsRegressor()),
                      {"kneighborsregressor__n_neighbors": ks},
                      cv=cv, scoring="neg_mean_absolute_error")
search.fit(train[feat], train[y])
print(pd.DataFrame({"k": ks, "cv_mae": -search.cv_results_["mean_test_score"]}).round(3))
k_best = search.best_params_["kneighborsregressor__n_neighbors"]

# 4. (O GridSearchCV já reajustou o melhor modelo com todos os pacientes de treino)

# 5. Abrir o conjunto de teste UMA VEZ -----------------------------------------
pred_test = search.predict(test[feat])
print({"k": int(k_best),
       "test_MAE": round(float(mean_absolute_error(test[y], pred_test)), 3),
       "test_RMSE": round(float(np.sqrt(mean_squared_error(test[y], pred_test))), 3),
       "test_R2": round(float(1 - ((test[y] - pred_test) ** 2).sum() / ((test[y] - train[y].mean()) ** 2).sum()), 3)})

# 6. kNN com os dez preditores (P10; site como colunas indicadoras), k = 1..150 por VC --
p10 = ["age", "sofa", "lactate", "map_min", "baseline_creat", "protocol_adherence", "bun", "crp", "albumin"]
X10 = pd.concat([pd.get_dummies(icu["site"], prefix="site", drop_first=True, dtype=float), icu[p10]], axis=1)
is_train = (split["set"] == "train").to_numpy()
search10 = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsRegressor()),
                        {"kneighborsregressor__n_neighbors": list(range(1, 151))},
                        cv=cv, scoring="neg_mean_absolute_error")
search10.fit(X10[is_train], train[y])
print({"k": int(search10.best_params_["kneighborsregressor__n_neighbors"]),
       "cv_MAE": round(float(-search10.best_score_), 3),
       "test_MAE": round(float(mean_absolute_error(test[y], search10.predict(X10[~is_train]))), 3)})

# 7. Implantar: predições para os pacientes desta noite ------------------------
print(tonight.assign(predicted_change=search.predict(tonight[feat]).round(2))[["name", "sofa", "baseline_creat", "predicted_change"]])
