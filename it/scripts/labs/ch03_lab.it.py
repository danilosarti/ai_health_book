# Laboratorio del capitolo 3
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Caricare i dati e la suddivisione usata in tutto il libro ----------------
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
test = icu[split["set"] == "test"]                      # chiuso fino al passo 5
fold = split.loc[split["set"] == "train", "fold"].astype(int).to_numpy()
cv = PredefinedSplit(fold - 1)                          # i blocchi usati in tutto il libro
feat, y = ["sofa", "baseline_creat"], "delta_creatinine_48h"

# 2. Riferimenti, in validazione incrociata ------------------------------------
def cv_mae(model, cols):
    return float(-cross_val_score(model, train[cols], train[y], cv=cv, scoring="neg_mean_absolute_error").mean())
print({"mean": round(cv_mae(DummyRegressor(strategy="mean"), feat), 3),
       "sofa_line": round(cv_mae(LinearRegression(), ["sofa"]), 3),
       "two_line": round(cv_mae(LinearRegression(), feat), 3)})

# 3. kNN ottimizzato con la validazione incrociata. La standardizzazione sta DENTRO
#    la pipeline, quindi viene ristimata sui blocchi di stima di ogni turno: nessuna fuga.
ks = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150]
search = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsRegressor()),
                      {"kneighborsregressor__n_neighbors": ks},
                      cv=cv, scoring="neg_mean_absolute_error")
search.fit(train[feat], train[y])
print(pd.DataFrame({"k": ks, "cv_mae": -search.cv_results_["mean_test_score"]}).round(3))
k_best = search.best_params_["kneighborsregressor__n_neighbors"]

# 4. (GridSearchCV ha già ristimato il modello migliore su tutti i pazienti di addestramento)

# 5. Aprire l'insieme di test UNA VOLTA ----------------------------------------
pred_test = search.predict(test[feat])
print({"k": int(k_best),
       "test_MAE": round(float(mean_absolute_error(test[y], pred_test)), 3),
       "test_RMSE": round(float(np.sqrt(mean_squared_error(test[y], pred_test))), 3),
       "test_R2": round(float(1 - ((test[y] - pred_test) ** 2).sum() / ((test[y] - train[y].mean()) ** 2).sum()), 3)})

# 6. kNN con tutti i dieci predittori (P10; site come colonne indicatrici), k = 1..150 con VC --
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

# 7. Impiego: previsioni per i pazienti di stasera -----------------------------
print(tonight.assign(predicted_change=search.predict(tonight[feat]).round(2))[["name", "sofa", "baseline_creat", "predicted_change"]])
