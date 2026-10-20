# Laboratorio del capitolo 7
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Caricare le immagini e le previsioni precalcolate della CNN ------------
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
sub = pd.read_csv("data/pneumonia_subset.csv")
pred = pd.read_csv("data/precomputed/ch07/test_predictions.csv")
px = [c for c in sub.columns if c.startswith("px")]      # le 784 colonne dei pixel
train, test = sub[sub["split"] == "train"], sub[sub["split"] == "test"]
print(pd.crosstab(sub["split"], sub["label"]))           # 0 = normale, 1 = polmonite

# 2. Le immagini sono tensori ------------------------------------------------
X_train = train[px].to_numpy().reshape(-1, 28, 28) / 255  # (immagini, altezza, larghezza)
X_test = test[px].to_numpy().reshape(-1, 28, 28) / 255
print(X_train.shape, X_train[0].shape)

# 3. Pooling: media su blocchi 4 x 4, così ogni immagine diventa 7 x 7 = 49 numeri
def pool7(X):
    return X.reshape(-1, 7, 4, 7, 4).mean(axis=(2, 4)).reshape(len(X), 49)
P_train, P_test = pool7(X_train), pool7(X_test)
mu, sd = P_train.mean(axis=0), P_train.std(axis=0)   # standardizzare con le statistiche di ADDESTRAMENTO
P_train, P_test = (P_train - mu) / sd, (P_test - mu) / sd
print(P_train.shape)

# 4. Un neurone (regressione logistica) contro uno strato nascosto ----------
y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()
neuron = LogisticRegression(C=1, max_iter=5000).fit(P_train, y_train)
mlp = MLPClassifier(hidden_layer_sizes=(4,), alpha=1, solver="lbfgs", max_iter=5000,
                    random_state=2026).fit(P_train, y_train)
p_neuron = neuron.predict_proba(P_test)[:, 1]
p_mlp = mlp.predict_proba(P_test)[:, 1]
p_cnn = pred["p_cnn"].to_numpy()[test["orig_index"].to_numpy()]   # CNN addestrata su tutte le 4.708 immagini

def accuracy(p, y, t=0.5):
    return round(float(np.mean((p >= t) == y)), 3)
print({"neuron": accuracy(p_neuron, y_test), "hidden_layer": accuracy(p_mlp, y_test),
       "cnn_full_data": accuracy(p_cnn, y_test)})

# 5. Quali errori? -----------------------------------------------------------
print(pd.crosstab(y_test, (p_cnn >= 0.5).astype(int), rownames=["truth"], colnames=["predicted"]))
print(pd.crosstab(y_test, (p_cnn >= 0.3).astype(int), rownames=["truth"], colnames=["predicted"]))
