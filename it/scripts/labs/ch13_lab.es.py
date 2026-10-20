# Laboratorio del capítulo 13
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. UCI Sala 7: ingenua frente a ajustada (verdad -0.28) -------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def ols_coef(data, y, cols):
    X = np.column_stack([np.ones(len(data))] + [data[c] for c in cols])
    beta, *_ = np.linalg.lstsq(X, data[y], rcond=None)
    return float(beta[1])

icu = pd.read_csv("data/icu_aki.csv")
print({"naive": round(ols_coef(icu, "delta_creatinine_48h", ["protocol_adherence"]), 3),
       "adjusted": round(ols_coef(icu, "delta_creatinine_48h", ["protocol_adherence", "sofa"]), 3),
       "truth": -0.28})

# 2. UCI Sala 7, versión ampliada: un conjunto válido, un mediador y un colisionador
dag = pd.read_csv("data/ward7_dag.csv")
sets = {"valid": ["sofa", "baseline_creat"], "mediator": ["sofa", "nephrotox"],
        "collider": ["sofa", "creat_tests"]}
print({k: round(ols_coef(dag, "delta_creatinine_48h", ["protocol_adherence"] + v), 3)
       for k, v in sets.items()})

# 3. Ensayo ONCO-1: la referencia aleatorizada ------------------------------
onco = pd.read_csv("data/onco1.csv")
arm = onco.groupby("treatment")["response_percent"].mean()
trial = arm["chemo"] - arm["no_chemo"]

# 4. ONCO-1 en la práctica habitual: ingenua y ajuste por regresión ---------
obs = pd.read_csv("data/onco1_routine.csv")
X = pd.get_dummies(obs[["patient_age", "tumor_grade", "performance_score", "gene_05"]],
                   columns=["tumor_grade"], drop_first=True).astype(float)
t1 = (obs["chemo"] == 1).to_numpy()
y = obs["response_percent"].to_numpy()
naive_obs = y[t1].mean() - y[~t1].mean()
Xr = np.column_stack([np.ones(len(obs)), obs["chemo"], X])
reg = np.linalg.lstsq(Xr, y, rcond=None)[0][1]

# 5. Puntuación de propensión y ponderación por el inverso de la probabilidad
ps = LogisticRegression(penalty=None, max_iter=5000).fit(X, obs["chemo"]).predict_proba(X)[:, 1]
w = np.where(t1, 1 / ps, 1 / (1 - ps))
ipw = np.average(y[t1], weights=w[t1]) - np.average(y[~t1], weights=w[~t1])

# 6. Emparejamiento voraz 1:1 por el vecino más cercano en el logit de la puntuación
rng = np.random.default_rng(1)
lp = np.log(ps / (1 - ps)); caliper = 0.2 * lp.std(ddof=1)
pool = list(np.where(~t1)[0]); pairs = []
for i in rng.permutation(np.where(t1)[0]):
    if not pool:
        break
    d = np.abs(lp[pool] - lp[i]); j = int(np.argmin(d))
    if d[j] <= caliper:
        pairs.append((i, pool.pop(j)))
pairs = np.array(pairs)
matched = np.mean(y[pairs[:, 0]] - y[pairs[:, 1]])
print({k: round(float(v), 1) for k, v in
       {"trial": trial, "naive": naive_obs, "regression": reg, "ipw": ipw, "matching": matched}.items()})

# 7. Equilibrio: diferencias de medias estandarizadas antes y después de ponderar
def smd(x, w=None):
    w = np.ones(len(x)) if w is None else w
    num = np.average(x[t1], weights=w[t1]) - np.average(x[~t1], weights=w[~t1])
    return num / np.sqrt((x[t1].var(ddof=1) + x[~t1].var(ddof=1)) / 2)
covs = {"age": obs["patient_age"].to_numpy(), "G3": (obs["tumor_grade"] == "G3").to_numpy(float),
        "performance": obs["performance_score"].to_numpy(float), "gene_05": obs["gene_05"].to_numpy()}
balance = pd.DataFrame({"before": {k: smd(v) for k, v in covs.items()},
                        "after_ipw": {k: smd(v, w) for k, v in covs.items()}}).round(3)
print(balance)
