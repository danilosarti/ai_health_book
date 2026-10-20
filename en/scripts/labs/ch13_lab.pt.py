# Laboratório do capítulo 13
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. UTI Ala 7: ingênuo versus ajustado (verdade -0.28) ---------------------
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

# 2. UTI Ala 7, versão ampliada: um conjunto válido, um mediador e um colisor
dag = pd.read_csv("data/ward7_dag.csv")
sets = {"valid": ["sofa", "baseline_creat"], "mediator": ["sofa", "nephrotox"],
        "collider": ["sofa", "creat_tests"]}
print({k: round(ols_coef(dag, "delta_creatinine_48h", ["protocol_adherence"] + v), 3)
       for k, v in sets.items()})

# 3. Ensaio ONCO-1: a referência randomizada --------------------------------
onco = pd.read_csv("data/onco1.csv")
arm = onco.groupby("treatment")["response_percent"].mean()
trial = arm["chemo"] - arm["no_chemo"]

# 4. ONCO-1 na prática de rotina: ingênuo e ajuste por regressão ------------
obs = pd.read_csv("data/onco1_routine.csv")
X = pd.get_dummies(obs[["patient_age", "tumor_grade", "performance_score", "gene_05"]],
                   columns=["tumor_grade"], drop_first=True).astype(float)
t1 = (obs["chemo"] == 1).to_numpy()
y = obs["response_percent"].to_numpy()
naive_obs = y[t1].mean() - y[~t1].mean()
Xr = np.column_stack([np.ones(len(obs)), obs["chemo"], X])
reg = np.linalg.lstsq(Xr, y, rcond=None)[0][1]

# 5. Escore de propensão e ponderação pelo inverso da probabilidade ---------
ps = LogisticRegression(penalty=None, max_iter=5000).fit(X, obs["chemo"]).predict_proba(X)[:, 1]
w = np.where(t1, 1 / ps, 1 / (1 - ps))
ipw = np.average(y[t1], weights=w[t1]) - np.average(y[~t1], weights=w[~t1])

# 6. Pareamento guloso 1:1 pelo vizinho mais próximo no logito do escore ----
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

# 7. Equilíbrio: diferenças médias padronizadas antes e depois da ponderação
def smd(x, w=None):
    w = np.ones(len(x)) if w is None else w
    num = np.average(x[t1], weights=w[t1]) - np.average(x[~t1], weights=w[~t1])
    return num / np.sqrt((x[t1].var(ddof=1) + x[~t1].var(ddof=1)) / 2)
covs = {"age": obs["patient_age"].to_numpy(), "G3": (obs["tumor_grade"] == "G3").to_numpy(float),
        "performance": obs["performance_score"].to_numpy(float), "gene_05": obs["gene_05"].to_numpy()}
balance = pd.DataFrame({"before": {k: smd(v) for k, v in covs.items()},
                        "after_ipw": {k: smd(v, w) for k, v in covs.items()}}).round(3)
print(balance)
