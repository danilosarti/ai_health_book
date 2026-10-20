# Laboratório do capítulo 4
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. UTI Ala 7: carregar e dividir (a divisão compartilhada por todo o livro) -----
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

icu = pd.read_csv("data/icu_aki.csv")
tonight = pd.read_csv("data/ward7_tonight.csv")
split_w7 = pd.read_csv("data/splits/ward7_split.csv").set_index("patient_id")
set_w7 = split_w7.loc[icu["patient_id"], "set"].to_numpy()
train, test = icu[set_w7 == "train"], icu[set_w7 == "test"]

# 2. Regressão linear múltipla (variáveis indicadoras com ICU_A como referência) --
num = ["age", "sofa", "lactate", "map_min", "baseline_creat",
       "protocol_adherence", "bun", "crp", "albumin"]
def design(df):
    site = pd.Categorical(df["site"], categories=["ICU_A", "ICU_B", "ICU_C", "ICU_D"])
    dummies = pd.get_dummies(site, prefix="site", drop_first=True).astype(float)
    dummies.index = df.index
    return pd.concat([dummies, df[num]], axis=1)
ols = LinearRegression().fit(design(train), train["delta_creatinine_48h"])
print(pd.Series(ols.coef_, index=design(train).columns).round(4))

# 3. Diagnóstico: alavancagem e distância de Cook ---------------------------------
X = np.column_stack([np.ones(len(train)), design(train)])
y = train["delta_creatinine_48h"].to_numpy()
resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
lev = np.einsum("ij,ji->i", X, np.linalg.pinv(X))
k = X.shape[1]
s2 = resid @ resid / (len(y) - k)
cooks = resid**2 / (k * s2) * lev / (1 - lev)**2
print({"max_cook": round(float(cooks.max()), 4), "flagged": int((cooks > 4 / len(y)).sum())})

# 4. Erro no conjunto de teste comparado com a referência da 'média' -------------
def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))
y_te = test["delta_creatinine_48h"].to_numpy()
pred = ols.predict(design(test))
avg = train["delta_creatinine_48h"].mean()
print({"average": (round(mae(y_te, avg), 3), round(rmse(y_te, avg), 3)),
       "linear": (round(mae(y_te, pred), 3), round(rmse(y_te, pred), 3))})

# 5. Os pacientes desta noite ------------------------------------------------------
print(tonight[["name"]].assign(predicted=ols.predict(design(tonight)).round(2)))

# 6. ONCO-1: OLS, ridge, elastic net e lasso ---------------------------------------
# O scikit-learn chama a intensidade da penalização de 'alpha' e o parâmetro de mistura de 'l1_ratio'.
onco = pd.read_csv("data/onco1.csv")
drop = ["patient_id", "high_response", "baseline_tumor_mm", "post_tumor_mm",
        "gene_08", "gene_14", "gene_19"]
d = onco.drop(columns=drop)
Xo = pd.get_dummies(d.drop(columns="response_percent"),
                    columns=["treatment", "tumor_grade"], drop_first=True).astype(float)
yo = d["response_percent"].to_numpy()
split_o = pd.read_csv("data/splits/onco1_split.csv").set_index("patient_id").loc[onco["patient_id"]]
tr = np.where(split_o["set"].to_numpy() == "train")[0]
te = np.where(split_o["set"].to_numpy() == "test")[0]
fold = split_o["fold"].to_numpy()[tr]               # partições compartilhadas, as mesmas para todos os modelos
cv = [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(1, 6)]
models = {
    "OLS": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 60), cv=cv)),
    "Elastic net": make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=0.8, cv=cv, max_iter=50000)),
    "Lasso": make_pipeline(StandardScaler(), LassoCV(cv=cv, max_iter=50000)),
}
for name, m in models.items():
    m.fit(Xo.iloc[tr], yo[tr])
    print(f"{name:12s} RMSE de teste = {rmse(yo[te], m.predict(Xo.iloc[te])):.2f}")
lasso = models["Lasso"][-1]
kept = pd.Series(lasso.coef_, index=Xo.columns)
print(kept[kept != 0].round(3).sort_values(key=abs, ascending=False))
