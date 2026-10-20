# Laboratório do capítulo 8
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Aprendizado conjugado da taxa de LRA do ICU_D ------------------------------------
import numpy as np, pandas as pd
from scipy.stats import beta, norm
icu = pd.read_csv("data/icu_aki.csv")
akiD = (icu.loc[icu["site"] == "ICU_D", "delta_creatinine_48h"] >= 0.3).to_numpy()
a0, b0 = 55, 45                                       # a priori vinda dos outros hospitais
for n in [0, 10, 50, 313]:
    k = int(akiD[:n].sum())
    lo, hi = beta.ppf([0.025, 0.975], a0 + k, b0 + n - k)
    print(f"n = {n:3d}  k = {k:3d}  média a posteriori {(a0 + k) / (a0 + b0 + n):.3f}  ICr 95% {lo:.3f}-{hi:.3f}")

# 2. Distribuição a posteriori em grade da inclinação da fragilidade (intercepto e sigma fixos) -
hb = pd.read_csv("data/homebeat.csv")
x = (hb["frailty"] - hb["frailty"].mean()).to_numpy(); y = hb["resting_hr"].to_numpy()
grid = np.arange(20, 70.0001, 0.05)
logpost = np.array([norm.logpdf(y, y.mean() + b1 * x, 11.2).sum() for b1 in grid]) + norm.logpdf(grid, 0, 20)
post = np.exp(logpost - logpost.max()); post /= post.sum()
print("média a posteriori:", round(float((grid * post).sum()), 2), " mínimos quadrados:", round(float(np.polyfit(x, y, 1)[0]), 2))

# 3. Amostras do Stan: resumos e a lição da confusão -------------------------------
simple = pd.read_csv("data/precomputed/ch08/homebeat_simple_draws.csv")
adjusted = pd.read_csv("data/precomputed/ch08/homebeat_adjusted_draws.csv")
print(simple.groupby("chain")["b_frailty"].mean().round(2))          # as duas cadeias concordam?
for name, d in [("simples", simple), ("ajustado", adjusted)]:
    q = np.quantile(d["b_frailty"], [0.025, 0.5, 0.975]).round(2)
    print(name, q, "P(inclinação > 0) =", round(float((d["b_frailty"] > 0).mean()), 3))

# 4. Lena: intervalo de credibilidade da média versus intervalo de predição -----------------
mu = adjusted["b0"] + adjusted["b_frailty"] * 0.55 + adjusted["b_age"] * 72 + adjusted["b_infl"] * 4.1
y_new = np.random.default_rng(8).normal(loc=mu, scale=adjusted["sigma"])
print("média de pessoas como a Lena:", np.quantile(mu, [0.025, 0.975]).round(1))
print("próxima leitura da Lena:     ", np.quantile(y_new, [0.025, 0.975]).round(1))
print("P(leitura > 88):", round(float((y_new > 88).mean()), 3))

# 5. Esta noite na UTI Ala 7: risco de LRA com incerteza --------------------------------
risk = pd.read_csv("data/precomputed/ch08/ward7_tonight_risk_draws.csv")
for t in [0.25, 0.09]:
    print(f"limiar {t}:")
    print(pd.DataFrame({"média": risk.mean(), "inf": risk.quantile(0.025), "sup": risk.quantile(0.975),
                        "p_acima_de_t": (risk > t).mean()}).round(3))

# 6. BART versus florestas e boosting (erros honestos no conjunto de teste) ---------------------
# O BART e os outros modelos foram ajustados uma vez, fora do navegador, nas divisões comuns;
# os seus erros de teste são lidos do registro de modelos do livro.
hbl = pd.read_csv("data/precomputed/ledger/homebeat.csv")
print(hbl.loc[hbl["role"].isin(["reference", "canonical"]), ["model", "test_MAE", "test_RMSE", "test_R2"]].round(2))
w7 = pd.read_csv("data/precomputed/ledger/ward7_reg.csv")
print(w7.loc[(w7["chapter"] <= 8) & w7["role"].isin(["reference", "canonical", "teaching"]),
             ["chapter", "model", "test_MAE", "test_RMSE"]].round(3))
print(pd.read_csv("data/precomputed/ch08/bart_comparison.csv")[["data", "model", "train_RMSE", "coverage95"]])
