# Laboratorio del capítulo 8
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. Aprendizaje conjugado de la tasa de LRA de ICU_D ------------------------------------
import numpy as np, pandas as pd
from scipy.stats import beta, norm
icu = pd.read_csv("data/icu_aki.csv")
akiD = (icu.loc[icu["site"] == "ICU_D", "delta_creatinine_48h"] >= 0.3).to_numpy()
a0, b0 = 55, 45                                       # a priori tomada de los otros hospitales
for n in [0, 10, 50, 313]:
    k = int(akiD[:n].sum())
    lo, hi = beta.ppf([0.025, 0.975], a0 + k, b0 + n - k)
    print(f"n = {n:3d}  k = {k:3d}  media a posteriori {(a0 + k) / (a0 + b0 + n):.3f}  ICr 95% {lo:.3f}-{hi:.3f}")

# 2. Distribución a posteriori en rejilla de la pendiente de la fragilidad (intercepto y sigma fijos) -
hb = pd.read_csv("data/homebeat.csv")
x = (hb["frailty"] - hb["frailty"].mean()).to_numpy(); y = hb["resting_hr"].to_numpy()
grid = np.arange(20, 70.0001, 0.05)
logpost = np.array([norm.logpdf(y, y.mean() + b1 * x, 11.2).sum() for b1 in grid]) + norm.logpdf(grid, 0, 20)
post = np.exp(logpost - logpost.max()); post /= post.sum()
print("media a posteriori:", round(float((grid * post).sum()), 2), " mínimos cuadrados:", round(float(np.polyfit(x, y, 1)[0]), 2))

# 3. Muestras de Stan: resúmenes y la lección de la confusión -------------------------------
simple = pd.read_csv("data/precomputed/ch08/homebeat_simple_draws.csv")
adjusted = pd.read_csv("data/precomputed/ch08/homebeat_adjusted_draws.csv")
print(simple.groupby("chain")["b_frailty"].mean().round(2))          # ¿coinciden las dos cadenas?
for name, d in [("simple", simple), ("ajustado", adjusted)]:
    q = np.quantile(d["b_frailty"], [0.025, 0.5, 0.975]).round(2)
    print(name, q, "P(pendiente > 0) =", round(float((d["b_frailty"] > 0).mean()), 3))

# 4. Lena: intervalo de credibilidad de la media frente a intervalo de predicción -----------------
mu = adjusted["b0"] + adjusted["b_frailty"] * 0.55 + adjusted["b_age"] * 72 + adjusted["b_infl"] * 4.1
y_new = np.random.default_rng(8).normal(loc=mu, scale=adjusted["sigma"])
print("media de las personas como Lena:", np.quantile(mu, [0.025, 0.975]).round(1))
print("próxima lectura de Lena:        ", np.quantile(y_new, [0.025, 0.975]).round(1))
print("P(lectura > 88):", round(float((y_new > 88).mean()), 3))

# 5. Esta noche en la UCI Sala 7: riesgo de LRA con incertidumbre --------------------------------
risk = pd.read_csv("data/precomputed/ch08/ward7_tonight_risk_draws.csv")
for t in [0.25, 0.09]:
    print(f"umbral {t}:")
    print(pd.DataFrame({"media": risk.mean(), "inf": risk.quantile(0.025), "sup": risk.quantile(0.975),
                        "p_por_encima_de_t": (risk > t).mean()}).round(3))

# 6. BART frente a bosques y boosting (errores honestos en el conjunto de prueba) ---------------------
# BART y los demás modelos se ajustaron una vez, fuera del navegador, en las particiones comunes;
# sus errores de prueba se leen del registro de modelos del libro.
hbl = pd.read_csv("data/precomputed/ledger/homebeat.csv")
print(hbl.loc[hbl["role"].isin(["reference", "canonical"]), ["model", "test_MAE", "test_RMSE", "test_R2"]].round(2))
w7 = pd.read_csv("data/precomputed/ledger/ward7_reg.csv")
print(w7.loc[(w7["chapter"] <= 8) & w7["role"].isin(["reference", "canonical", "teaching"]),
             ["chapter", "model", "test_MAE", "test_RMSE"]].round(3))
print(pd.read_csv("data/precomputed/ch08/bart_comparison.csv")[["data", "model", "train_RMSE", "coverage95"]])
