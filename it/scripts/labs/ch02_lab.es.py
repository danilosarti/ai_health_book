# Laboratorio del capítulo 2
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. Cargar los datos ---------------------------------------------------------
import pandas as pd
import numpy as np
from math import comb
from statistics import NormalDist
icu = pd.read_csv("data/icu_aki.csv")
hb = pd.read_csv("data/homebeat.csv")
icu["aki"] = icu["delta_creatinine_48h"] >= 0.3   # criterio KDIGO: un desenlace de Bernoulli

# 2. Bernoulli y binomial ---------------------------------------------------
theta = icu["aki"].mean()                          # EMV de la probabilidad de LRA
print("theta:", round(theta, 3))
binom = [comb(20, k) * theta**k * (1 - theta)**(20 - k) for k in range(21)]
print("P(15 o más de 20):", round(sum(binom[15:]), 4))

# 3. Modelo normal para la frecuencia cardíaca en reposo --------------------
mu, s = hb["resting_hr"].mean(), hb["resting_hr"].std()
print("P(FC > 80): modelo", round(1 - NormalDist(mu, s).cdf(80), 3),
      "datos", round((hb["resting_hr"] > 80).mean(), 3))

# 4. Verosimilitud y EMV en una rejilla -------------------------------------
x = icu["aki"].to_numpy()
theta_grid = np.arange(0.01, 0.99, 0.001)
loglik = np.array([np.sum(x * np.log(t) + (1 - x) * np.log(1 - t)) for t in theta_grid])
print("EMV por búsqueda en rejilla:", round(theta_grid[np.argmax(loglik)], 3))

# 5. Bayes contando: fragilidad dada una FC alta (HomeBeat) -----------------
frail = hb["frailty"] >= 0.5
fast = hb["resting_hr"] > 85
print(pd.crosstab(frail, fast))
pF, a, b = frail.mean(), fast[frail].mean(), fast[~frail].mean()
print("contada:", round(frail[fast].mean(), 3), " Bayes:", round(a * pF / (a * pF + b * (1 - pF)), 3))

# 6. Una prueba diagnóstica para los pacientes de esta noche ----------------
se, sp = 0.85, 0.80
prior = pd.Series({"Tomas": 0.07, "Aoife": 0.49, "Priya": 0.99})   # riesgos del modelo del Capítulo 5
p_pos = prior * se + (1 - prior) * (1 - sp)
post_pos = prior * se / p_pos
post_neg = prior * (1 - se) / (1 - p_pos)

# 7. Pérdida esperada, umbral y valor de la información ---------------------
C_FP, C_FN = 10, 30
t_star = C_FP / (C_FP + C_FN)
def best(q):
    return np.minimum((1 - q) * C_FP, q * C_FN)    # pérdida esperada de la mejor acción
loss_no_test = best(prior)
loss_with_test = p_pos * best(post_pos) + (1 - p_pos) * best(post_neg)

# 8. Entropía ----------------------------------------------------------------
H = -prior * np.log2(prior) - (1 - prior) * np.log2(1 - prior)
out = pd.DataFrame({"prior": prior, "post_pos": post_pos, "post_neg": post_neg,
                    "loss_no_test": loss_no_test, "loss_with_test": loss_with_test,
                    "value_of_test": loss_no_test - loss_with_test, "entropy": H})
print(out.round(3))
