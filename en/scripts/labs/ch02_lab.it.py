# Laboratorio del capitolo 2
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Caricare i dati --------------------------------------------------------
import pandas as pd
import numpy as np
from math import comb
from statistics import NormalDist
icu = pd.read_csv("data/icu_aki.csv")
hb = pd.read_csv("data/homebeat.csv")
icu["aki"] = icu["delta_creatinine_48h"] >= 0.3   # criterio KDIGO: un esito di Bernoulli

# 2. Bernoulli e binomiale --------------------------------------------------
theta = icu["aki"].mean()                          # stima di massima verosimiglianza della probabilità di AKI
print("theta:", round(theta, 3))
binom = [comb(20, k) * theta**k * (1 - theta)**(20 - k) for k in range(21)]
print("P(15 o più su 20):", round(sum(binom[15:]), 4))

# 3. Modello normale per la frequenza cardiaca a riposo ---------------------
mu, s = hb["resting_hr"].mean(), hb["resting_hr"].std()
print("P(FC > 80): modello", round(1 - NormalDist(mu, s).cdf(80), 3),
      "dati", round((hb["resting_hr"] > 80).mean(), 3))

# 4. Verosimiglianza e stima di massima verosimiglianza su una griglia ------
x = icu["aki"].to_numpy()
theta_grid = np.arange(0.01, 0.99, 0.001)
loglik = np.array([np.sum(x * np.log(t) + (1 - x) * np.log(1 - t)) for t in theta_grid])
print("Stima di massima verosimiglianza con ricerca su griglia:", round(theta_grid[np.argmax(loglik)], 3))

# 5. Bayes contando: fragilità data una FC elevata (HomeBeat) ---------------
frail = hb["frailty"] >= 0.5
fast = hb["resting_hr"] > 85
print(pd.crosstab(frail, fast))
pF, a, b = frail.mean(), fast[frail].mean(), fast[~frail].mean()
print("contando:", round(frail[fast].mean(), 3), " Bayes:", round(a * pF / (a * pF + b * (1 - pF)), 3))

# 6. Un test diagnostico per i pazienti di stasera --------------------------
se, sp = 0.85, 0.80
prior = pd.Series({"Tomas": 0.07, "Aoife": 0.49, "Priya": 0.99})   # rischi del modello del Capitolo 5
p_pos = prior * se + (1 - prior) * (1 - sp)
post_pos = prior * se / p_pos
post_neg = prior * (1 - se) / (1 - p_pos)

# 7. Perdita attesa, soglia e valore dell'informazione ----------------------
C_FP, C_FN = 10, 30
t_star = C_FP / (C_FP + C_FN)
def best(q):
    return np.minimum((1 - q) * C_FP, q * C_FN)    # perdita attesa dell'azione migliore
loss_no_test = best(prior)
loss_with_test = p_pos * best(post_pos) + (1 - p_pos) * best(post_neg)

# 8. Entropia ----------------------------------------------------------------
H = -prior * np.log2(prior) - (1 - prior) * np.log2(1 - prior)
out = pd.DataFrame({"prior": prior, "post_pos": post_pos, "post_neg": post_neg,
                    "loss_no_test": loss_no_test, "loss_with_test": loss_with_test,
                    "value_of_test": loss_no_test - loss_with_test, "entropy": H})
print(out.round(3))
