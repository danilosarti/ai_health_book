# Laboratório do capítulo 2
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Carregar os dados ------------------------------------------------------
import pandas as pd
import numpy as np
from math import comb
from statistics import NormalDist
icu = pd.read_csv("data/icu_aki.csv")
hb = pd.read_csv("data/homebeat.csv")
icu["aki"] = icu["delta_creatinine_48h"] >= 0.3   # critério KDIGO: um desfecho de Bernoulli

# 2. Bernoulli e binomial ---------------------------------------------------
theta = icu["aki"].mean()                          # EMV da probabilidade de LRA
print("theta:", round(theta, 3))
binom = [comb(20, k) * theta**k * (1 - theta)**(20 - k) for k in range(21)]
print("P(15 ou mais em 20):", round(sum(binom[15:]), 4))

# 3. Modelo normal para a frequência cardíaca de repouso --------------------
mu, s = hb["resting_hr"].mean(), hb["resting_hr"].std()
print("P(FC > 80): modelo", round(1 - NormalDist(mu, s).cdf(80), 3),
      "dados", round((hb["resting_hr"] > 80).mean(), 3))

# 4. Verossimilhança e EMV em uma grade -------------------------------------
x = icu["aki"].to_numpy()
theta_grid = np.arange(0.01, 0.99, 0.001)
loglik = np.array([np.sum(x * np.log(t) + (1 - x) * np.log(1 - t)) for t in theta_grid])
print("EMV por busca em grade:", round(theta_grid[np.argmax(loglik)], 3))

# 5. Bayes por contagem: fragilidade dada uma FC alta (HomeBeat) ------------
frail = hb["frailty"] >= 0.5
fast = hb["resting_hr"] > 85
print(pd.crosstab(frail, fast))
pF, a, b = frail.mean(), fast[frail].mean(), fast[~frail].mean()
print("contada:", round(frail[fast].mean(), 3), " Bayes:", round(a * pF / (a * pF + b * (1 - pF)), 3))

# 6. Um teste diagnóstico para os pacientes desta noite ---------------------
se, sp = 0.85, 0.80
prior = pd.Series({"Tomas": 0.07, "Aoife": 0.49, "Priya": 0.99})   # riscos do modelo do Capítulo 5
p_pos = prior * se + (1 - prior) * (1 - sp)
post_pos = prior * se / p_pos
post_neg = prior * (1 - se) / (1 - p_pos)

# 7. Perda esperada, limiar e valor da informação ---------------------------
C_FP, C_FN = 10, 30
t_star = C_FP / (C_FP + C_FN)
def best(q):
    return np.minimum((1 - q) * C_FP, q * C_FN)    # perda esperada da melhor ação
loss_no_test = best(prior)
loss_with_test = p_pos * best(post_pos) + (1 - p_pos) * best(post_neg)

# 8. Entropia ----------------------------------------------------------------
H = -prior * np.log2(prior) - (1 - prior) * np.log2(1 - prior)
out = pd.DataFrame({"prior": prior, "post_pos": post_pos, "post_neg": post_neg,
                    "loss_no_test": loss_no_test, "loss_with_test": loss_with_test,
                    "value_of_test": loss_no_test - loss_with_test, "entropy": H})
print(out.round(3))
