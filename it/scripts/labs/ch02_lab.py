# Chapter 2 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Load the data ----------------------------------------------------------
import pandas as pd
import numpy as np
from math import comb
from statistics import NormalDist
icu = pd.read_csv("data/icu_aki.csv")
hb = pd.read_csv("data/homebeat.csv")
icu["aki"] = icu["delta_creatinine_48h"] >= 0.3   # KDIGO criterion: a Bernoulli outcome

# 2. Bernoulli and binomial -------------------------------------------------
theta = icu["aki"].mean()                          # MLE of the probability of AKI
print("theta:", round(theta, 3))
binom = [comb(20, k) * theta**k * (1 - theta)**(20 - k) for k in range(21)]
print("P(15 or more of 20):", round(sum(binom[15:]), 4))

# 3. Normal model for resting heart rate ------------------------------------
mu, s = hb["resting_hr"].mean(), hb["resting_hr"].std()
print("P(HR > 80): model", round(1 - NormalDist(mu, s).cdf(80), 3),
      "data", round((hb["resting_hr"] > 80).mean(), 3))

# 4. Likelihood and MLE on a grid -------------------------------------------
x = icu["aki"].to_numpy()
theta_grid = np.arange(0.01, 0.99, 0.001)
loglik = np.array([np.sum(x * np.log(t) + (1 - x) * np.log(1 - t)) for t in theta_grid])
print("MLE by grid search:", round(theta_grid[np.argmax(loglik)], 3))

# 5. Bayes by counting: frailty given a fast heart rate (HomeBeat) ----------
frail = hb["frailty"] >= 0.5
fast = hb["resting_hr"] > 85
print(pd.crosstab(frail, fast))
pF, a, b = frail.mean(), fast[frail].mean(), fast[~frail].mean()
print("counted:", round(frail[fast].mean(), 3), " Bayes:", round(a * pF / (a * pF + b * (1 - pF)), 3))

# 6. A diagnostic test for tonight's patients -------------------------------
se, sp = 0.85, 0.80
prior = pd.Series({"Tomas": 0.07, "Aoife": 0.49, "Priya": 0.99})   # risks from the Chapter 5 model
p_pos = prior * se + (1 - prior) * (1 - sp)
post_pos = prior * se / p_pos
post_neg = prior * (1 - se) / (1 - p_pos)

# 7. Expected loss, threshold and value of information ----------------------
C_FP, C_FN = 10, 30
t_star = C_FP / (C_FP + C_FN)
def best(q):
    return np.minimum((1 - q) * C_FP, q * C_FN)    # expected loss of the best action
loss_no_test = best(prior)
loss_with_test = p_pos * best(post_pos) + (1 - p_pos) * best(post_neg)

# 8. Entropy -----------------------------------------------------------------
H = -prior * np.log2(prior) - (1 - prior) * np.log2(1 - prior)
out = pd.DataFrame({"prior": prior, "post_pos": post_pos, "post_neg": post_neg,
                    "loss_no_test": loss_no_test, "loss_with_test": loss_with_test,
                    "value_of_test": loss_no_test - loss_with_test, "entropy": H})
print(out.round(3))
