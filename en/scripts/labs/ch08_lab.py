# Chapter 8 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Conjugate learning of the ICU_D AKI rate ------------------------------------
import numpy as np, pandas as pd
from scipy.stats import beta, norm
icu = pd.read_csv("data/icu_aki.csv")
akiD = (icu.loc[icu["site"] == "ICU_D", "delta_creatinine_48h"] >= 0.3).to_numpy()
a0, b0 = 55, 45                                       # prior from the other hospitals
for n in [0, 10, 50, 313]:
    k = int(akiD[:n].sum())
    lo, hi = beta.ppf([0.025, 0.975], a0 + k, b0 + n - k)
    print(f"n = {n:3d}  k = {k:3d}  posterior mean {(a0 + k) / (a0 + b0 + n):.3f}  95% CrI {lo:.3f}-{hi:.3f}")

# 2. Grid posterior of the HomeBeat frailty slope (intercept and sigma held fixed) -
hb = pd.read_csv("data/homebeat.csv")
x = (hb["frailty"] - hb["frailty"].mean()).to_numpy(); y = hb["resting_hr"].to_numpy()
grid = np.arange(20, 70.0001, 0.05)
logpost = np.array([norm.logpdf(y, y.mean() + b1 * x, 11.2).sum() for b1 in grid]) + norm.logpdf(grid, 0, 20)
post = np.exp(logpost - logpost.max()); post /= post.sum()
print("posterior mean:", round(float((grid * post).sum()), 2), " least squares:", round(float(np.polyfit(x, y, 1)[0]), 2))

# 3. Stan draws: summaries and the confounding lesson -------------------------------
simple = pd.read_csv("data/precomputed/ch08/homebeat_simple_draws.csv")
adjusted = pd.read_csv("data/precomputed/ch08/homebeat_adjusted_draws.csv")
print(simple.groupby("chain")["b_frailty"].mean().round(2))          # do the two chains agree?
for name, d in [("simple", simple), ("adjusted", adjusted)]:
    q = np.quantile(d["b_frailty"], [0.025, 0.5, 0.975]).round(2)
    print(name, q, "P(slope > 0) =", round(float((d["b_frailty"] > 0).mean()), 3))

# 4. Lena: credible interval of the mean versus prediction interval -----------------
mu = adjusted["b0"] + adjusted["b_frailty"] * 0.55 + adjusted["b_age"] * 72 + adjusted["b_infl"] * 4.1
y_new = np.random.default_rng(8).normal(loc=mu, scale=adjusted["sigma"])
print("average of people like Lena:", np.quantile(mu, [0.025, 0.975]).round(1))
print("Lena's next reading:        ", np.quantile(y_new, [0.025, 0.975]).round(1))
print("P(reading > 88):", round(float((y_new > 88).mean()), 3))

# 5. Tonight on Ward 7: risk of AKI with uncertainty --------------------------------
risk = pd.read_csv("data/precomputed/ch08/ward7_tonight_risk_draws.csv")
for t in [0.25, 0.09]:
    print(f"threshold {t}:")
    print(pd.DataFrame({"mean": risk.mean(), "lo": risk.quantile(0.025), "hi": risk.quantile(0.975),
                        "p_above_t": (risk > t).mean()}).round(3))

# 6. BART versus forests and boosting (honest test-set errors) ---------------------
# BART and the other models were fitted once, outside the browser, on the shared splits;
# their test errors are read from the book's model ledger.
hbl = pd.read_csv("data/precomputed/ledger/homebeat.csv")
print(hbl.loc[hbl["role"].isin(["reference", "canonical"]), ["model", "test_MAE", "test_RMSE", "test_R2"]].round(2))
w7 = pd.read_csv("data/precomputed/ledger/ward7_reg.csv")
print(w7.loc[(w7["chapter"] <= 8) & w7["role"].isin(["reference", "canonical", "teaching"]),
             ["chapter", "model", "test_MAE", "test_RMSE"]].round(3))
print(pd.read_csv("data/precomputed/ch08/bart_comparison.csv")[["data", "model", "train_RMSE", "coverage95"]])
