# Laboratorio del capitolo 8
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Apprendimento coniugato del tasso di AKI di ICU_D ------------------------------------
import numpy as np, pandas as pd
from scipy.stats import beta, norm
icu = pd.read_csv("data/icu_aki.csv")
akiD = (icu.loc[icu["site"] == "ICU_D", "delta_creatinine_48h"] >= 0.3).to_numpy()
a0, b0 = 55, 45                                       # a priori dagli altri ospedali
for n in [0, 10, 50, 313]:
    k = int(akiD[:n].sum())
    lo, hi = beta.ppf([0.025, 0.975], a0 + k, b0 + n - k)
    print(f"n = {n:3d}  k = {k:3d}  media a posteriori {(a0 + k) / (a0 + b0 + n):.3f}  ICr 95% {lo:.3f}-{hi:.3f}")

# 2. Distribuzione a posteriori su griglia della pendenza della fragilità (intercetta e sigma fisse) -
hb = pd.read_csv("data/homebeat.csv")
x = (hb["frailty"] - hb["frailty"].mean()).to_numpy(); y = hb["resting_hr"].to_numpy()
grid = np.arange(20, 70.0001, 0.05)
logpost = np.array([norm.logpdf(y, y.mean() + b1 * x, 11.2).sum() for b1 in grid]) + norm.logpdf(grid, 0, 20)
post = np.exp(logpost - logpost.max()); post /= post.sum()
print("media a posteriori:", round(float((grid * post).sum()), 2), " minimi quadrati:", round(float(np.polyfit(x, y, 1)[0]), 2))

# 3. Estrazioni da Stan: sintesi e la lezione sul confondimento -------------------------------
simple = pd.read_csv("data/precomputed/ch08/homebeat_simple_draws.csv")
adjusted = pd.read_csv("data/precomputed/ch08/homebeat_adjusted_draws.csv")
print(simple.groupby("chain")["b_frailty"].mean().round(2))          # le due catene sono d'accordo?
for name, d in [("semplice", simple), ("aggiustato", adjusted)]:
    q = np.quantile(d["b_frailty"], [0.025, 0.5, 0.975]).round(2)
    print(name, q, "P(pendenza > 0) =", round(float((d["b_frailty"] > 0).mean()), 3))

# 4. Lena: intervallo di credibilità della media contro intervallo di previsione -----------------
mu = adjusted["b0"] + adjusted["b_frailty"] * 0.55 + adjusted["b_age"] * 72 + adjusted["b_infl"] * 4.1
y_new = np.random.default_rng(8).normal(loc=mu, scale=adjusted["sigma"])
print("media delle persone come Lena:", np.quantile(mu, [0.025, 0.975]).round(1))
print("prossima misura di Lena:      ", np.quantile(y_new, [0.025, 0.975]).round(1))
print("P(misura > 88):", round(float((y_new > 88).mean()), 3))

# 5. Stasera nel Reparto 7: rischio di AKI con incertezza --------------------------------
risk = pd.read_csv("data/precomputed/ch08/ward7_tonight_risk_draws.csv")
for t in [0.25, 0.09]:
    print(f"soglia {t}:")
    print(pd.DataFrame({"media": risk.mean(), "inf": risk.quantile(0.025), "sup": risk.quantile(0.975),
                        "p_sopra_t": (risk > t).mean()}).round(3))

# 6. BART contro foreste e boosting (errori onesti sull'insieme di test) ---------------------
# Il BART e gli altri modelli sono stati stimati una volta, fuori dal browser, sulle suddivisioni comuni;
# i loro errori di test si leggono dal registro dei modelli del libro.
hbl = pd.read_csv("data/precomputed/ledger/homebeat.csv")
print(hbl.loc[hbl["role"].isin(["reference", "canonical"]), ["model", "test_MAE", "test_RMSE", "test_R2"]].round(2))
w7 = pd.read_csv("data/precomputed/ledger/ward7_reg.csv")
print(w7.loc[(w7["chapter"] <= 8) & w7["role"].isin(["reference", "canonical", "teaching"]),
             ["chapter", "model", "test_MAE", "test_RMSE"]].round(3))
print(pd.read_csv("data/precomputed/ch08/bart_comparison.csv")[["data", "model", "train_RMSE", "coverage95"]])
