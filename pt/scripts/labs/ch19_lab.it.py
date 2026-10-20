# Laboratorio del capitolo 19
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Riparametrizzazione e penalità KL ----------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def kl_normal(mu, sigma):
    return 0.5 * (mu**2 + sigma**2 - np.log(sigma**2) - 1)
print(kl_normal(0, 1), kl_normal(2, 1), kl_normal(0, 0.1))
rng = np.random.default_rng(19)
z = 1.2 + 0.5 * rng.standard_normal(10000)          # z = mu + sigma * eps
print("media", z.mean().round(2), "ds", z.std(ddof=1).round(2))

# 2. Diffusione diretta su una radiografia ------------------------------------
px = pd.read_csv("data/pneumonia_subset.csv")
x0 = px.loc[px["label"] == 1, px.columns.str.startswith("px")].iloc[0].to_numpy() / 127.5 - 1
beta = np.linspace(1e-4, 0.02, 1000)
abar = np.cumprod(1 - beta)
print("primo passo con più rumore che segnale:", int(np.argmax(abar < 0.5)) + 1)
eps = rng.standard_normal(x0.size)
fig, axes = plt.subplots(1, 6, figsize=(12, 2.4))
for ax, t in zip(axes, [0, 50, 100, 250, 500, 1000]):
    a = 1.0 if t == 0 else abar[t - 1]
    xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * eps
    ax.imshow(np.clip(xt, -1, 1).reshape(28, 28), cmap="gray", vmin=-1, vmax=1)
    ax.set_title(f"t = {t}"); ax.axis("off")
plt.show()

# 3. Una copula gaussiana per HomeBeat, da zero -------------------------------
hb = pd.read_csv("data/homebeat.csv")
vars_ = ["age", "baseline_inflammation", "frailty", "resting_hr", "hrv",
         "skin_temp", "activity_index", "stress_score"]
spl = pd.read_csv("data/splits/homebeat_split.csv").sort_values("perm_order")  # la divisione fissa di HomeBeat usata nel libro (come nel Capitolo 8)
hbi = hb.set_index("participant_id")
real = hbi.loc[spl.loc[spl["set"] == "train", "participant_id"], vars_].reset_index(drop=True)  # il generatore vede questi
hold = hbi.loc[spl.loc[spl["set"] == "test", "participant_id"], vars_].reset_index(drop=True)   # persone reali nuove, mai viste
n = len(real)
scores = norm.ppf(real.rank().to_numpy() / (n + 1))        # passo 1
R = np.corrcoef(scores, rowvar=False)                      # passo 2
Z = rng.standard_normal((400, len(vars_))) @ np.linalg.cholesky(R).T   # passo 3
U = norm.cdf(Z)                                            # passo 4
fake = pd.DataFrame({v: np.quantile(real[v], U[:, j]) for j, v in enumerate(vars_)})

# 4. Fedeltà: marginali e relazioni -------------------------------------------
print(pd.DataFrame({"real": real.mean(), "synthetic": fake.mean()}).round(2))
print("r(FC, HRV): reale", round(real["resting_hr"].corr(real["hrv"]), 2),
      "sintetico", round(fake["resting_hr"].corr(fake["hrv"]), 2))
gap = fake.corr().to_numpy() - real.corr().to_numpy()
print("scarto medio di correlazione:", round(np.abs(gap[np.triu_indices(len(vars_), 1)]).mean(), 3))

# 5. Privacy: distanza dal record reale di addestramento più vicino -----------
mu, s = real.mean().to_numpy(), real.std().to_numpy()
def dcr(a, b):
    A, B = (a.to_numpy() - mu) / s, (b.to_numpy() - mu) / s
    return np.sqrt(((A[:, None, :] - B[None, :, :])**2).sum(2)).min(1)
d_hold, d_fake = dcr(hold, real), dcr(fake, real)
thr = np.quantile(d_hold, 0.05)
print({"median_new_real": round(float(np.median(d_hold)), 2),
       "median_synthetic": round(float(np.median(d_fake)), 2),
       "share_too_close": round(float(np.mean(d_fake < thr)), 3)})

# 6. Utilità: addestrare sul sintetico, testare sul reale ---------------------
xcols = [v for v in vars_ if v != "stress_score"]
def rmse(train):
    X = np.column_stack([np.ones(len(train)), train[xcols]])
    b = np.linalg.lstsq(X, train["stress_score"], rcond=None)[0]
    Xh = np.column_stack([np.ones(len(hold)), hold[xcols]])
    return float(np.sqrt(np.mean((hold["stress_score"] - Xh @ b)**2)))
print({"train_on_real": round(rmse(real), 2), "train_on_synthetic": round(rmse(fake), 2)})
