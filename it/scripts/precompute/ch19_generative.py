"""Chapter 19 precompute: generative models beyond text.

Runs once on a laptop CPU in a few minutes; the book only reads the outputs.

Usage (from the book folder):
    python3 scripts/precompute/ch19_generative.py [path/to/pneumoniamnist.npz]

If no path is given, the MedMNIST file is downloaded with `pip install medmnist`:
    from medmnist import PneumoniaMNIST; PneumoniaMNIST(split="test", download=True, root=".")

Part A - images (PneumoniaMNIST, 28 x 28):
    a variational autoencoder (VAE) and a plain autoencoder (AE), both with a 2-D latent space,
    trained with the same encoder/decoder sizes. A 15 x 15 grid of decoded images over
    the latent plane, the latent codes of the test images, reconstructions, and (VAE only) images
    decoded from random draws z ~ N(0, I). One test image for the diffusion widget.
Part B - tables (HomeBeat): four synthetic-data generators fitted on the 400 training participants
    of data/splits/homebeat_split.csv, with its 100 test participants held out: independent marginals, Gaussian copula, a small tabular VAE, and a
    "copy and jitter" generator that memorises. Fidelity, diversity, utility and privacy metrics.

Outputs
    data/precomputed/ch19/vae_images.json        grids, codes, reconstructions, samples, history
    data/precomputed/ch19/vae_history.csv        VAE and AE losses per epoch
    data/precomputed/ch19/diffusion_image.json   one test X-ray (pixels 0-255) for the noising widget
    data/precomputed/ch19/homebeat_synthetic.csv real (train, holdout) and synthetic records
    data/precomputed/ch19/homebeat_metrics.csv   one row per generator: fidelity, diversity, utility, privacy
    data/precomputed/ch19/homebeat_corr.csv      correlation matrices (long format) per source
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm, ks_2samp

torch.set_num_threads(1)
SEED = 2026
OUT = "data/precomputed/ch19"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def seed_all(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)


def enc64(img):
    """28 x 28 image in [0, 1] -> 784 characters, one of 64 grey levels per pixel."""
    q = np.clip(np.round(np.asarray(img).ravel() * 63), 0, 63).astype(int)
    return "".join(B64[v] for v in q)


# =============================================================== Part A: images
path = sys.argv[1] if len(sys.argv) > 1 else "pneumoniamnist.npz"
if not os.path.exists(path):
    from medmnist import PneumoniaMNIST
    PneumoniaMNIST(split="test", download=True, root=".")
    path = "pneumoniamnist.npz"
d = np.load(path)
Xtr = torch.tensor(d["train_images"], dtype=torch.float32).reshape(-1, 784) / 255.0
Xva = torch.tensor(d["val_images"], dtype=torch.float32).reshape(-1, 784) / 255.0
Xte = torch.tensor(d["test_images"], dtype=torch.float32).reshape(-1, 784) / 255.0
yte = d["test_labels"].ravel().astype(int)
print("train/val/test:", len(Xtr), len(Xva), len(Xte))

H, Z = 400, 2


class AE(nn.Module):
    """Encoder 784 -> 400 -> 2 numbers (or 2 means + 2 log-variances); decoder 2 -> 400 -> 784."""
    def __init__(self, variational):
        super().__init__()
        self.variational = variational
        self.e1 = nn.Linear(784, H)
        self.mu = nn.Linear(H, Z)
        self.lv = nn.Linear(H, Z)
        self.d1 = nn.Linear(Z, H)
        self.d2 = nn.Linear(H, 784)

    def encode(self, x):
        h = F.relu(self.e1(x))
        return self.mu(h), self.lv(h)

    def decode(self, z):
        return torch.sigmoid(self.d2(F.relu(self.d1(z))))

    def forward(self, x):
        mu, lv = self.encode(x)
        if self.variational and self.training:
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)   # reparameterisation trick
        else:
            z = mu
        return self.decode(z), mu, lv


def losses(model, x):
    xhat, mu, lv = model(x)
    rec = F.binary_cross_entropy(xhat, x, reduction="sum") / len(x)
    kl = (-0.5 * torch.sum(1 + lv - mu ** 2 - lv.exp())) / len(x)
    return rec, kl


def train(variational, epochs=30):
    seed_all()
    m = AE(variational)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    hist = []
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(n)
        tot_r = tot_k = 0.0
        for i in range(0, n, 128):
            xb = Xtr[perm[i:i + 128]]
            r, k = losses(m, xb)
            loss = r + (k if variational else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_r += r.item() * len(xb); tot_k += k.item() * len(xb)
        m.eval()
        with torch.no_grad():
            vr, vk = losses(m, Xva)
        hist.append(dict(model="VAE" if variational else "AE", epoch=ep,
                         train_reconstruction=tot_r / n, train_kl=tot_k / n,
                         val_reconstruction=vr.item(), val_kl=vk.item()))
        if ep % 5 == 0:
            print(hist[-1], f"{time.time() - t0:.0f}s")
    return m, hist


vae, h_vae = train(True)
ae, h_ae = train(False)
pd.DataFrame(h_vae + h_ae).round(3).to_csv(f"{OUT}/vae_history.csv", index=False)

G = 15
out = {"grid_size": G, "models": {}}
recon_idx = [int(i) for i in np.r_[np.where(yte == 0)[0][:3], np.where(yte == 1)[0][:3]]]
for name, m in [("VAE", vae), ("AE", ae)]:
    m.eval()
    with torch.no_grad():
        mu, lv = m.encode(Xte)
        mu = mu.numpy()
        if name == "VAE":
            lo, hi = np.array([-2.5, -2.5]), np.array([2.5, 2.5])
        else:
            lo, hi = np.percentile(mu, 1, axis=0), np.percentile(mu, 99, axis=0)
        g1, g2 = np.linspace(lo[0], hi[0], G), np.linspace(lo[1], hi[1], G)
        zz = torch.tensor([[a, b] for b in g2 for a in g1], dtype=torch.float32)   # row-major: z2 outer
        imgs = m.decode(zz).numpy()
        rec = m(Xte[recon_idx])[0].numpy()
        entry = {
            "codes": [{"z1": round(float(a), 3), "z2": round(float(b), 3),
                       "label": "pneumonia" if l == 1 else "normal"} for (a, b), l in zip(mu, yte)],
            "recon": [enc64(im) for im in rec],
        }
        if name == "VAE":                      # the grid is drawn for the VAE only
            entry.update({"z1": np.round(g1, 3).tolist(), "z2": np.round(g2, 3).tolist(),
                          "grid": [enc64(im) for im in imgs]})
            entry["kl_val"] = round(h_vae[-1]["val_kl"], 2)
        # new images from random draws z ~ N(0, I): the recipe of a VAE, no recipe for a plain AE
        g = torch.Generator().manual_seed(SEED)
        zs = torch.randn(16, Z, generator=g)
        entry["samples"] = [enc64(im) for im in m.decode(zs).numpy()]
        # mean absolute pixel error of the reconstructions of all test images (grey levels 0-255)
        entry["pixel_mae"] = round(float(torch.mean(torch.abs(m(Xte)[0] - Xte)).item() * 255), 1)
        entry["code_mean"] = np.round(mu.mean(0), 2).tolist()
        entry["code_sd"] = np.round(mu.std(0), 2).tolist()
        out["models"][name] = entry
out["originals"] = [enc64(Xte[i].numpy()) for i in recon_idx]
out["original_labels"] = ["pneumonia" if yte[i] == 1 else "normal" for i in recon_idx]
with open(f"{OUT}/vae_images.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))
print({k: (v["pixel_mae"], v["code_mean"], v["code_sd"]) for k, v in out["models"].items()})

# one image for the diffusion widget: the first pneumonia image of the test set
k = int(np.where(yte == 1)[0][0])
with open(f"{OUT}/diffusion_image.json", "w") as f:
    json.dump({"label": "pneumonia", "test_index": k,
               "pixels": d["test_images"][k].ravel().astype(int).tolist()}, f, separators=(",", ":"))
print(f"Part A done in {time.time() - t0:.0f}s")

# =============================================================== Part B: HomeBeat
hb = pd.read_csv("data/homebeat.csv")
COLS = ["age", "baseline_inflammation", "frailty", "resting_hr", "hrv", "skin_temp",
        "activity_index", "stress_score", "event_occurred"]
DEC = {"age": 0, "baseline_inflammation": 3, "frailty": 3, "resting_hr": 1, "hrv": 1,
       "skin_temp": 2, "activity_index": 1, "stress_score": 1, "event_occurred": 0}
# The book's fixed HomeBeat split (scripts/simulate/make_homebeat_split.py), shared with Chapter 8
# and the model ledger. It stores np.random.default_rng(2026).permutation(500); rows sorted by
# perm_order reproduce the order this script used before the split file existed, so no number moves.
sph = pd.read_csv("data/splits/homebeat_split.csv").sort_values("perm_order")
hb = hb.set_index("participant_id")
real = hb.loc[sph.loc[sph["set"] == "train", "participant_id"], COLS].reset_index(drop=True)
hold = hb.loc[sph.loc[sph["set"] == "test", "participant_id"], COLS].reset_index(drop=True)
assert len(real) == 400 and len(hold) == 100
R = real.to_numpy(float)
N_SYN = 400
mu_r, sd_r = R.mean(0), R.std(0, ddof=1)


def tidy(A):
    df = pd.DataFrame(A, columns=COLS)
    df["event_occurred"] = (df["event_occurred"] >= 0.5).astype(int)
    for c in COLS:
        df[c] = df[c].round(DEC[c])
    return df


def emp_quantile(col, u):
    """Inverse of the empirical distribution function, interpolated between order statistics."""
    s = np.sort(col)
    pos = u * (len(s) - 1)
    return np.interp(pos, np.arange(len(s)), s)


# 1. independent marginals: each column resampled on its own
rng = np.random.default_rng(SEED + 1)
indep = np.column_stack([rng.choice(R[:, j], N_SYN) for j in range(len(COLS))])

# 2. Gaussian copula: normal scores -> correlation -> multivariate normal -> back through the marginals
ranks = np.argsort(np.argsort(R, axis=0), axis=0) + 1
Zs = norm.ppf(ranks / (len(R) + 1))
C = np.corrcoef(Zs, rowvar=False)
rng = np.random.default_rng(SEED + 2)
Znew = rng.standard_normal((N_SYN, len(COLS))) @ np.linalg.cholesky(C).T
U = norm.cdf(Znew)
cop = np.column_stack([emp_quantile(R[:, j], U[:, j]) for j in range(len(COLS))])


# 3. tabular VAE on standardised columns
class TabVAE(nn.Module):
    def __init__(self, p, h=32, z=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(p, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu, self.lv = nn.Linear(h, z), nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, p))
        self.log_s = nn.Parameter(torch.zeros(p))   # learned noise scale of each column

    def forward(self, x):
        hh = self.enc(x)
        mu, lv = self.mu(hh), self.lv(hh)
        zz = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.dec(zz), mu, lv


seed_all(SEED + 3)
Xs = torch.tensor((R - mu_r) / sd_r, dtype=torch.float32)
tv = TabVAE(len(COLS))
opt = torch.optim.Adam(tv.parameters(), lr=3e-3)
for ep in range(3000):
    xh, m_, l_ = tv(Xs)
    s2 = torch.exp(2 * tv.log_s)
    nll = (0.5 * (Xs - xh) ** 2 / s2 + tv.log_s).sum(1).mean()
    kl = (-0.5 * (1 + l_ - m_ ** 2 - l_.exp()).sum(1)).mean()
    loss = nll + kl
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    zz = torch.randn(N_SYN, 4)
    xs = tv.dec(zz) + torch.exp(tv.log_s) * torch.randn(N_SYN, len(COLS))
vae_tab = xs.numpy() * sd_r + mu_r

# 4. copy and jitter: real rows with 2 % noise - looks perfect, protects nobody
rng = np.random.default_rng(SEED + 4)
copy = R[rng.integers(0, len(R), N_SYN)] + rng.normal(0, 0.02, (N_SYN, len(COLS))) * sd_r

gens = {"independent": tidy(indep), "copula": tidy(cop), "vae": tidy(vae_tab), "copy": tidy(copy)}
allrows = [real.assign(source="real_train"), hold.assign(source="real_holdout")] + \
          [g.assign(source=k) for k, g in gens.items()]
pd.concat(allrows, ignore_index=True).to_csv(f"{OUT}/homebeat_synthetic.csv", index=False)


# ---- metrics
def stdz(A):
    return (np.asarray(A, float) - mu_r) / sd_r


def nn_dist(A, Bm):
    D = np.sqrt(((A[:, None, :] - Bm[None, :, :]) ** 2).sum(2))
    return D


Rs, Hs = stdz(real), stdz(hold)
d_hold = nn_dist(Hs, Rs).min(1)          # how close a NEW real person is to a training person
thr = np.quantile(d_hold, 0.05)
k_nn = 5
Drr = nn_dist(Rs, Rs); np.fill_diagonal(Drr, np.inf)
rad_r = np.sort(Drr, 1)[:, k_nn - 1]      # k-NN radius of each real record


def ols_rmse(train_df, test_df):
    xcols = [c for c in COLS if c not in ("stress_score", "event_occurred")]
    Xa = np.column_stack([np.ones(len(train_df)), train_df[xcols].to_numpy(float)])
    beta = np.linalg.lstsq(Xa, train_df["stress_score"].to_numpy(float), rcond=None)[0]
    Xb = np.column_stack([np.ones(len(test_df)), test_df[xcols].to_numpy(float)])
    return float(np.sqrt(np.mean((test_df["stress_score"].to_numpy(float) - Xb @ beta) ** 2)))


corr_rows = []
for src, df in [("real_train", real), ("real_holdout", hold)] + list(gens.items()):
    Cm = df[COLS].corr().to_numpy()
    for i, a in enumerate(COLS):
        for j, b in enumerate(COLS):
            corr_rows.append(dict(source=src, var1=a, var2=b, r=round(float(Cm[i, j]), 3)))
pd.DataFrame(corr_rows).to_csv(f"{OUT}/homebeat_corr.csv", index=False)

Creal = real[COLS].corr().to_numpy()
iu = np.triu_indices(len(COLS), 1)
rows = []
ref = [("real_holdout", hold)]
for name, df in ref + list(gens.items()):
    S = stdz(df[COLS])
    Dsr = nn_dist(S, Rs)
    dcr = Dsr.min(1)
    Drs = Dsr.T
    precision = float(np.mean((Dsr <= rad_r[None, :]).any(1)))      # synthetic inside the real cloud
    Dss = nn_dist(S, S); np.fill_diagonal(Dss, np.inf)
    rad_s = np.sort(Dss, 1)[:, k_nn - 1]
    recall = float(np.mean((Drs <= rad_s[None, :]).any(1)))          # real covered by the synthetic cloud
    rows.append(dict(
        source=name,
        ks_mean=round(float(np.mean([ks_2samp(df[c], real[c]).statistic for c in COLS])), 3),
        corr_mae=round(float(np.mean(np.abs(df[COLS].corr().to_numpy()[iu] - Creal[iu]))), 3),
        r_hr_hrv=round(float(df["resting_hr"].corr(df["hrv"])), 3),
        precision=round(precision, 3), recall=round(recall, 3),
        tstr_rmse=round(ols_rmse(df, hold) if name != "real_holdout" else ols_rmse(real, hold), 2),
        dcr_median=round(float(np.median(dcr)), 3),
        share_too_close=round(float(np.mean(dcr < thr)), 3)))
met = pd.DataFrame(rows)
met.loc[met.source == "real_holdout", "source"] = "real_holdout"
met.to_csv(f"{OUT}/homebeat_metrics.csv", index=False)
print(met.to_string())
print("holdout 5th percentile DCR:", round(thr, 3))
print(f"done in {time.time() - t0:.0f}s")
