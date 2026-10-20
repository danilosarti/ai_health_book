"""Chapter 7 precompute: PneumoniaMNIST models, the 'find the bug' experiments and the ONCO-1
multi-output network. Runs once on a laptop CPU in a few minutes; the book only reads the outputs.

Usage (from the book folder):
    python3 scripts/precompute/ch07_pneumonia_cnn.py [path/to/pneumoniamnist.npz]

If no path is given, the MedMNIST file is downloaded with `pip install medmnist`:
    from medmnist import PneumoniaMNIST; PneumoniaMNIST(split="test", download=True, root=".")

Outputs
    data/pneumonia_subset.csv                         200 training + 200 test images (28 x 28, 0-255)
    data/precomputed/ch07/pneumonia_widget.json       12 test images + learned first-layer filters
    data/precomputed/ch07/cnn_history.csv             loss and accuracy per epoch (train, validation)
    data/precomputed/ch07/test_predictions.csv        predicted probabilities, all 624 test images
    data/precomputed/ch07/metrics.json                test metrics and confusion matrices
    data/precomputed/ch07/bugs.json                   buggy versus corrected pipelines
    data/precomputed/ch07/onco1_multioutput.json      ONCO-1 multi-output network versus baselines
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score

torch.set_num_threads(1)
SEED = 2026
OUT = "data/precomputed/ch07"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()


def seed_all(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)


# ---------------------------------------------------------------- data
path = sys.argv[1] if len(sys.argv) > 1 else "pneumoniamnist.npz"
if not os.path.exists(path):
    from medmnist import PneumoniaMNIST
    PneumoniaMNIST(split="test", download=True, root=".")
    path = "pneumoniamnist.npz"
d = np.load(path)
Xtr_u8, ytr = d["train_images"], d["train_labels"].ravel().astype(np.float32)
Xva_u8, yva = d["val_images"], d["val_labels"].ravel().astype(np.float32)
Xte_u8, yte = d["test_images"], d["test_labels"].ravel().astype(np.float32)
to_t = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(1) / 255.0   # (n, 1, 28, 28) in [0, 1]
Xtr, Xva, Xte = to_t(Xtr_u8), to_t(Xva_u8), to_t(Xte_u8)
Ytr, Yva, Yte = map(torch.tensor, (ytr, yva, yte))
print("train/val/test:", len(ytr), len(yva), len(yte), "prevalence test:", yte.mean())


# ---------------------------------------------------------------- models
class TinyCNN(nn.Module):
    """Two convolution-ReLU-pooling blocks, then a small dense layer and one output neuron."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)     # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)     # 14 -> 7
        x = F.relu(self.fc1(x.flatten(1)))
        return self.out(x).squeeze(1)                   # a logit; sigmoid gives P(pneumonia)


class MLP(nn.Module):
    """No convolutions: the 784 pixels go straight into one hidden layer."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        return self.out(F.relu(self.fc1(x.flatten(1)))).squeeze(1)


class GapCNN(nn.Module):
    """Like ResNet and EfficientNet, ends with global average pooling, so it accepts ANY image size
    without complaint. That is exactly why a size mismatch between training and testing runs silently."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        return self.out(x.mean(dim=(2, 3))).squeeze(1)


def batches(X, Y, bs=64, shuffle=True, gen=None):
    idx = torch.randperm(len(X), generator=gen) if shuffle else torch.arange(len(X))
    for i in range(0, len(X), bs):
        j = idx[i:i + bs]
        yield X[j], Y[j]


def evaluate(model, X, Y):
    model.eval()
    with torch.no_grad():
        logit = torch.cat([model(xb) for xb, _ in batches(X, Y, 256, shuffle=False)])
        p = torch.sigmoid(logit)
        loss = F.binary_cross_entropy(p.clamp(1e-7, 1 - 1e-7), Y).item()
        acc = ((p >= 0.5).float() == Y).float().mean().item()
    return p.numpy(), loss, acc


def train(model, epochs=10, lr=1e-3, transform=lambda x: x, history=None, keep_best=True, loader=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(SEED)
    best, best_state = np.inf, None
    for ep in range(1, epochs + 1):
        model.train()
        it = loader if loader is not None else batches(Xtr, Ytr, gen=g)
        nb = 0
        for xb, yb in it:
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(transform(xb)), yb)
            loss.backward()
            opt.step()
            nb += 1
        _, tl, ta = evaluate(model, transform(Xtr), Ytr)
        _, vl, va = evaluate(model, transform(Xva), Yva)
        if history is not None:
            history.append(dict(epoch=ep, batches=nb, train_loss=round(tl, 4), train_acc=round(ta, 4),
                                val_loss=round(vl, 4), val_acc=round(va, 4)))
        print(f"  epoch {ep:2d} batches {nb:3d} train {tl:.3f}/{ta:.3f} val {vl:.3f}/{va:.3f}")
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    if keep_best and best_state is not None:
        model.load_state_dict(best_state)
    return model


def summary(p, y, thr=0.5):
    pred = (p >= thr).astype(int)
    y = y.astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    auc = float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else 0.5
    return dict(n=int(len(y)), accuracy=round((tp + tn) / len(y), 3),
                sensitivity=round(tp / (tp + fn), 3), specificity=round(tn / (tn + fp), 3),
                auc=round(auc, 3), confusion=dict(tn=tn, fp=fp, fn=fn, tp=tp))


metrics = {"prevalence_test": round(float(yte.mean()), 3), "n_train": len(ytr), "n_val": len(yva),
           "n_test": len(yte)}

# 1. One neuron = logistic regression on the 784 raw pixels -----------------------------------
print("logistic regression on pixels")
lr_pix = LogisticRegression(C=0.05, max_iter=3000)
lr_pix.fit(Xtr.flatten(1).numpy(), ytr)
p_lr = lr_pix.predict_proba(Xte.flatten(1).numpy())[:, 1]
metrics["logistic_pixels"] = summary(p_lr, yte)

# 2. MLP without convolutions ----------------------------------------------------------------
print("MLP")
seed_all()
mlp = train(MLP(), epochs=10)
p_mlp, _, _ = evaluate(mlp, Xte, Yte)
metrics["mlp"] = summary(p_mlp, yte)
metrics["mlp"]["parameters"] = sum(p.numel() for p in mlp.parameters())

# 3. TinyCNN, 10 epochs, keep the epoch with the lowest validation loss (early stopping) -------
print("TinyCNN")
seed_all()
hist = []
cnn = train(TinyCNN(), epochs=10, history=hist)
p_cnn, test_loss, _ = evaluate(cnn, Xte, Yte)
metrics["cnn"] = summary(p_cnn, yte)
metrics["cnn"]["test_loss"] = round(test_loss, 3)
metrics["cnn"]["parameters"] = sum(p.numel() for p in cnn.parameters())
metrics["cnn"]["best_epoch"] = int(min(hist, key=lambda h: h["val_loss"])["epoch"])
metrics["cnn"]["at_threshold_0.3"] = summary(p_cnn, yte, 0.3)
pd.DataFrame(hist).to_csv(f"{OUT}/cnn_history.csv", index=False)

# 4. Find the bug --------------------------------------------------------------------------
bugs = {}
print("bug A: a generator used as the training loader (exhausted after epoch 1)")
seed_all()
gen_once = ((xb, yb) for xb, yb in batches(Xtr, Ytr, gen=torch.Generator().manual_seed(SEED)))
h_gen = []
gap_gen = train(GapCNN(), epochs=10, lr=3e-3, loader=gen_once, history=h_gen, keep_best=False)
p_gen, _, _ = evaluate(gap_gen, Xte, Yte)
seed_all()
h_ok = []
gap_ok = train(GapCNN(), epochs=10, lr=3e-3, history=h_ok, keep_best=False)
p_ok, _, _ = evaluate(gap_ok, Xte, Yte)
bugs["exhausted_generator"] = {"buggy": summary(p_gen, yte), "fixed": summary(p_ok, yte),
                               "batches_per_epoch_buggy": [h["batches"] for h in h_gen],
                               "batches_per_epoch_fixed": [h["batches"] for h in h_ok]}

print("bug B: trained on images enlarged to 56 px, tested at 28 px")
up = lambda x: F.interpolate(x, size=(56, 56), mode="bilinear", align_corners=False)
seed_all()
gap56 = train(GapCNN(), epochs=10, lr=3e-3, transform=up, keep_best=False)
p_56_wrong, _, _ = evaluate(gap56, Xte, Yte)        # BUG: forgot to resize the test images
p_56_right, _, _ = evaluate(gap56, up(Xte), Yte)    # FIX: same preprocessing as in training
bugs["size_mismatch"] = {"buggy": summary(p_56_wrong, yte), "fixed": summary(p_56_right, yte)}
bugs["constant_pneumonia"] = summary(np.ones_like(yte) * 0.99, yte)
with open(f"{OUT}/bugs.json", "w") as f:
    json.dump(bugs, f, indent=1)

with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=1)

pd.DataFrame({"test_index": np.arange(len(yte)), "label": yte.astype(int),
              "p_logistic": p_lr.round(4), "p_mlp": p_mlp.round(4), "p_cnn": p_cnn.round(4)}
             ).to_csv(f"{OUT}/test_predictions.csv", index=False)

# ---------------------------------------------------------------- browser subsets
rng = np.random.default_rng(SEED)
tr_idx = np.sort(rng.choice(len(ytr), 200, replace=False))
te_idx = np.sort(rng.choice(len(yte), 200, replace=False))
cols = [f"px{k:03d}" for k in range(784)]
sub = pd.concat([
    pd.DataFrame({"split": "train", "orig_index": tr_idx, "label": ytr[tr_idx].astype(int)}),
    pd.DataFrame({"split": "test", "orig_index": te_idx, "label": yte[te_idx].astype(int)})],
    ignore_index=True)
pix = np.vstack([Xtr_u8[tr_idx].reshape(-1, 784), Xte_u8[te_idx].reshape(-1, 784)])
sub = pd.concat([sub, pd.DataFrame(pix, columns=cols)], axis=1)
sub.to_csv("data/pneumonia_subset.csv", index=False)

# 12 test images for the convolution widget: 6 normal and 6 pneumonia, with the CNN's probability
wid = []
for lab in (0, 1):
    cand = te_idx[yte[te_idx] == lab][:6]
    for k in cand:
        wid.append({"test_index": int(k), "label": int(lab), "p_cnn": round(float(p_cnn[k]), 3),
                    "pixels": Xte_u8[k].reshape(-1).tolist()})
filters = np.round(cnn.conv1.weight.detach().numpy()[:, 0].astype(float), 3).tolist()   # 8 learned 3 x 3 kernels
with open(f"{OUT}/pneumonia_widget.json", "w") as f:
    json.dump({"images": wid, "learned_filters": filters}, f, separators=(",", ":"))

# ---------------------------------------------------------------- ONCO-1 multi-output network
print("ONCO-1 multi-output")
onco = pd.read_csv("data/onco1.csv")
leaky = ["gene_08", "gene_14", "gene_19"]       # noisy copies of the outcome (see sim_onco1.R)
genes = [c for c in onco if c.startswith("gene_")]
clin = ["dose_intensity", "patient_age", "performance_score"]
base = pd.concat([onco[clin], (onco["treatment"] == "chemo").astype(int).rename("chemo"),
                  pd.get_dummies(onco["tumor_grade"], prefix="grade", drop_first=True).astype(int)], axis=1)
idx = np.random.default_rng(SEED).permutation(len(onco))
n_tr = int(0.7 * len(onco))
tr, te = idx[:n_tr], idx[n_tr:]
y_c = onco["response_percent"].to_numpy(np.float32)
y_b = onco["high_response"].to_numpy(np.float32)


def onco_fit(feats):
    X = pd.concat([base, onco[feats]], axis=1).to_numpy(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8          # scale with TRAINING statistics only
    Xs = (X - mu) / sd
    res = {}
    lin = LinearRegression().fit(Xs[tr], y_c[tr])
    pc = lin.predict(Xs[te])
    log = LogisticRegression(max_iter=5000).fit(Xs[tr], y_b[tr])
    pb = log.predict_proba(Xs[te])[:, 1]
    res["linear_and_logistic"] = dict(mae=round(float(np.mean(np.abs(pc - y_c[te]))), 2),
                                      rmse=round(float(np.sqrt(np.mean((pc - y_c[te]) ** 2))), 2),
                                      accuracy=round(float(np.mean((pb >= 0.5) == y_b[te])), 3),
                                      auc=round(float(roc_auc_score(y_b[te], pb)), 3))
    # multi-output network: shared layers, two heads, early stopping on an inner validation split
    seed_all()
    inner = np.random.default_rng(SEED + 1).permutation(tr)
    fit_i, val_i = inner[:int(0.85 * len(inner))], inner[int(0.85 * len(inner)):]
    T = lambda a: torch.tensor(a)
    shared = nn.Sequential(nn.Linear(Xs.shape[1], 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 16), nn.ReLU())
    head_c, head_b = nn.Linear(16, 1), nn.Linear(16, 1)
    params = list(shared.parameters()) + list(head_c.parameters()) + list(head_b.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    sc = y_c[fit_i].std()

    def loss_fn(i):
        h = shared(T(Xs[i]))
        return (F.mse_loss(head_c(h).squeeze(1), T(y_c[i])) / sc ** 2 +
                F.binary_cross_entropy_with_logits(head_b(h).squeeze(1), T(y_b[i])))
    best, state, wait, ep = np.inf, None, 0, 0
    for ep in range(1, 501):
        shared.train()
        for j in np.array_split(np.random.default_rng(ep).permutation(fit_i), 9):
            opt.zero_grad(); l = loss_fn(j); l.backward(); opt.step()
        shared.eval()
        with torch.no_grad():
            vl = loss_fn(val_i).item()
        if vl < best - 1e-4:
            best, wait = vl, 0
            state = [m.state_dict() for m in (shared, head_c, head_b)]
            state = [{k: v.clone() for k, v in s.items()} for s in state]
        else:
            wait += 1
            if wait >= 20:
                break
    for m, s in zip((shared, head_c, head_b), state):
        m.load_state_dict(s)
    shared.eval()
    with torch.no_grad():
        h = shared(T(Xs[te]))
        pc = head_c(h).squeeze(1).numpy(); pb = torch.sigmoid(head_b(h).squeeze(1)).numpy()
    res["multi_output_network"] = dict(mae=round(float(np.mean(np.abs(pc - y_c[te]))), 2),
                                       rmse=round(float(np.sqrt(np.mean((pc - y_c[te]) ** 2))), 2),
                                       accuracy=round(float(np.mean((pb >= 0.5) == y_b[te])), 3),
                                       auc=round(float(roc_auc_score(y_b[te], pb)), 3),
                                       epochs_run=ep)
    res["n_features"] = int(Xs.shape[1])
    return res


onco_res = {"n_train": int(n_tr), "n_test": int(len(te)),
            "without_leaky_genes": onco_fit([g for g in genes if g not in leaky]),
            "with_leaky_genes": onco_fit(genes)}
with open(f"{OUT}/onco1_multioutput.json", "w") as f:
    json.dump(onco_res, f, indent=1)

print(json.dumps(metrics, indent=1))
print(json.dumps(bugs, indent=1)[:3000])
print(json.dumps(onco_res, indent=1))
print(f"done in {time.time() - t0:.0f} s")
