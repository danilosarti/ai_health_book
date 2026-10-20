"""
Chapter 16 — precomputed language models and embeddings.

Trains, once, everything that is too slow for the browser and writes small JSON/CSV files to
data/precomputed/ch16/:

  neural_lm.json      tiny neural (fixed-window) language model trained in numpy on the Ward 7 notes:
                      learning curve, test perplexity, full next-token logits for a few contexts,
                      and the "meropenem" generalisation check
  word_vectors.csv    skip-gram embeddings (numpy, negative sampling) of the Ward 7 notes vocabulary
                      (input + output vectors, not normalised)
  embeddings.json     the same vectors plus a 2-D PCA projection, neighbours, analogies, bias scores
  lstm_word.json      small word-level LSTM (torch): test perplexity and the long-range check
  char_samples.json   character-level LSTM (torch) trained on Tiny Shakespeare: samples and
                      validation loss at several stages of training

Run from the book folder:  python3 scripts/precompute/ch16_language_models.py
Total time on one CPU core: about six minutes (the character model takes four of them; it is
skipped if char_samples.json already exists, unless CH16_FORCE=1). Seeds are fixed.
The character model needs the Tiny Shakespeare text (public domain; about 1 MB, not shipped with
the book): set CH16_SHAKESPEARE to its path.
"""
import csv, json, math, os, time
import numpy as np

OUT = "data/precomputed/ch16"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()

def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)

# ---------------------------------------------------------------- data
rows = list(csv.DictReader(open("data/ward7_notes.csv", encoding="utf-8")))
train_notes = [r["text"].split() for r in rows if r["set"] == "train"]
test_notes = [r["text"].split() for r in rows if r["set"] == "test"]
vocab = ["<s>", "</s>"] + sorted({t for s in train_notes + test_notes for t in s})
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
log(f"{len(train_notes)} train notes, {len(test_notes)} test notes, |V| = {V}")

CTX = 3   # the neural LM sees the previous three tokens

def windows(notes, ctx=CTX):
    X, Y = [], []
    for s in notes:
        seq = ["<s>"] * ctx + s + ["</s>"]
        ids = [idx[w] for w in seq]
        for t in range(ctx, len(ids)):
            X.append(ids[t - ctx:t]); Y.append(ids[t])
    return np.array(X), np.array(Y)

Xtr, Ytr = windows(train_notes)
Xte, Yte = windows(test_notes)
log(f"{len(Ytr)} training and {len(Yte)} test prediction events")

# ---------------------------------------------------------------- 1. tiny neural LM in numpy
# x = [e(w_{t-3}), e(w_{t-2}), e(w_{t-1})] -> h = tanh(W1 x + b1) -> z = W2 h + b2 -> softmax
rng = np.random.default_rng(16)
D, H = 16, 128
P = {"E": rng.normal(0, 0.1, (V, D)),
     "W1": rng.normal(0, 1 / math.sqrt(CTX * D), (CTX * D, H)), "b1": np.zeros(H),
     "W2": rng.normal(0, 1 / math.sqrt(H), (H, V)), "b2": np.zeros(V)}

def forward(P, X):
    x = P["E"][X].reshape(len(X), -1)
    h = np.tanh(x @ P["W1"] + P["b1"])
    z = h @ P["W2"] + P["b2"]
    return x, h, z

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_bits(P, X, Y):
    _, _, z = forward(P, X)
    p = softmax(z)[np.arange(len(Y)), Y]
    return float(-np.mean(np.log2(p)))

adam_m = {k: np.zeros_like(v) for k, v in P.items()}
adam_v = {k: np.zeros_like(v) for k, v in P.items()}
lr, b1m, b2m, eps, step = 0.005, 0.9, 0.999, 1e-8, 0
curve = [{"epoch": 0.0, "train_bits": cross_entropy_bits(P, Xtr, Ytr),
          "test_bits": cross_entropy_bits(P, Xte, Yte)}]
EPOCHS, BATCH = 8, 128
n_batches = math.ceil(len(Ytr) / BATCH)
for epoch in range(1, EPOCHS + 1):
    lr = 0.005 if epoch <= 4 else 0.001 if epoch <= 6 else 0.0003   # smaller steps once the loss levels off
    order = rng.permutation(len(Ytr))
    for start in range(0, len(order), BATCH):
        b = order[start:start + BATCH]
        X, Y = Xtr[b], Ytr[b]
        x, h, z = forward(P, X)
        g = softmax(z); g[np.arange(len(Y)), Y] -= 1; g /= len(Y)       # dLoss/dz
        grads = {"W2": h.T @ g, "b2": g.sum(0)}
        gh = (g @ P["W2"].T) * (1 - h ** 2)
        grads["W1"] = x.T @ gh; grads["b1"] = gh.sum(0)
        gx = (gh @ P["W1"].T).reshape(len(X), CTX, D)
        gE = np.zeros_like(P["E"]); np.add.at(gE, X, gx); grads["E"] = gE
        step += 1
        for k in P:
            adam_m[k] = b1m * adam_m[k] + (1 - b1m) * grads[k]
            adam_v[k] = b2m * adam_v[k] + (1 - b2m) * grads[k] ** 2
            mh = adam_m[k] / (1 - b1m ** step); vh = adam_v[k] / (1 - b2m ** step)
            P[k] -= lr * mh / (np.sqrt(vh) + eps)
        if epoch == 1 and step % 25 == 0:
            curve.append({"epoch": round(step / n_batches, 3), "train_bits": cross_entropy_bits(P, Xtr, Ytr),
                          "test_bits": cross_entropy_bits(P, Xte, Yte)})
    curve.append({"epoch": epoch, "train_bits": cross_entropy_bits(P, Xtr, Ytr),
                  "test_bits": cross_entropy_bits(P, Xte, Yte)})
    log(f"neural LM epoch {epoch}: train {curve[-1]['train_bits']:.3f} bits, test {curve[-1]['test_bits']:.3f} bits")

def next_logits(context_words):
    ctx = (["<s>"] * CTX + context_words)[-CTX:]
    _, _, z = forward(P, np.array([[idx[w] for w in ctx]]))
    return z[0]

CONTEXTS = {
    "patient reports": ["patient", "reports"],
    "started vancomycin for": ["started", "vancomycin", "for"],
    "no evidence of": ["no", "evidence", "of"],
    "creatinine rising ,": ["creatinine", "rising", ","],
    "plan :": ["plan", ":"],
    "the surgeon says": ["the", "surgeon", "says"],
}
contexts_out = []
for label, ws in CONTEXTS.items():
    z = next_logits(ws)
    contexts_out.append({"context": label, "tokens": vocab, "logits": [round(float(v), 3) for v in z]})

# generalisation check: "started meropenem" never occurs in training
p_start = softmax(next_logits(["started"])[None, :])[0]
abx_rank = sorted([(vocab[i], float(p_start[i])) for i in range(V)], key=lambda t: -t[1])
merop_nn = {"p_meropenem": float(p_start[idx["meropenem"]]),
            "rank_meropenem": [w for w, _ in abx_rank].index("meropenem") + 1,
            "p_insulin": float(p_start[idx["insulin"]]),
            "top": [[w, round(p, 4)] for w, p in abx_rank[:12]]}

json.dump({"context_size": CTX, "embedding_dim": D, "hidden": H, "epochs": EPOCHS,
           "batch": BATCH, "learning_rate": lr, "vocab_size": V,
           "n_parameters": int(sum(v.size for v in P.values())),
           "n_train_events": int(len(Ytr)), "n_test_events": int(len(Yte)),
           "curve": curve, "test_bits": curve[-1]["test_bits"],
           "test_perplexity": 2 ** curve[-1]["test_bits"],
           "contexts": contexts_out, "meropenem": merop_nn},
          open(f"{OUT}/neural_lm.json", "w"))
log("wrote neural_lm.json")

# ---------------------------------------------------------------- 2. skip-gram embeddings
PUNCT = {",", ".", ":"}
sg_notes = [[w for w in s if w not in PUNCT] for s in train_notes]
counts = {}
for s in sg_notes:
    for w in s: counts[w] = counts.get(w, 0) + 1
sg_vocab = sorted(counts)
sg_idx = {w: i for i, w in enumerate(sg_vocab)}
Vs = len(sg_vocab)
WIN, DIM, NEG = 2, 24, 5
pairs = []
for s in sg_notes:
    ids = [sg_idx[w] for w in s]
    for i, c in enumerate(ids):
        for j in range(max(0, i - WIN), min(len(ids), i + WIN + 1)):
            if j != i: pairs.append((c, ids[j]))
pairs = np.array(pairs)
freq = np.array([counts[w] for w in sg_vocab], dtype=float)
noise = freq ** 0.75; noise /= noise.sum()
rng = np.random.default_rng(2013)
Win = rng.normal(0, 0.1, (Vs, DIM))
Wout = np.zeros((Vs, DIM))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

SG_EPOCHS, SG_BATCH = 20, 512
for epoch in range(SG_EPOCHS):
    lr_sg = 0.05 * (1 - epoch / SG_EPOCHS) + 0.002
    order = rng.permutation(len(pairs))
    for start in range(0, len(order), SG_BATCH):
        b = pairs[order[start:start + SG_BATCH]]
        c, o = b[:, 0], b[:, 1]
        negs = rng.choice(Vs, size=(len(b), NEG), p=noise)
        vc = Win[c]
        # positive pairs: push sigma(vc . uo) towards 1; negatives towards 0
        uo = Wout[o]; gpos = sigmoid((vc * uo).sum(1)) - 1
        un = Wout[negs]; gneg = sigmoid(np.einsum("bd,bkd->bk", vc, un))
        grad_vc = gpos[:, None] * uo + np.einsum("bk,bkd->bd", gneg, un)
        np.add.at(Wout, o, -lr_sg * gpos[:, None] * vc)
        np.add.at(Wout, negs, -lr_sg * gneg[:, :, None] * vc[:, None, :])
        np.add.at(Win, c, -lr_sg * grad_vc)
log(f"skip-gram: {len(pairs)} pairs, {Vs} words, {SG_EPOCHS} epochs")

Wsum = Win + Wout                     # input + output vectors, as GloVe does
Evec = Wsum / np.linalg.norm(Wsum, axis=1, keepdims=True)

def cat_of(w):
    groups = {
        "antibiotic": ["vancomycin", "gentamicin", "piperacillin", "meropenem", "ceftriaxone"],
        "analgesic": ["paracetamol", "morphine", "oxycodone"],
        "diuretic": ["furosemide", "bumetanide"],
        "vasopressor": ["noradrenaline", "vasopressin"],
        "anticoagulant": ["heparin", "apixaban"],
        "other drug": ["insulin"],
        "symptom": ["nausea", "vomiting", "fatigue", "dizziness", "breathlessness", "fever",
                    "confusion", "headache", "pain"],
        "body site": ["chest", "abdominal", "back", "leg"],
        "lab marker": ["creatinine", "bilirubin", "troponin", "lipase"],
        "organ": ["kidney", "liver", "heart", "pancreas"],
        "specialty": ["nephrology", "hepatology", "cardiology", "gastroenterology"],
        "condition": ["aki", "sepsis", "bleeding", "infection", "delirium", "pneumonia",
                      "hypotension", "shock", "oedema"],
        "staff role": ["nurse", "physiotherapist", "pharmacist", "doctor", "consultant", "surgeon"],
        "pronoun": ["she", "he"],
    }
    for g, ws in groups.items():
        if w in ws: return g
    return "other"

def neighbours(v, k=6, exclude=()):
    s = Evec @ (v / np.linalg.norm(v))
    out = []
    for i in np.argsort(-s):
        if sg_vocab[i] in exclude: continue
        out.append([sg_vocab[i], round(float(s[i]), 3)])
        if len(out) == k: break
    return out

def vec(w):
    return Evec[sg_idx[w]]

nn_examples = {w: neighbours(vec(w), 6, exclude=(w,)) for w in
               ["vancomycin", "meropenem", "morphine", "furosemide", "nausea", "creatinine",
                "kidney", "nurse", "surgeon"]}
ANALOGIES = [("kidney", "creatinine", "liver"), ("kidney", "creatinine", "heart"),
             ("kidney", "nephrology", "liver"), ("creatinine", "nephrology", "troponin"),
             ("vancomycin", "sepsis", "furosemide"), ("morphine", "pain", "noradrenaline")]
analogies = []
for a, b, c in ANALOGIES:
    target = vec(b) - vec(a) + vec(c)
    analogies.append({"a": a, "b": b, "c": c,
                      "answers": neighbours(target, 4, exclude=(a, b, c))})
gdir = vec("she") - vec("he"); gdir = gdir / np.linalg.norm(gdir)
bias = []
for role in ["nurse", "physiotherapist", "pharmacist", "doctor", "consultant", "surgeon"]:
    she, he = float(vec(role) @ vec("she")), float(vec(role) @ vec("he"))
    bias.append({"role": role, "cos_she": round(she, 3), "cos_he": round(he, 3),
                 "she_minus_he": round(she - he, 3),
                 "projection": round(float(vec(role) @ gdir), 3)})
# share of "she" after each role in the training notes (the bias in the data itself)
for bdict in bias:
    n_she = n_he = 0
    for s in train_notes:
        for i in range(len(s) - 2):
            if s[i] == bdict["role"] and s[i + 2] in ("she", "he"):
                n_she += s[i + 2] == "she"; n_he += s[i + 2] == "he"
    bdict["share_she_in_notes"] = round(n_she / max(1, n_she + n_he), 3)
    bdict["n_mentions"] = n_she + n_he
# 2-D projection (PCA of the unit vectors)
C = Evec - Evec.mean(0)
U, S, Vt = np.linalg.svd(C, full_matrices=False)
xy = C @ Vt[:2].T
expl = (S[:2] ** 2 / (S ** 2).sum()).tolist()
words_out = [{"word": w, "cat": cat_of(w), "freq": int(counts[w]),
              "x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4),
              "v": [round(float(t), 4) for t in Wsum[i]]} for i, w in enumerate(sg_vocab)]
json.dump({"dim": DIM, "window": WIN, "negatives": NEG, "epochs": SG_EPOCHS,
           "n_pairs": int(len(pairs)), "pca_explained": expl, "words": words_out,
           "neighbours": nn_examples, "analogies": analogies, "bias": bias,
           "onehot_cos": 0.0}, open(f"{OUT}/embeddings.json", "w"))
with open(f"{OUT}/word_vectors.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["word", "category", "freq"] + [f"d{j + 1}" for j in range(DIM)])
    for i, w in enumerate(sg_vocab):
        wr.writerow([w, cat_of(w), counts[w]] + [f"{t:.4f}" for t in Wsum[i]])
log("wrote embeddings.json and word_vectors.csv")

# ---------------------------------------------------------------- 3. word-level LSTM (torch)
import torch
import torch.nn as nn
torch.set_num_threads(1)
torch.manual_seed(1997)

def as_seq(notes):
    return [[idx["<s>"]] + [idx[w] for w in s] + [idx["</s>"]] for s in notes]

tr_seq, te_seq = as_seq(train_notes), as_seq(test_notes)

def pad_batch(seqs):
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L - 1), 0, dtype=torch.long)
    y = torch.full((len(seqs), L - 1), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s) - 1] = torch.tensor(s[:-1]); y[i, :len(s) - 1] = torch.tensor(s[1:])
    return x, y

class WordLSTM(nn.Module):
    def __init__(self, V, d=32, h=64):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.rnn = nn.LSTM(d, h, batch_first=True)
        self.out = nn.Linear(h, V)
    def forward(self, x):
        o, _ = self.rnn(self.emb(x)); return self.out(o)

wl = WordLSTM(V)
opt = torch.optim.Adam(wl.parameters(), lr=0.005)
lossf = nn.CrossEntropyLoss(ignore_index=-100)

def eval_bits(model, seqs):
    model.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(seqs), 256):
            x, y = pad_batch(seqs[i:i + 256]); z = model(x)
            l = nn.functional.cross_entropy(z.reshape(-1, V), y.reshape(-1), ignore_index=-100, reduction="sum")
            tot += float(l); n += int((y != -100).sum())
    model.train(); return tot / n / math.log(2)

g = torch.Generator().manual_seed(7)
lstm_curve = []
for epoch in range(1, 9):
    perm = torch.randperm(len(tr_seq), generator=g).tolist()
    for i in range(0, len(perm), 64):
        x, y = pad_batch([tr_seq[j] for j in perm[i:i + 64]])
        opt.zero_grad(); l = lossf(wl(x).reshape(-1, V), y.reshape(-1)); l.backward(); opt.step()
    lstm_curve.append({"epoch": epoch, "test_bits": eval_bits(wl, te_seq)})
    log(f"word LSTM epoch {epoch}: test {lstm_curve[-1]['test_bits']:.3f} bits")

def lstm_next(words):
    with torch.no_grad():
        x = torch.tensor([[idx["<s>"]] + [idx[w] for w in words]])
        p = torch.softmax(wl(x)[0, -1], -1).numpy()
    return p

def nn_next(words):
    return softmax(next_logits(words)[None, :])[0]

# long-range check: which drug follows "check" after "started <drug> today , ... ,"?
counts_bi, counts_tri = {}, {}
for s in train_notes:
    seq = ["<s>", "<s>"] + s + ["</s>"]
    for t in range(2, len(seq)):
        counts_bi.setdefault(seq[t - 1], {}).setdefault(seq[t], 0); counts_bi[seq[t - 1]][seq[t]] += 1
        k = (seq[t - 2], seq[t - 1]); counts_tri.setdefault(k, {}).setdefault(seq[t], 0); counts_tri[k][seq[t]] += 1

def mle(table, key, w):
    row = table.get(key, {}); tot = sum(row.values())
    return row.get(w, 0) / tot if tot else float("nan")

long_range = []
for drug in ["gentamicin", "vancomycin"]:
    for filler in [[], ["patient", "reports", "nausea", ","],
                   ["patient", "reports", "nausea", ",", "patient", "denies", "fever", ","]]:
        ctx = ["started", drug, "today", ","] + filler + ["check"]
        other = "vancomycin" if drug == "gentamicin" else "gentamicin"
        pl, pn = lstm_next(ctx), nn_next(ctx)
        long_range.append({
            "context": " ".join(ctx), "drug": drug, "gap_tokens": len(filler) + 2,
            "bigram": round(mle(counts_bi, "check", drug), 3),
            "trigram": round(mle(counts_tri, (",", "check"), drug), 3),
            "neural_window3": round(float(pn[idx[drug]]), 3),
            "neural_window3_other": round(float(pn[idx[other]]), 3),
            "lstm": round(float(pl[idx[drug]]), 3),
            "lstm_other": round(float(pl[idx[other]]), 3)})
json.dump({"embedding_dim": 32, "hidden": 64, "epochs": len(lstm_curve), "curve": lstm_curve,
           "test_bits": lstm_curve[-1]["test_bits"], "test_perplexity": 2 ** lstm_curve[-1]["test_bits"],
           "n_parameters": int(sum(p.numel() for p in wl.parameters())),
           "long_range": long_range,
           "p_meropenem_after_started": float(lstm_next(["started"])[idx["meropenem"]])},
          open(f"{OUT}/lstm_word.json", "w"))
log("wrote lstm_word.json")

# ---------------------------------------------------------------- 4. character-level LSTM on Tiny Shakespeare
if os.path.exists(f"{OUT}/char_samples.json") and not os.environ.get("CH16_FORCE"):
    log("char_samples.json exists: character model skipped (set CH16_FORCE=1 to retrain)")
    raise SystemExit(0)
path = os.environ.get("CH16_SHAKESPEARE", "tinyshakespeare.txt")
text = open(path, encoding="utf-8").read()
chars = sorted(set(text)); ci = {c: i for i, c in enumerate(chars)}; Vc = len(chars)
data = torch.tensor([ci[c] for c in text], dtype=torch.long)
n_val = len(data) // 10
train_d, val_d = data[:-n_val], data[-n_val:]
torch.manual_seed(1606)

class CharLSTM(nn.Module):
    def __init__(self, V, d=48, h=256):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.rnn = nn.LSTM(d, h, batch_first=True)
        self.out = nn.Linear(h, V)
    def forward(self, x, state=None):
        o, state = self.rnn(self.emb(x), state); return self.out(o), state

cm = CharLSTM(Vc)
opt = torch.optim.Adam(cm.parameters(), lr=0.004)
SEQ, B = 100, 32
gen = torch.Generator().manual_seed(11)

def batch(d):
    ix = torch.randint(0, len(d) - SEQ - 1, (B,), generator=gen)
    x = torch.stack([d[i:i + SEQ] for i in ix]); y = torch.stack([d[i + 1:i + SEQ + 1] for i in ix])
    return x, y

def val_bits():
    cm.eval(); tot = 0.0
    g2 = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for _ in range(10):
            ix = torch.randint(0, len(val_d) - SEQ - 1, (B,), generator=g2)
            x = torch.stack([val_d[i:i + SEQ] for i in ix]); y = torch.stack([val_d[i + 1:i + SEQ + 1] for i in ix])
            z, _ = cm(x); tot += float(nn.functional.cross_entropy(z.reshape(-1, Vc), y.reshape(-1)))
    cm.train(); return tot / 10 / math.log(2)

def sample(n=300, tau=0.8, seed=5, prompt="ROMEO:\n"):
    cm.eval(); gs = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        x = torch.tensor([[ci[c] for c in prompt]]); z, st = cm(x); out = prompt
        for _ in range(n):
            p = torch.softmax(z[0, -1] / tau, -1)
            c = int(torch.multinomial(p, 1, generator=gs)); out += chars[c]
            z, st = cm(torch.tensor([[c]]), st)
    cm.train(); return out

STAGES = [0, 100, 300, 1000, 2500]
MAX_STEPS = STAGES[-1]
stages, char_curve = [], []
for step in range(MAX_STEPS + 1):
    if step % 100 == 0:
        char_curve.append({"step": step, "val_bits": round(val_bits(), 4)})
    if step in STAGES:
        stages.append({"step": step, "val_bits": char_curve[-1]["val_bits"],
                       "samples": {str(t): sample(tau=t) for t in (0.5, 0.8, 1.2)}})
        log(f"char LSTM step {step}: val {char_curve[-1]['val_bits']:.3f} bits/char")
    if step == MAX_STEPS: break
    x, y = batch(train_d)
    opt.zero_grad(); z, _ = cm(x)
    l = nn.functional.cross_entropy(z.reshape(-1, Vc), y.reshape(-1)); l.backward()
    nn.utils.clip_grad_norm_(cm.parameters(), 1.0); opt.step()
json.dump({"corpus": "Tiny Shakespeare", "n_chars": len(text), "vocab_size": Vc,
           "seq_len": SEQ, "batch": B, "hidden": 256,
           "n_parameters": int(sum(p.numel() for p in cm.parameters())),
           "uniform_bits": math.log2(Vc), "curve": char_curve, "stages": stages},
          open(f"{OUT}/char_samples.json", "w"))
log("wrote char_samples.json — done")
