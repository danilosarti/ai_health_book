"""Chapter 17 precompute: a tiny decoder-only (GPT-style) transformer trained on synthetic
Ward 7 medication notes, compared with fixed-window n-gram predictors.

The corpus is generated here from templates (fixed seed). In every note the last word (the
blood test to check tonight) is determined by the drug named at the start of the note, and the
pronouns are determined by the patient's name, so the decisive word is often 10-30 tokens back.

Runs on a laptop CPU in about a minute (2 threads). The book only reads the outputs.

Usage (from the book folder):
    python3 scripts/precompute/ch17_tiny_transformer.py

Outputs
    data/precomputed/ch17/summary.json        corpus, model size, parameter counts, accuracies
    data/precomputed/ch17/ngram_vs_tf.csv     accuracy on the last word: n-gram windows vs transformer
    data/precomputed/ch17/attention.csv       attention weights (sentence, layer, head, query, key)
    data/precomputed/ch17/sentences.csv       tokens of the example sentences shown in the widget
    data/precomputed/ch17/training_loss.csv   training and validation loss during training
    data/precomputed/ch17/corpus_sample.csv   twelve notes from the corpus
"""
import json, math, os, random, time
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
SEED = 2026
OUT = "data/precomputed/ch17"
os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
t0 = time.time()

# ------------------------------------------------------------------ corpus
NAMES = {"aoife": ("she", "her", "her"), "priya": ("she", "her", "her"), "tomas": ("he", "his", "him")}
DRUG_TEST = {"vancomycin": "creatinine", "gentamicin": "creatinine", "furosemide": "potassium",
             "warfarin": "inr", "insulin": "glucose", "heparin": "platelets",
             "digoxin": "pulse", "morphine": "breathing"}
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
FILLERS = ["{s} slept well", "{p} family visited", "the physio saw {o}", "{s} ate some lunch",
           "{p} lines were changed", "{s} sat in the chair", "{s} walked to the window",
           "the team reviewed {o}", "{s} asked for tea", "{p} daughter phoned"]


def make_note(rng):
    name = rng.choice(list(NAMES)); s, p, o = NAMES[name]
    drug = rng.choice(list(DRUG_TEST)); day = rng.choice(DAYS)
    k = rng.randint(0, 5)
    fill = [rng.choice(FILLERS).format(s=s, p=p, o=o) for _ in range(k)]
    middle = (" . " + " and ".join(fill)) if fill else ""
    text = f"{name} was started on {drug} on {day}{middle} . tonight we check {p} {DRUG_TEST[drug]}"
    return text.split(), drug, name


rng = random.Random(SEED)
train_notes = [make_note(rng) for _ in range(6000)]
val_notes = [make_note(rng) for _ in range(500)]
test_notes = [make_note(rng) for _ in range(1000)]
vocab = ["<pad>", "<bos>"] + sorted({w for t, _, _ in train_notes + val_notes + test_notes for w in t})
stoi = {w: i for i, w in enumerate(vocab)}
PAD, BOS = 0, 1
max_len = max(len(t) for t, _, _ in train_notes + val_notes + test_notes) + 1
print("vocabulary:", len(vocab), "max length:", max_len)


def encode(notes):
    X = torch.full((len(notes), max_len), PAD, dtype=torch.long)
    for i, (t, _, _) in enumerate(notes):
        ids = [BOS] + [stoi[w] for w in t]
        X[i, :len(ids)] = torch.tensor(ids)
    return X


Xtr, Xva, Xte = encode(train_notes), encode(val_notes), encode(test_notes)


# ------------------------------------------------------------------ model (pre-LayerNorm)
class CausalSelfAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d // h
        self.Wq, self.Wk, self.Wv, self.Wo = (nn.Linear(d, d) for _ in range(4))
        self.last = None

    def forward(self, x, pad_mask):
        n, T, d = x.shape
        split = lambda z: z.view(n, T, self.h, self.dk).transpose(1, 2)        # (n, h, T, dk)
        Q, K, V = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))
        S = Q @ K.transpose(-2, -1) / math.sqrt(self.dk)                        # (n, h, T, T)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(causal, float("-inf"))
        S = S.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        A = torch.softmax(S, dim=-1)
        A = torch.nan_to_num(A)                       # rows that are entirely padding
        self.last = A.detach()
        Z = (A @ V).transpose(1, 2).reshape(n, T, d)  # concatenate heads
        return self.Wo(Z)


class Block(nn.Module):
    def __init__(self, d, h, d_ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, h)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, x, pad_mask):
        x = x + self.attn(self.ln1(x), pad_mask)      # X' = X + MHA(LN(X))
        x = x + self.ffn(self.ln2(x))                 # Y  = X' + FFN(LN(X'))
        return x


class TinyGPT(nn.Module):
    def __init__(self, V, T, d=32, h=2, L=2, d_ff=128):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, h, d_ff) for _ in range(L))
        self.ln_f = nn.LayerNorm(d)

    def forward(self, idx):
        n, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))[None]
        pad = idx == PAD
        for b in self.blocks:
            x = b(x, pad)
        return self.ln_f(x) @ self.tok.weight.T          # weight tying: logits = h E^T


CFG = dict(d=32, h=2, L=2, d_ff=128)
model = TinyGPT(len(vocab), max_len, **CFG)
n_params = sum(p.numel() for p in model.parameters())
per_block = sum(p.numel() for p in model.blocks[0].parameters())
print("parameters:", n_params, "per block:", per_block)


def lm_loss(X):
    logits = model(X[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), X[:, 1:].reshape(-1), ignore_index=PAD)


opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
STEPS, BATCH = 1500, 64
hist = []
for step in range(1, STEPS + 1):
    model.train()
    idx = torch.randint(0, len(Xtr), (BATCH,))
    loss = lm_loss(Xtr[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0 or step == 1:
        model.eval()
        with torch.no_grad():
            vl = lm_loss(Xva).item()
        hist.append((step, round(loss.item(), 4), round(vl, 4)))
        if step % 250 == 0:
            print(f"step {step}: train {loss.item():.3f} val {vl:.3f}")


# ------------------------------------------------------------------ evaluation on the last word
def last_word_accuracy(X, notes):
    model.eval()
    with torch.no_grad():
        logits = model(X[:, :-1])
    lens = torch.tensor([len(t) for t, _, _ in notes])          # tokens without <bos>
    pos_target = lens - 1                                       # index into logits predicting token lens
    pos_pron = lens - 2
    pred_t = logits[torch.arange(len(notes)), pos_target].argmax(-1)
    pred_p = logits[torch.arange(len(notes)), pos_pron].argmax(-1)
    true_t = X[torch.arange(len(notes)), lens]
    true_p = X[torch.arange(len(notes)), lens - 1]
    return (pred_t == true_t).float().mean().item(), (pred_p == true_p).float().mean().item()


acc_tf, acc_pron_tf = last_word_accuracy(Xte, test_notes)
print("transformer accuracy on last word:", acc_tf, "on pronoun:", acc_pron_tf)


def ngram_accuracy(k, which="target"):
    table = defaultdict(Counter); overall = Counter()
    off = 1 if which == "target" else 2
    for t, _, _ in train_notes:
        seq = ["<bos>"] + t
        j = len(seq) - off
        ctx = tuple(seq[max(0, j - k):j]); table[ctx][seq[j]] += 1; overall[seq[j]] += 1
    hit = 0
    for t, _, _ in test_notes:
        seq = ["<bos>"] + t
        j = len(seq) - off
        ctx = tuple(seq[max(0, j - k):j])
        c = table.get(ctx) or overall
        hit += c.most_common(1)[0][0] == seq[j]
    return hit / len(test_notes)


# distance between the drug and the last word, in tokens: the drug is inside a window of the
# last k words before the target exactly when dist <= k
dist = [len(t) - 1 - t.index(drug) for t, drug, _ in test_notes]
ks = [1, 2, 3, 4, 6, 8, 12, 16]
rows = [("n-gram", k, round(ngram_accuracy(k), 4), round(ngram_accuracy(k, "pronoun"), 4),
         round(float(np.mean(np.array(dist) <= k)), 4)) for k in ks]
rows.append(("transformer", max_len, round(acc_tf, 4), round(acc_pron_tf, 4), 1.0))
with open(f"{OUT}/ngram_vs_tf.csv", "w") as f:
    f.write("model,window,acc_last_word,acc_pronoun,drug_visible\n")
    for r in rows:
        f.write(",".join(map(str, r)) + "\n")

# ------------------------------------------------------------------ attention maps for the widget
examples = [
    "aoife was started on vancomycin on monday . she slept well and her family visited . tonight we check her creatinine",
    "tomas was started on warfarin on friday . the physio saw him . tonight we check his inr",
    "priya was started on insulin on sunday . her lines were changed and she sat in the chair and the team reviewed her . tonight we check her glucose",
    "tomas was started on furosemide on tuesday . he asked for tea and his daughter phoned and he walked to the window . tonight we check his potassium",
    "aoife was started on heparin on thursday . tonight we check her platelets",
]
sent_rows, att_rows, ex_preds = [], [], []
model.eval()
for s_id, text in enumerate(examples, start=1):
    toks = text.split()
    ids = torch.tensor([[BOS] + [stoi[w] for w in toks]])
    with torch.no_grad():
        logits = model(ids[:, :-1])
    probs = torch.softmax(logits[0, -1], -1)
    top = torch.topk(probs, 3)
    ex_preds.append({"sentence": s_id, "target": toks[-1],
                     "top": [[vocab[i], round(float(p), 4)] for p, i in zip(top.values, top.indices)]})
    shown = ["<bos>"] + toks[:-1]                     # positions the model reads
    for q, w in enumerate(shown):
        sent_rows.append((s_id, q, w))
    for l, b in enumerate(model.blocks):
        A = b.attn.last[0]                            # (h, T, T)
        for hh in range(A.shape[0]):
            for q in range(A.shape[1]):
                for k in range(q + 1):
                    att_rows.append((s_id, l + 1, hh + 1, q, k, round(float(A[hh, q, k]), 4)))

with open(f"{OUT}/sentences.csv", "w") as f:
    f.write("sentence,pos,token\n")
    for r in sent_rows:
        f.write(",".join(map(str, r)) + "\n")
with open(f"{OUT}/attention.csv", "w") as f:
    f.write("sentence,layer,head,query,key,weight\n")
    for r in att_rows:
        f.write(",".join(map(str, r)) + "\n")
with open(f"{OUT}/training_loss.csv", "w") as f:
    f.write("step,train_loss,val_loss\n")
    for r in hist:
        f.write(",".join(map(str, r)) + "\n")
with open(f"{OUT}/corpus_sample.csv", "w") as f:
    f.write("note\n")
    for t, _, _ in train_notes[:12]:
        f.write('"' + " ".join(t) + '"\n')

# Which head looks at the drug when the model predicts the last word? (share of attention)
drug_focus = []
for s_id, text in enumerate(examples, start=1):
    toks = ["<bos>"] + text.split()[:-1]
    drug_pos = next(i for i, w in enumerate(toks) if w in DRUG_TEST)
    q = len(toks) - 1
    for r in att_rows:
        if r[0] == s_id and r[3] == q and r[4] == drug_pos:
            drug_focus.append({"sentence": s_id, "layer": r[1], "head": r[2], "weight_on_drug": r[5]})

summary = {
    "seed": SEED, "n_train": len(train_notes), "n_val": len(val_notes), "n_test": len(test_notes),
    "vocab_size": len(vocab), "max_len": max_len, "config": CFG,
    "n_params": n_params, "params_per_block": per_block,
    "params_embeddings": sum(p.numel() for p in model.tok.parameters()) + sum(p.numel() for p in model.pos.parameters()),
    "steps": STEPS, "batch": BATCH,
    "final_train_loss": hist[-1][1], "final_val_loss": hist[-1][2],
    "acc_last_word_tf": round(acc_tf, 4), "acc_pronoun_tf": round(acc_pron_tf, 4),
    "chance_last_word": round(Counter(DRUG_TEST[d] for _, d, _ in test_notes).most_common(1)[0][1] / len(test_notes), 4),
    "dist_drug_median": float(np.median(dist)), "dist_drug_max": int(max(dist)), "dist_drug_min": int(min(dist)),
    "example_predictions": ex_preds, "drug_focus": drug_focus,
    "runtime_seconds": round(time.time() - t0, 1),
}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k not in ("example_predictions", "drug_focus")}, indent=1))
print("drug focus:", drug_focus)
