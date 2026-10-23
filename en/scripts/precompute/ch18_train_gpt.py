"""Chapter 18 precompute: train a GPT on Florence Nightingale's *Notes on Nursing*.

Writes data/precomputed/ch18/ (every number quoted in the chapter comes from these files):
  summary.json            corpus, split, configurations, parameter counts, perplexities, timings,
                          gradient check, NumPy-versus-PyTorch check, memorisation summary
  baselines.csv           validation cross-entropy and perplexity of the baselines and the GPTs
  loss_small.csv          browser model (NumPy): training loss at every step, validation loss
  loss_big.csv            larger model (PyTorch): training and validation loss during training
  checkpoint_samples.json text written by the larger model at several stages of training
  samples.json            text written by the final model for 3 prompts x 5 temperatures x 2 top-k
  next_char.json          next-character logits of the final model after a few prompts
  memorisation.csv        verbatim overlap of generated text (and of real held-out text) with training
  gpt_nightingale.json    configuration, vocabulary and tensor layout of the final model
  gpt_nightingale.bin     its weights, float16, little-endian (read by numpy_gpt.py and the R Lab)

Usage (from the book folder; about 25 to 30 minutes on a 2-core laptop CPU):
    python3 scripts/precompute/ch18_train_gpt.py
    python3 scripts/precompute/ch18_train_gpt.py --big-steps 300   (quick check of the pipeline)
Needs numpy and torch (CPU). Fixed seeds throughout.
"""
import os, sys, json, time, math, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")          # time the NumPy model on one core,
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")     # as a browser would run it
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "gpt"))
import numpy_gpt as G

ap = argparse.ArgumentParser()
ap.add_argument("--big-steps", type=int, default=6000)
ap.add_argument("--small-steps", type=int, default=2000)
ap.add_argument("--out", default=os.path.join(ROOT, "data", "precomputed", "ch18"))
args = ap.parse_args()
OUT = args.out
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:7.1f} s]", *a, flush=True)

# ---------------------------------------------------------------------------------------------
# A. Data: characters as tokens, split by position (first 90 % train, last 10 % validation)
# ---------------------------------------------------------------------------------------------
text = open(os.path.join(ROOT, "data", "notes_on_nursing.txt"), encoding="utf-8").read()
tok = G.CharTokenizer(text)
data = tok.encode(text)
n_train = int(0.9 * len(data))
train_ids, val_ids = data[:n_train], data[n_train:]
V = tok.vocab_size
words = text.split()
summary = {"corpus": {"n_chars": len(text), "n_words": len(words), "n_paragraphs": text.count("\n"),
                      "vocab_size": V, "chars": "".join(tok.chars),
                      "n_train": int(n_train), "n_val": int(len(val_ids)),
                      "n_letters": int(sum(c.isalpha() for c in tok.chars)),
                      "val_starts_with": tok.decode(val_ids[:80])}}
log("corpus", summary["corpus"]["n_chars"], "characters, vocabulary", V)

# ---------------------------------------------------------------------------------------------
# B. Baselines on the validation text (Chapter 16 style, in bits per character)
# ---------------------------------------------------------------------------------------------
from collections import Counter, defaultdict
tr_txt, va_txt = text[:n_train], text[n_train:]

def ngram_tables(txt, n):
    """counts[k][context][char] for context lengths k = 0 .. n-1."""
    tabs = [defaultdict(Counter) for _ in range(n)]
    for i in range(len(txt)):
        for k in range(n):
            if i - k < 0:
                break
            tabs[k][txt[i - k:i]][txt[i]] += 1
    return tabs

def witten_bell_bits(tabs, txt, n):
    """Interpolated Witten-Bell estimate (Chen and Goodman, 1999); scores txt[1:]."""
    bits = []
    for i in range(1, len(txt)):
        p = 1.0 / V                                       # order "-1": uniform
        for k in range(0, min(n - 1, i) + 1):
            ctx = tabs[k].get(txt[i - k:i])
            if not ctx:
                break
            c_h = sum(ctx.values()); n1 = len(ctx)
            p = (ctx.get(txt[i], 0) + n1 * p) / (c_h + n1)
        bits.append(-math.log2(p))
    return float(np.mean(bits))

uni = Counter(tr_txt)
bits_uniform = math.log2(V)
bits_unigram = float(np.mean([-math.log2((uni.get(c, 0) + 1) / (len(tr_txt) + V)) for c in va_txt[1:]]))
bi = defaultdict(Counter)
for a, b in zip(tr_txt[:-1], tr_txt[1:]):
    bi[a][b] += 1
bits_bigram = float(np.mean([-math.log2((bi[a].get(b, 0) + 1) / (sum(bi[a].values()) + V))
                             for a, b in zip(va_txt[:-1], va_txt[1:])]))
tabs5 = ngram_tables(tr_txt, 5)
bits_tri = witten_bell_bits(tabs5[:3], va_txt, 3)
bits_five = witten_bell_bits(tabs5, va_txt, 5)
baselines = [("Uniform (every character equally likely)", bits_uniform),
             ("Unigram (character frequencies), add-one", bits_unigram),
             ("Bigram (previous character), add-one", bits_bigram),
             ("Trigram (two previous characters), Witten-Bell", bits_tri),
             ("5-gram (four previous characters), Witten-Bell", bits_five)]
log("baselines", [(n, round(b, 3)) for n, b in baselines])

# bigram table on all positions of the training text, for the chapter's quick numbers
after_t = {c: n / sum(bi["t"].values()) for c, n in bi["t"].most_common(5)}
summary["bigram_after_t"] = after_t
summary["unigram_top"] = {c: n / len(tr_txt) for c, n in uni.most_common(6)}

# ---------------------------------------------------------------------------------------------
# C. Gradient check of numpy_gpt.backward against finite differences (float64)
# ---------------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
pc, cc = G.init_params(V, d=16, n_layer=2, n_head=2, block_size=8, seed=3)
for k in pc:
    pc[k] += 0.1 * rng.standard_normal(pc[k].shape)
xb, yb = G.get_batch(train_ids, 8, 4, rng)
_, _, cache = G.forward(pc, cc, xb, yb)
grads = G.backward(pc, cc, cache)
rel_errs = []
for k in pc:
    for _ in range(6):
        i = tuple(int(rng.integers(0, s)) for s in pc[k].shape)
        if k == "wte":                                    # pick a row that is actually used
            i = (int(xb.reshape(-1)[rng.integers(0, xb.size)]), i[1])
        old, h = pc[k][i], 1e-5
        pc[k][i] = old + h; lp = G.forward(pc, cc, xb, yb)[1]
        pc[k][i] = old - h; lm = G.forward(pc, cc, xb, yb)[1]
        pc[k][i] = old
        num, an = (lp - lm) / (2 * h), grads[k][i]
        rel_errs.append(abs(num - an) / max(abs(num) + abs(an), 1e-10))
# one worked example for the chapter: the gradient of one output-layer (embedding) weight
k0, i0 = "lnf.g", (0,)
old = pc[k0][i0]; h = 1e-5
pc[k0][i0] = old + h; lp = G.forward(pc, cc, xb, yb)[1]
pc[k0][i0] = old - h; lm = G.forward(pc, cc, xb, yb)[1]
pc[k0][i0] = old
summary["gradient_check"] = {"n_checked": len(rel_errs), "max_rel_error": float(max(rel_errs)),
                             "median_rel_error": float(np.median(rel_errs)),
                             "example": {"param": k0, "finite_difference": float((lp - lm) / (2 * h)),
                                         "backprop": float(grads[k0][i0])}}
log("gradient check: max relative error", max(rel_errs))

# ---------------------------------------------------------------------------------------------
# D. The browser model: NumPy GPT, same settings and seeds as the chapter's cells
# ---------------------------------------------------------------------------------------------
SMALL = dict(d=64, n_layer=2, n_head=4, block_size=32)
SMALL_TRAIN = dict(batch_size=16, lr=3e-3, init_seed=0, batch_seed=1, browser_steps=200)
ps, cs = G.init_params(V, **SMALL, seed=SMALL_TRAIN["init_seed"])
counts_small = G.count_params(ps, cs)
prompt_small = "The nurse"
def sample_small():
    return tok.decode(G.generate(ps, cs, tok.encode(prompt_small), 200, temperature=0.8, seed=2))
small_before = sample_small()
val_small = [(0, G.estimate_loss(ps, cs, val_ids))]
opt = None
rows_small = []
t_start = time.time()
done = 0
milestones = sorted({SMALL_TRAIN["browser_steps"], 500, 1000, args.small_steps})
timing = None
for m in milestones:
    if m > args.small_steps:
        continue
    t1 = time.time()
    opt, hist = G.train(ps, cs, train_ids, steps=m - done, batch_size=SMALL_TRAIN["batch_size"],
                        lr=SMALL_TRAIN["lr"], seed=SMALL_TRAIN["batch_seed"] + done, opt=opt,
                        print_every=0)
    if done == 0:
        timing = (time.time() - t1) / m
    rows_small += hist
    done = m
    val_small.append((m, G.estimate_loss(ps, cs, val_ids)))
    if m == SMALL_TRAIN["browser_steps"]:
        small_after = sample_small()
        small_browser_params = {k: v.copy() for k, v in ps.items()}
    log(f"small model, step {m}: validation loss {val_small[-1][1]:.3f}")
small_final = sample_small()
with open(os.path.join(OUT, "loss_small.csv"), "w") as f:
    f.write("step,train_loss,val_loss\n")
    vs = dict(val_small)
    f.write(f"0,,{vs[0]:.5f}\n")
    for s, l in rows_small:
        f.write(f"{s},{l:.5f},{vs[s]:.5f}\n" if s in vs else f"{s},{l:.5f},\n")
summary["small"] = {"config": {**SMALL, "vocab_size": V}, "train": SMALL_TRAIN, "params": counts_small,
                    "ms_per_step_native": 1000 * timing, "steps_total": args.small_steps,
                    "val_loss": {str(s): v for s, v in val_small},
                    "prompt": prompt_small, "sample_before": small_before,
                    "sample_after_browser_steps": small_after, "sample_final": small_final}

# ---------------------------------------------------------------------------------------------
# E. The larger model, trained with PyTorch
# ---------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(2)
torch.manual_seed(1337)

BIG = dict(d=128, n_layer=4, n_head=4, block_size=64)
BIG_TRAIN = dict(batch_size=32, lr=3e-3, min_lr=3e-4, warmup=100, dropout=0.1, steps=args.big_steps,
                 eval_every=100, clip=1.0, seed=1337)


class Block(nn.Module):
    """Pre-LN transformer block: x + MHA(LN(x)), then x + FFN(LN(x)) (Chapter 17)."""
    def __init__(self, d, n_head, dropout):
        super().__init__()
        self.n_head, self.dropout = n_head, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.attn(self.ln1(x)).split(d, dim=2)
        q, k, v = [a.view(B, T, self.n_head, d // self.n_head).transpose(1, 2) for a in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, d)))
        return x + self.drop(self.fc2(F.gelu(self.fc(self.ln2(x)), approximate="tanh")))


class GPT(nn.Module):
    def __init__(self, vocab_size, d, n_layer, n_head, block_size, dropout=0.0):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(vocab_size, d), nn.Embedding(block_size, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, n_head, dropout) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        for name, prm in self.named_parameters():          # GPT-2 style initialisation
            if prm.dim() == 2:
                nn.init.normal_(prm, 0.0, 0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(prm)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.drop(self.wte(idx) + self.wpe(torch.arange(T)))
        for b in self.blocks:
            x = b(x)
        return self.lnf(x) @ self.wte.weight.T              # weight tying


def to_numpy_params(model):
    """PyTorch state -> the parameter names and layouts of numpy_gpt (Linear weights transposed)."""
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    p = {"wte": sd["wte.weight"], "wpe": sd["wpe.weight"], "lnf.g": sd["lnf.weight"], "lnf.b": sd["lnf.bias"]}
    for l in range(len(model.blocks)):
        s = f"blocks.{l}."
        p[f"h{l}.ln1.g"], p[f"h{l}.ln1.b"] = sd[s + "ln1.weight"], sd[s + "ln1.bias"]
        p[f"h{l}.ln2.g"], p[f"h{l}.ln2.b"] = sd[s + "ln2.weight"], sd[s + "ln2.bias"]
        for name in ("attn", "proj", "fc", "fc2"):
            p[f"h{l}.{name}.w"] = sd[s + name + ".weight"].T.copy()
            p[f"h{l}.{name}.b"] = sd[s + name + ".bias"]
    return p


def from_numpy_params(model, p):
    sd = {"wte.weight": p["wte"], "wpe.weight": p["wpe"], "lnf.weight": p["lnf.g"], "lnf.bias": p["lnf.b"]}
    for l in range(len(model.blocks)):
        s = f"blocks.{l}."
        sd[s + "ln1.weight"], sd[s + "ln1.bias"] = p[f"h{l}.ln1.g"], p[f"h{l}.ln1.b"]
        sd[s + "ln2.weight"], sd[s + "ln2.bias"] = p[f"h{l}.ln2.g"], p[f"h{l}.ln2.b"]
        for name in ("attn", "proj", "fc", "fc2"):
            sd[s + name + ".weight"] = p[f"h{l}.{name}.w"].T
            sd[s + name + ".bias"] = p[f"h{l}.{name}.b"]
    model.load_state_dict({k: torch.tensor(np.ascontiguousarray(v), dtype=torch.float32) for k, v in sd.items()})


def eval_windows(n, T):
    """Windows of T characters every T/2; each window scores only its second half (the first
    window scores everything), so every validation character after the first is scored once,
    with at least T/2 characters of context except at the very start."""
    starts = list(range(0, n - T - 1, T // 2))
    return [(s, 0 if s == 0 else T // 2) for s in starts]


@torch.no_grad()
def val_loss_torch(model, ids, T):
    model.eval()
    ids_t = torch.tensor(ids)
    tot, cnt = 0.0, 0
    win = eval_windows(len(ids), T)
    for i in range(0, len(win), 64):
        chunk = win[i:i + 64]
        x = torch.stack([ids_t[s:s + T] for s, _ in chunk])
        y = torch.stack([ids_t[s + 1:s + T + 1] for s, _ in chunk])
        l = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), reduction="none").view(len(chunk), T)
        for r, (s, a) in enumerate(chunk):
            tot += float(l[r, a:].sum()); cnt += T - a
    model.train()
    return tot / cnt


def val_loss_numpy(p, cfg, ids):
    T = cfg["block_size"]
    tot, cnt = 0.0, 0
    win = eval_windows(len(ids), T)
    for i in range(0, len(win), 16):
        chunk = win[i:i + 16]
        x = np.stack([ids[s:s + T] for s, _ in chunk])
        y = np.stack([ids[s + 1:s + T + 1] for s, _ in chunk])
        logits, _, _ = G.forward(p, cfg, x)
        lp = logits - logits.max(-1, keepdims=True)
        lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
        nll = -np.take_along_axis(lp, y[..., None], -1)[..., 0]
        for r, (s, a) in enumerate(chunk):
            tot += float(nll[r, a:].sum()); cnt += T - a
    return tot / cnt


@torch.no_grad()
def sample_torch(model, prompt, n, temperature=0.8, seed=0):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    ids = list(tok.encode(prompt))
    for _ in range(n):
        x = torch.tensor([ids[-BIG["block_size"]:]])
        probs = F.softmax(model(x)[0, -1] / temperature, dim=-1)
        ids.append(int(torch.multinomial(probs, 1, generator=g)))
    model.train()
    return tok.decode(ids)


model = GPT(V, **BIG, dropout=BIG_TRAIN["dropout"])
counts_big = G.count_params(to_numpy_params(model), {**BIG, "vocab_size": V})
assert counts_big["total"] == sum(p.numel() for p in model.parameters())
optim = torch.optim.Adam(model.parameters(), lr=BIG_TRAIN["lr"], betas=(0.9, 0.99))
train_t = torch.tensor(train_ids)
gen = torch.Generator().manual_seed(BIG_TRAIN["seed"])
CKPT = [0, 100, 300, 1000, args.big_steps]
PROMPT_CKPT = "The nurse"
ckpt_samples, loss_rows = [], []
best = (float("inf"), 0, None)
steps = BIG_TRAIN["steps"]
def lr_at(it):
    """Linear warm-up, then cosine decay from lr to min_lr."""
    if it < BIG_TRAIN["warmup"]:
        return BIG_TRAIN["lr"] * (it + 1) / BIG_TRAIN["warmup"]
    r = (it - BIG_TRAIN["warmup"]) / max(1, steps - BIG_TRAIN["warmup"])
    return BIG_TRAIN["min_lr"] + 0.5 * (1 + math.cos(math.pi * r)) * (BIG_TRAIN["lr"] - BIG_TRAIN["min_lr"])

t_big = time.time()
running = []
for it in range(0, steps + 1):
    if it % BIG_TRAIN["eval_every"] == 0 or it in CKPT:
        vl = val_loss_torch(model, val_ids, BIG["block_size"])
        tl = float(np.mean(running)) if running else None
        loss_rows.append((it, tl, vl)); running = []
        if vl < best[0]:
            best = (vl, it, {k: v.clone() for k, v in model.state_dict().items()})
        log(f"big model step {it}: train {tl if tl is None else round(tl, 3)}  val {vl:.3f}")
    if it in CKPT:
        ckpt_samples.append({"step": it, "val_loss": vl,
                             "text": sample_torch(model, PROMPT_CKPT, 300, 0.8, seed=7)})
    if it == steps:
        break
    for gr in optim.param_groups:
        gr["lr"] = lr_at(it)
    ix = torch.randint(0, len(train_t) - BIG["block_size"] - 1, (BIG_TRAIN["batch_size"],), generator=gen)
    x = torch.stack([train_t[i:i + BIG["block_size"]] for i in ix])
    y = torch.stack([train_t[i + 1:i + BIG["block_size"] + 1] for i in ix])
    loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), BIG_TRAIN["clip"])
    optim.step()
    running.append(loss.item())
big_minutes = (time.time() - t_big) / 60
with open(os.path.join(OUT, "loss_big.csv"), "w") as f:
    f.write("step,train_loss,val_loss\n")
    for s, tl, vl in loss_rows:
        f.write(f"{s},{'' if tl is None else f'{tl:.5f}'},{vl:.5f}\n")
json.dump({"prompt": PROMPT_CKPT, "temperature": 0.8, "samples": ckpt_samples},
          open(os.path.join(OUT, "checkpoint_samples.json"), "w"), indent=1, ensure_ascii=False)

# keep the checkpoint with the lowest validation loss (early stopping)
model.load_state_dict(best[2])
model.eval()

# ---------------------------------------------------------------------------------------------
# F. Export (float16) and check the NumPy forward pass against PyTorch
# ---------------------------------------------------------------------------------------------
p_big = to_numpy_params(model)
p16 = {k: v.astype(np.float16) for k, v in p_big.items()}
layout, offset, blobs = [], 0, []
for k in sorted(p16):
    a = np.ascontiguousarray(p16[k]).astype("<f2")
    layout.append({"name": k, "shape": list(a.shape), "offset": offset, "size": int(a.size)})
    offset += a.size; blobs.append(a.reshape(-1))
np.concatenate(blobs).tofile(os.path.join(OUT, "gpt_nightingale.bin"))
json.dump({"config": {**BIG, "vocab_size": V}, "chars": "".join(tok.chars), "dtype": "float16",
           "tensors": layout}, open(os.path.join(OUT, "gpt_nightingale.json"), "w"), ensure_ascii=False)

pn, cn, tokn = G.load_pretrained(os.path.join(OUT, "gpt_nightingale.json"), dtype=np.float64)
assert tokn.chars == tok.chars
model16 = GPT(V, **BIG)
from_numpy_params(model16, {k: v.astype(np.float32) for k, v in p16.items()})
model16.eval()
xchk = np.stack([val_ids[i:i + BIG["block_size"]] for i in (0, 1000, 5000, 9000)])
with torch.no_grad():
    lt = model16(torch.tensor(xchk)).numpy().astype(np.float64)
ln_, _, _ = G.forward(pn, cn, xchk)
max_diff = float(np.abs(ln_ - lt).max())
val_big_fp32 = val_loss_torch(model, val_ids, BIG["block_size"])
val_big_fp16 = val_loss_torch(model16, val_ids, BIG["block_size"])
t1 = time.time(); val_big_numpy = val_loss_numpy(pn, cn, val_ids); t_np_eval = time.time() - t1
log(f"NumPy vs PyTorch logits: max abs diff {max_diff:.2e}; val loss fp32 {val_big_fp32:.4f}, "
    f"fp16 {val_big_fp16:.4f}, numpy {val_big_numpy:.4f}")
p32 = {k: v.astype(np.float32) for k, v in pn.items()}
t1 = time.time(); G.generate(p32, cn, tok.encode("The nurse"), 100, 0.8, seed=0)
ms_per_char = 1000 * (time.time() - t1) / 100
val_small_final = G.estimate_loss(ps, cs, val_ids)       # small model, on its own windows
val_small_windows = val_loss_numpy(ps, cs, val_ids)
val_small_browser_windows = val_loss_numpy(small_browser_params, cs, val_ids)

# ---------------------------------------------------------------------------------------------
# G. Samples at several temperatures and top-k (the same numpy_gpt.generate the browser runs)
# ---------------------------------------------------------------------------------------------
PROMPTS = ["The nurse", "Fresh air", "The patient"]
TEMPS = [0.2, 0.5, 0.8, 1.0, 1.5]
TOPKS = [0, 5]
samples = []
for pr in PROMPTS:
    for tau in TEMPS:
        for k in TOPKS:
            ids = G.generate(p32, cn, tok.encode(pr), 250, temperature=tau, top_k=k or None, seed=11)
            samples.append({"prompt": pr, "temperature": tau, "top_k": k, "text": tok.decode(ids)})
json.dump({"n_new": 250, "seed": 11, "samples": samples},
          open(os.path.join(OUT, "samples.json"), "w"), indent=1, ensure_ascii=False)
nxt = []
for pr in ["The nurse", "Fresh air", "The patient", "The sick ", "In watching the sick, the nurse must"]:
    logits, _, _ = G.forward(p32, cn, tok.encode(pr)[None, :])
    nxt.append({"prompt": pr, "logits": [round(float(z), 4) for z in logits[0, -1]]})
QUESTION = "What is the best treatment for a fever?"
question_sample = tok.decode(G.generate(p32, cn, tok.encode(QUESTION), 250, temperature=0.8, seed=11))
json.dump({"chars": "".join(tok.chars), "contexts": nxt},
          open(os.path.join(OUT, "next_char.json"), "w"), ensure_ascii=False)
log("samples written")

# ---------------------------------------------------------------------------------------------
# H. Memorisation: how much generated text is copied verbatim from the training text?
# ---------------------------------------------------------------------------------------------
W = 20
train_windows = {tr_txt[i:i + W] for i in range(len(tr_txt) - W + 1)}

def longest_copied(s):
    """Longest substring of s that appears verbatim in the training text."""
    best_len, best_str, j = 0, "", 0
    for i in range(len(s)):
        j = max(j, i + best_len)
        while j < len(s) and s[i:j + 1] in tr_txt:
            j += 1
        if j - i > best_len:
            best_len, best_str = j - i, s[i:j]
    return best_len, best_str

def share_copied(s):
    win = [s[i:i + W] for i in range(len(s) - W + 1)]
    return float(np.mean([w in train_windows for w in win]))

mem_rows = []
N_MEM, LEN_MEM = 20, 500
for tau in [0.5, 0.8, 1.0]:
    for r in range(N_MEM):
        g = sample_torch(model16, "\n", LEN_MEM, temperature=tau, seed=100 + r)[1:]   # PyTorch: faster
        L, s = longest_copied(g)
        mem_rows.append({"source": f"generated, temperature {tau}", "temperature": tau, "sample": r,
                         "longest": L, "longest_text": s, "share20": share_copied(g)})
for r in range(N_MEM):                                   # control: real held-out text
    st = (r * (len(va_txt) - LEN_MEM)) // N_MEM
    g = va_txt[st:st + LEN_MEM]
    L, s = longest_copied(g)
    mem_rows.append({"source": "real validation text", "temperature": None, "sample": r,
                     "longest": L, "longest_text": s, "share20": share_copied(g)})
import csv
with open(os.path.join(OUT, "memorisation.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(mem_rows[0].keys()))
    w.writeheader(); w.writerows(mem_rows)
mem_sum = {}
for src in sorted({r["source"] for r in mem_rows}):
    rr = [r for r in mem_rows if r["source"] == src]
    top = max(rr, key=lambda r: r["longest"])
    mem_sum[src] = {"median_longest": float(np.median([r["longest"] for r in rr])),
                    "max_longest": top["longest"], "max_longest_text": top["longest_text"],
                    "mean_share20": float(np.mean([r["share20"] for r in rr]))}
log("memorisation", {k: (v["median_longest"], round(v["mean_share20"], 3)) for k, v in mem_sum.items()})

# ---------------------------------------------------------------------------------------------
# I. Tables and summary
# ---------------------------------------------------------------------------------------------
rows = baselines + [(f"Small GPT (NumPy, browser size), {SMALL_TRAIN['browser_steps']} steps",
                     val_small_browser_windows / math.log(2)),
                    (f"Small GPT (NumPy, browser size), {args.small_steps} steps", val_small_windows / math.log(2)),
                    ("Larger GPT (PyTorch), best checkpoint", val_big_fp16 / math.log(2))]
with open(os.path.join(OUT, "baselines.csv"), "w") as f:
    f.write("model,bits_per_char,nats_per_char,perplexity\n")
    for n, b in rows:
        f.write(f"\"{n}\",{b:.4f},{b * math.log(2):.4f},{2 ** b:.4f}\n")

summary["baselines_bits"] = {n: b for n, b in baselines}
summary["big"] = {"config": {**BIG, "vocab_size": V}, "train": BIG_TRAIN, "params": counts_big,
                  "minutes": big_minutes, "s_per_step": 60 * big_minutes / steps,
                  "best_step": best[1], "best_val_loss": best[0],
                  "final_val_loss": loss_rows[-1][2], "final_train_loss": loss_rows[-1][1],
                  "val_loss_fp32": val_big_fp32, "val_loss_fp16": val_big_fp16,
                  "val_loss_numpy": val_big_numpy, "numpy_eval_seconds": t_np_eval,
                  "max_abs_logit_diff": max_diff, "ms_per_char_numpy": ms_per_char,
                  "chars_seen": steps * BIG_TRAIN["batch_size"] * BIG["block_size"],
                  "epochs": steps * BIG_TRAIN["batch_size"] * BIG["block_size"] / n_train,
                  "bin_bytes": os.path.getsize(os.path.join(OUT, "gpt_nightingale.bin"))}
summary["small"]["val_loss_windows_final"] = val_small_windows
summary["small"]["val_loss_windows_browser"] = val_small_browser_windows
summary["small"]["val_loss_final"] = val_small_final
summary["small"]["epochs_browser"] = SMALL_TRAIN["browser_steps"] * 16 * SMALL["block_size"] / n_train
summary["question"] = {"prompt": QUESTION, "sample": question_sample}
summary["memorisation"] = {"window": W, "n_samples": N_MEM, "length": LEN_MEM, "by_source": mem_sum}
summary["versions"] = {"numpy": np.__version__, "torch": torch.__version__, "python": sys.version.split()[0]}
summary["total_minutes"] = (time.time() - T0) / 60
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, ensure_ascii=False)
log("done in", round(summary["total_minutes"], 1), "minutes")
