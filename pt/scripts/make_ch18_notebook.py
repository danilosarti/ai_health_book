"""Build notebooks/ch18_train_gpt.ipynb (Chapter 18): train a character-level GPT on Florence
Nightingale's Notes on Nursing with PyTorch, on a laptop CPU or in Google Colab.

The model, data split, schedule and seeds are those of scripts/precompute/ch18_train_gpt.py.
    python3 scripts/make_ch18_notebook.py            # write the notebook
    python3 scripts/make_ch18_notebook.py --check    # also run a copy top to bottom with 200 steps
"""
import os, sys, nbformat as nbf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "notebooks", "ch18_train_gpt.ipynb")
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# Train your first GPT: *Notes on Nursing*

Companion notebook to Chapter 18 of *AI and Machine Learning for Health Sciences* (Danilo Augusto Sarti).

You will train, from scratch, a small GPT that writes one character at a time, on Florence
Nightingale's *Notes on Nursing: What It Is, and What It Is Not* (1859), a public-domain text.
The model has about 0.8 million weights: four transformer blocks, four attention heads, vectors of
length 128 and a context of 64 characters. It is the "larger model" of the chapter.

**How long?** On a recent laptop CPU, a few hundred training steps take about a minute and the full
6 000 steps 15 to 40 minutes. In Google Colab with a GPU (Runtime > Change runtime type > GPU) the
full run takes a few minutes. You can stop early: set `MAX_STEPS` below to a smaller number.

**Files needed:** `notes_on_nursing.txt` (download it from the chapter). In Colab, the first code
cell asks you to upload it; on a laptop, put it in the same folder as this notebook.

Run the cells from top to bottom (Shift + Enter)."""),
md("## 1. Settings"),
code("""import math, os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_STEPS = 6000          # training steps (try 500 for a quick run)
BATCH_SIZE = 32           # windows of text per step
BLOCK_SIZE = 64           # context length: characters the model can look back at
D, N_LAYER, N_HEAD = 128, 4, 4
DROPOUT = 0.1
LR, MIN_LR, WARMUP = 3e-3, 3e-4, 100
EVAL_EVERY = 250

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print("PyTorch", torch.__version__, "on", device)"""),
md("## 2. The data\n\nThe text is read as one long string. If the file is missing and you are in Colab, you will be asked to upload it."),
code("""PATH = "notes_on_nursing.txt"
if not os.path.exists(PATH):
    try:
        from google.colab import files          # only available in Google Colab
        print("Please upload notes_on_nursing.txt (downloaded from Chapter 18).")
        files.upload()
    except ImportError:
        raise FileNotFoundError("Put notes_on_nursing.txt in the same folder as this notebook.")
text = open(PATH, encoding="utf-8").read()
print(f"{len(text):,} characters")
print(text[:500])"""),
md("## 3. The tokenizer: every character is a token\n\nReal GPTs cut text into sub-word tokens (Chapter 15); here each distinct character gets a number."),
code("""chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(chars[i] for i in ids)
print(vocab_size, "distinct characters:", repr("".join(chars)))
print(encode("Fresh air"), "->", decode(encode("Fresh air")))"""),
md("## 4. Training and validation text, and batches\n\nThe first 90 % of the book is for training and the last 10 % for validation (split by position, so that no sentence is shared). A batch is a set of random windows; the targets are the same windows shifted by one character, so every position is a training example."),
code("""data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
gen = torch.Generator().manual_seed(1337)

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,), generator=gen)
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)

x, y = get_batch("train")
print("input :", repr(decode(x[0, :40].tolist())))
print("target:", repr(decode(y[0, :40].tolist())))"""),
md("## 5. The model: a decoder-only, pre-LN transformer (Chapter 17)\n\nToken embedding plus position embedding, then blocks of *layer norm, masked multi-head attention, residual* and *layer norm, feed-forward network, residual*, a final layer norm and an output layer that reuses the token embeddings (weight tying)."),
code("""class Block(nn.Module):
    \"\"\"Pre-LN transformer block: x + MHA(LN(x)), then x + FFN(LN(x)).\"\"\"
    def __init__(self, d, n_head, dropout):
        super().__init__()
        self.n_head, self.dropout = n_head, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.attn(self.ln1(x)).split(d, dim=2)                  # queries, keys, values
        q, k, v = [a.view(B, T, self.n_head, d // self.n_head).transpose(1, 2) for a in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,     # causal mask
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, d)))  # residual 1
        return x + self.drop(self.fc2(F.gelu(self.fc(self.ln2(x)), approximate="tanh")))  # residual 2


class GPT(nn.Module):
    def __init__(self, vocab_size, d, n_layer, n_head, block_size, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.wte, self.wpe = nn.Embedding(vocab_size, d), nn.Embedding(block_size, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, n_head, dropout) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        for name, p in self.named_parameters():          # small random weights, zero biases
            if p.dim() == 2:
                nn.init.normal_(p, 0.0, 0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.drop(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        return self.lnf(x) @ self.wte.weight.T              # logits, weight tying

model = GPT(vocab_size, D, N_LAYER, N_HEAD, BLOCK_SIZE, DROPOUT).to(device)
print(f"{sum(p.numel() for p in model.parameters()):,} weights")"""),
md("## 6. Sampling: temperature and top-k\n\nBefore training, the model writes noise."),
code("""@torch.no_grad()
def generate(prompt, n_new=300, temperature=0.8, top_k=None, seed=0):
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    ids = encode(prompt)
    for _ in range(n_new):
        x = torch.tensor([ids[-BLOCK_SIZE:]], device=device)
        z = model(x)[0, -1] / temperature
        if top_k:
            z[z < torch.topk(z, top_k).values[-1]] = -float("inf")
        ids.append(int(torch.multinomial(F.softmax(z, dim=-1), 1, generator=g)))
    model.train()
    return decode(ids)

print(generate("The nurse"))"""),
md("## 7. Training: cross-entropy, backpropagation and Adam\n\nEvery step: take a batch, compute the cross-entropy of every next character, backpropagate, clip the gradient, and let Adam update the weights. The learning rate warms up, then decays. Every `EVAL_EVERY` steps we measure the loss on the validation text and keep the best model (early stopping)."),
code("""@torch.no_grad()
def val_loss():
    \"\"\"Mean cross-entropy (nats per character) on the validation text, in windows that overlap by half.\"\"\"
    model.eval()
    T, tot, cnt = BLOCK_SIZE, 0.0, 0
    starts = list(range(0, len(val_data) - T - 1, T // 2))
    for i in range(0, len(starts), 64):
        s = starts[i:i + 64]
        x = torch.stack([val_data[j:j + T] for j in s]).to(device)
        y = torch.stack([val_data[j + 1:j + T + 1] for j in s]).to(device)
        l = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1), reduction="none").view(len(s), T)
        for r, j in enumerate(s):
            a = 0 if j == 0 else T // 2                    # score each character once
            tot += l[r, a:].sum().item(); cnt += T - a
    model.train()
    return tot / cnt

def lr_at(it):
    if it < WARMUP:
        return LR * (it + 1) / WARMUP
    r = (it - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * r)) * (LR - MIN_LR)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))
history, best, t0, running = [], (float("inf"), None, 0), time.time(), []
for it in range(MAX_STEPS + 1):
    if it % EVAL_EVERY == 0 or it == MAX_STEPS:
        vl = val_loss()
        tl = float(np.mean(running)) if running else float("nan")
        history.append((it, tl, vl)); running = []
        if vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, it)
        print(f"step {it:5d}  train {tl:.3f}  validation {vl:.3f}  ({time.time() - t0:.0f} s)")
    if it == MAX_STEPS:
        break
    for group in optimizer.param_groups:
        group["lr"] = lr_at(it)
    x, y = get_batch("train")
    loss = F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    running.append(loss.item())

model.load_state_dict(best[1])
print(f"best validation loss {best[0]:.3f} nats per character at step {best[2]}"
      f" (perplexity {math.exp(best[0]):.2f})")"""),
md("## 8. The loss curves\n\nTraining loss falls steadily; validation loss tells you whether the model generalises to text it has not seen."),
code("""import matplotlib.pyplot as plt
h = np.array(history, dtype=float)
plt.plot(h[1:, 0], h[1:, 1], label="training")
plt.plot(h[:, 0], h[:, 2], label="validation")
plt.xlabel("training step"); plt.ylabel("cross-entropy (nats per character)"); plt.legend(); plt.show()"""),
md("## 9. Write like Nightingale\n\nChange the prompt, the temperature and top-k."),
code("""for tau in (0.5, 0.8, 1.2):
    print(f"--- temperature {tau}")
    print(generate("The nurse", 300, temperature=tau, seed=1))"""),
md("## 10. Save the model\n\n`ch18_gpt.pt` is a PyTorch checkpoint. The same weights are also written in the book's format (`ch18_gpt.json` plus `ch18_gpt.bin`, float16), which `numpy_gpt.load_pretrained` from the chapter can read."),
code("""torch.save({"model": model.state_dict(), "chars": chars,
            "config": dict(d=D, n_layer=N_LAYER, n_head=N_HEAD, block_size=BLOCK_SIZE)}, "ch18_gpt.pt")

sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
p = {"wte": sd["wte.weight"], "wpe": sd["wpe.weight"], "lnf.g": sd["lnf.weight"], "lnf.b": sd["lnf.bias"]}
for l in range(N_LAYER):
    s = f"blocks.{l}."
    p[f"h{l}.ln1.g"], p[f"h{l}.ln1.b"] = sd[s + "ln1.weight"], sd[s + "ln1.bias"]
    p[f"h{l}.ln2.g"], p[f"h{l}.ln2.b"] = sd[s + "ln2.weight"], sd[s + "ln2.bias"]
    for name in ("attn", "proj", "fc", "fc2"):
        p[f"h{l}.{name}.w"], p[f"h{l}.{name}.b"] = sd[s + name + ".weight"].T, sd[s + name + ".bias"]
layout, offset, blobs = [], 0, []
for k in sorted(p):
    a = np.ascontiguousarray(p[k]).astype("<f2")
    layout.append({"name": k, "shape": list(a.shape), "offset": offset, "size": int(a.size)})
    offset += a.size; blobs.append(a.reshape(-1))
np.concatenate(blobs).tofile("ch18_gpt.bin")
json.dump({"config": dict(d=D, n_layer=N_LAYER, n_head=N_HEAD, block_size=BLOCK_SIZE, vocab_size=vocab_size),
           "chars": "".join(chars), "dtype": "float16", "tensors": layout}, open("ch18_gpt.json", "w"))
print("saved ch18_gpt.pt, ch18_gpt.json and ch18_gpt.bin")"""),
md("""## 11. Things to try

1. Set `DROPOUT = 0.0` and `MAX_STEPS = 8000`: does the validation loss start to rise while the training loss keeps falling (overfitting)?
2. Halve `D` or `N_LAYER`. How does the best validation loss change? Count the weights.
3. Train on the first half of the book only. How much worse is the model?
4. Check memorisation: generate 2 000 characters and search for their longest piece that appears word for word in `text`."""),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT)

if "--check" in sys.argv:
    import copy, shutil, tempfile
    from nbclient import NotebookClient
    tmp = tempfile.mkdtemp()
    shutil.copy(os.path.join(ROOT, "data", "notes_on_nursing.txt"), tmp)
    nb2 = copy.deepcopy(nb)
    src = nb2.cells[2].source
    nb2.cells[2].source = src.replace("MAX_STEPS = 6000", "MAX_STEPS = 200").replace("EVAL_EVERY = 250", "EVAL_EVERY = 100")
    NotebookClient(nb2, timeout=1800, kernel_name="python3", resources={"metadata": {"path": tmp}}).execute()
    for c in nb2.cells:
        if c.cell_type == "code":
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream":
                    print(o["text"][-600:])
    print("check OK: all cells ran;", sorted(os.listdir(tmp)))
