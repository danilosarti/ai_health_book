"""numpy_gpt.py: a complete GPT, written from scratch in NumPy (Chapter 18).

Everything a GPT needs, in one readable file and with no deep-learning library:
  1. a character tokenizer (encode / decode) and random training batches;
  2. the parameters: token and position embeddings, N pre-LN transformer blocks (layer norm, causal
     multi-head self-attention, layer norm, feed-forward network with GELU, two residual
     connections), a final layer norm, and an output layer that reuses the token embeddings
     (weight tying), exactly the decoder of Chapter 17;
  3. the forward pass and the cross-entropy loss;
  4. the backward pass: the gradient of the loss with respect to every parameter, computed by
     hand with the chain rule (checked against finite differences in the precompute script);
  5. the Adam optimiser and a training step;
  6. text generation with temperature and top-k sampling;
  7. loading weights exported from the PyTorch model of the chapter (same names and layout).

Arrays have shape (B, T, d): B texts in a batch, T positions, d numbers per position.
The book runs this file in the browser (Pyodide) and in scripts/precompute/ch18_train_gpt.py.
"""
import time
import numpy as np


# ---------------------------------------------------------------------------------------------
# 1. Tokenizer and batches
# ---------------------------------------------------------------------------------------------
class CharTokenizer:
    """Every distinct character of the training text is one token."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, s):
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.chars[int(i)] for i in ids)


def get_batch(data, block_size, batch_size, rng):
    """B random windows of the text: inputs x and targets y (the same windows shifted by one)."""
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[s:s + block_size] for s in starts])
    y = np.stack([data[s + 1:s + block_size + 1] for s in starts])
    return x, y


# ---------------------------------------------------------------------------------------------
# 2. Parameters
# ---------------------------------------------------------------------------------------------
def init_params(vocab_size, d=32, n_layer=2, n_head=2, block_size=32, seed=0, dtype=np.float64):
    """Small random weights (standard deviation 0.02), zero biases, layer-norm gains of 1."""
    rng = np.random.default_rng(seed)
    w = lambda *shape: (0.02 * rng.standard_normal(shape)).astype(dtype)
    z = lambda *shape: np.zeros(shape, dtype=dtype)
    o = lambda *shape: np.ones(shape, dtype=dtype)
    p = {"wte": w(vocab_size, d), "wpe": w(block_size, d)}
    for l in range(n_layer):
        p.update({
            f"h{l}.ln1.g": o(d), f"h{l}.ln1.b": z(d),
            f"h{l}.attn.w": w(d, 3 * d), f"h{l}.attn.b": z(3 * d),     # W_Q, W_K, W_V side by side
            f"h{l}.proj.w": w(d, d), f"h{l}.proj.b": z(d),             # W_O
            f"h{l}.ln2.g": o(d), f"h{l}.ln2.b": z(d),
            f"h{l}.fc.w": w(d, 4 * d), f"h{l}.fc.b": z(4 * d),         # feed-forward, layer 1
            f"h{l}.fc2.w": w(4 * d, d), f"h{l}.fc2.b": z(d),           # feed-forward, layer 2
        })
    p["lnf.g"], p["lnf.b"] = o(d), z(d)
    cfg = dict(vocab_size=vocab_size, d=d, n_layer=n_layer, n_head=n_head, block_size=block_size)
    return p, cfg


def count_params(p, cfg):
    """Number of weights by component (the output layer reuses wte, so it adds nothing)."""
    L = cfg["n_layer"]
    size = lambda keys: int(sum(p[k].size for k in keys))
    blk = lambda part: size([f"h{l}.{part}.{s}" for l in range(L) for s in ("w", "b")])
    lns = size([f"h{l}.{n}.{s}" for l in range(L) for n in ("ln1", "ln2") for s in ("g", "b")])
    out = {"token embedding": size(["wte"]), "position embedding": size(["wpe"]),
           "attention": blk("attn") + blk("proj"), "feed-forward": blk("fc") + blk("fc2"),
           "layer norms": lns + size(["lnf.g", "lnf.b"]), "output layer (tied)": 0}
    out["total"] = sum(out.values())
    return out


# ---------------------------------------------------------------------------------------------
# 3. Forward pass
# ---------------------------------------------------------------------------------------------
GELU_C = np.sqrt(2.0 / np.pi)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(GELU_C * (x + 0.044715 * x ** 3)))


def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(x.var(axis=-1, keepdims=True) + eps)
    xhat = (x - mu) * rstd
    return g * xhat + b, (xhat, rstd, g)


def softmax(z, axis=-1):
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def attention(x, w, b, w_o, b_o, n_head):
    """Causal multi-head self-attention: softmax(Q K^T / sqrt(d_k) + M) V, then W_O."""
    B, T, d = x.shape
    dk = d // n_head
    qkv = x @ w + b                                                   # (B, T, 3d)
    q, k, v = np.split(qkv, 3, axis=-1)
    heads = lambda a: a.reshape(B, T, n_head, dk).transpose(0, 2, 1, 3)   # (B, h, T, dk)
    q, k, v = heads(q), heads(k), heads(v)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dk)                # (B, h, T, T)
    future = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(future, -np.inf, scores)                        # the causal mask
    A = softmax(scores)                                               # attention weights
    y = (A @ v).transpose(0, 2, 1, 3).reshape(B, T, d)                # heads side by side
    out = y @ w_o + b_o
    return out, (x, q, k, v, A, y)


def forward(p, cfg, idx, targets=None):
    """Logits (B, T, V) for every position; also the mean cross-entropy if targets are given."""
    B, T = idx.shape
    L, h = cfg["n_layer"], cfg["n_head"]
    x = p["wte"][idx] + p["wpe"][:T]                                  # token + position
    cache = {"idx": idx}
    for l in range(L):
        a_in, c_ln1 = layer_norm(x, p[f"h{l}.ln1.g"], p[f"h{l}.ln1.b"])
        a_out, c_att = attention(a_in, p[f"h{l}.attn.w"], p[f"h{l}.attn.b"],
                                 p[f"h{l}.proj.w"], p[f"h{l}.proj.b"], h)
        x = x + a_out                                                 # residual 1
        m_in, c_ln2 = layer_norm(x, p[f"h{l}.ln2.g"], p[f"h{l}.ln2.b"])
        pre = m_in @ p[f"h{l}.fc.w"] + p[f"h{l}.fc.b"]
        act = gelu(pre)
        x = x + act @ p[f"h{l}.fc2.w"] + p[f"h{l}.fc2.b"]             # residual 2
        cache[l] = (c_ln1, c_att, c_ln2, m_in, pre, act)
    xf, cache["lnf"] = layer_norm(x, p["lnf.g"], p["lnf.b"])
    logits = xf @ p["wte"].T                                          # weight tying
    cache["xf"] = xf
    if targets is None:
        return logits, None, cache
    probs = softmax(logits)
    n = B * T
    pt = probs.reshape(n, -1)[np.arange(n), targets.reshape(n)]
    loss = float(-np.mean(np.log(pt)))
    cache["probs"], cache["targets"] = probs, targets
    return logits, loss, cache


# ---------------------------------------------------------------------------------------------
# 4. Backward pass: the chain rule, one operation at a time, from the loss back to the inputs
# ---------------------------------------------------------------------------------------------
def layer_norm_backward(dy, c):
    xhat, rstd, g = c
    dg = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(axis=0)
    db = dy.reshape(-1, xhat.shape[-1]).sum(axis=0)
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean(axis=-1, keepdims=True)
                 - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
    return dx, dg, db


def gelu_backward(dy, x):
    u = GELU_C * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    du = GELU_C * (1.0 + 3 * 0.044715 * x ** 2)
    return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du)


def attention_backward(dout, c, w, w_o, n_head):
    x, q, k, v, A, y = c
    B, T, d = x.shape
    dk = d // n_head
    flat = lambda a: a.reshape(-1, a.shape[-1])
    dw_o = flat(y).T @ flat(dout)
    db_o = flat(dout).sum(axis=0)
    dy = (dout @ w_o.T).reshape(B, T, n_head, dk).transpose(0, 2, 1, 3)
    dA = dy @ v.transpose(0, 1, 3, 2)
    dv = A.transpose(0, 1, 3, 2) @ dy
    dscores = A * (dA - (dA * A).sum(axis=-1, keepdims=True))        # softmax backward
    dscores /= np.sqrt(dk)                                            # masked cells have A = 0
    dq = dscores @ k
    dk_ = dscores.transpose(0, 1, 3, 2) @ q
    merge = lambda a: a.transpose(0, 2, 1, 3).reshape(B, T, d)
    dqkv = np.concatenate([merge(dq), merge(dk_), merge(dv)], axis=-1)
    dw = flat(x).T @ flat(dqkv)
    db = flat(dqkv).sum(axis=0)
    dx = dqkv @ w.T
    return dx, dw, db, dw_o, db_o


def backward(p, cfg, cache):
    """Gradient of the mean cross-entropy with respect to every parameter (same keys as p)."""
    g = {k: np.zeros_like(v) for k, v in p.items()}
    probs, targets, idx = cache["probs"], cache["targets"], cache["idx"]
    B, T, V = probs.shape
    # loss = mean of -log softmax: d loss / d logits = (probs - one_hot(target)) / (B T)
    dlogits = probs.copy()
    dlogits.reshape(B * T, V)[np.arange(B * T), targets.reshape(-1)] -= 1.0
    dlogits /= B * T
    xf = cache["xf"]
    g["wte"] += dlogits.reshape(-1, V).T @ xf.reshape(-1, xf.shape[-1])   # output layer (tied)
    dxf = dlogits @ p["wte"]
    dx, g["lnf.g"], g["lnf.b"] = layer_norm_backward(dxf, cache["lnf"])
    for l in reversed(range(cfg["n_layer"])):
        c_ln1, c_att, c_ln2, m_in, pre, act = cache[l]
        # residual 2: x = x + FFN(LN(x)); the gradient flows both straight through and into FFN
        flat = lambda a: a.reshape(-1, a.shape[-1])
        g[f"h{l}.fc2.w"] = flat(act).T @ flat(dx)
        g[f"h{l}.fc2.b"] = flat(dx).sum(axis=0)
        dpre = gelu_backward(dx @ p[f"h{l}.fc2.w"].T, pre)
        g[f"h{l}.fc.w"] = flat(m_in).T @ flat(dpre)
        g[f"h{l}.fc.b"] = flat(dpre).sum(axis=0)
        dm_in = dpre @ p[f"h{l}.fc.w"].T
        d_ln2, g[f"h{l}.ln2.g"], g[f"h{l}.ln2.b"] = layer_norm_backward(dm_in, c_ln2)
        dx = dx + d_ln2
        # residual 1: x = x + MHA(LN(x))
        da_in, g[f"h{l}.attn.w"], g[f"h{l}.attn.b"], g[f"h{l}.proj.w"], g[f"h{l}.proj.b"] = \
            attention_backward(dx, c_att, p[f"h{l}.attn.w"], p[f"h{l}.proj.w"], cfg["n_head"])
        d_ln1, g[f"h{l}.ln1.g"], g[f"h{l}.ln1.b"] = layer_norm_backward(da_in, c_ln1)
        dx = dx + d_ln1
    # embeddings: each row that was looked up receives the gradient of its positions
    np.add.at(g["wte"], idx.reshape(-1), dx.reshape(-1, dx.shape[-1]))
    g["wpe"][:T] += dx.sum(axis=0)
    return g


# ---------------------------------------------------------------------------------------------
# 5. Adam and the training step
# ---------------------------------------------------------------------------------------------
class Adam:
    """Adam (Kingma and Ba, 2015): a step size for each weight from running averages of its
    gradient (m) and squared gradient (v)."""

    def __init__(self, params, lr=3e-3, beta1=0.9, beta2=0.99, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, beta1, beta2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_gradients(grads, max_norm=1.0):
    """Shrink all gradients together if their overall length exceeds max_norm."""
    norm = float(np.sqrt(sum(float((v ** 2).sum()) for v in grads.values())))
    if norm > max_norm:
        for k in grads:
            grads[k] *= max_norm / norm
    return norm


def train(p, cfg, data, steps=200, batch_size=16, lr=3e-3, seed=1, opt=None,
          print_every=50, val_data=None, eval_batches=8, log=print):
    """Run `steps` training steps; return the optimiser and a list of (step, train loss)."""
    rng = np.random.default_rng(seed)
    opt = opt or Adam(p, lr=lr)
    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(data, cfg["block_size"], batch_size, rng)
        _, loss, cache = forward(p, cfg, x, y)
        grads = backward(p, cfg, cache)
        clip_gradients(grads)
        opt.step(p, grads)
        history.append((opt.t, loss))
        if print_every and (step % print_every == 0 or step == steps):
            msg = f"step {opt.t:5d}  training loss {loss:.3f}  ({time.time() - t0:.1f} s)"
            if val_data is not None:
                msg += f"  validation loss {estimate_loss(p, cfg, val_data, eval_batches):.3f}"
            log(msg)
    return opt, history


def estimate_loss(p, cfg, data, n_batches=None, batch_size=16):
    """Mean cross-entropy (nats per character) over consecutive windows of the text."""
    T = cfg["block_size"]
    n_win = (len(data) - 1) // T
    if n_batches is not None:
        n_win = min(n_win, n_batches * batch_size)
    total, count = 0.0, 0
    for s in range(0, n_win, batch_size):
        starts = np.arange(s, min(s + batch_size, n_win)) * T
        x = np.stack([data[i:i + T] for i in starts])
        y = np.stack([data[i + 1:i + T + 1] for i in starts])
        _, loss, _ = forward(p, cfg, x, y)
        total += loss * x.size
        count += x.size
    return total / count


# ---------------------------------------------------------------------------------------------
# 6. Generation
# ---------------------------------------------------------------------------------------------
def next_char_probs(p, cfg, ids, temperature=1.0, top_k=None):
    """Probabilities of the next character after the context ids (the last block_size of them)."""
    ctx = np.asarray(ids)[-cfg["block_size"]:][None, :]
    logits, _, _ = forward(p, cfg, ctx)
    z = logits[0, -1] / temperature
    if top_k:
        cut = np.sort(z)[-top_k]
        z = np.where(z < cut, -np.inf, z)
    return softmax(z)


def generate(p, cfg, ids, n_new, temperature=1.0, top_k=None, seed=0):
    """Append n_new sampled characters to ids, one at a time."""
    rng = np.random.default_rng(seed)
    ids = list(ids)
    for _ in range(n_new):
        probs = next_char_probs(p, cfg, ids, temperature, top_k)
        ids.append(int(rng.choice(len(probs), p=probs)))
    return ids


# ---------------------------------------------------------------------------------------------
# 7. Loading pretrained weights (exported by the precompute script: a JSON description plus a
#    binary file of float16 numbers, which R can read too)
# ---------------------------------------------------------------------------------------------
def load_pretrained(json_path, dtype=np.float32):
    """Weights, configuration and tokenizer of the model saved by the precompute script."""
    import json, os
    meta = json.load(open(json_path, encoding="utf-8"))
    flat = np.fromfile(os.path.splitext(json_path)[0] + ".bin", dtype="<f2")
    p = {t["name"]: flat[t["offset"]:t["offset"] + t["size"]].reshape(t["shape"]).astype(dtype)
         for t in meta["tensors"]}
    return p, dict(meta["config"]), CharTokenizer(meta["chars"])
