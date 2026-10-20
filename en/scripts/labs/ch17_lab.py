# Chapter 17 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Tokens, embeddings and positions ---------------------------------------
import numpy as np
import pandas as pd
rng = np.random.default_rng(2026)
vocab  = ["<bos>", "the", "patient", "reports", "denies", "chest", "pain", "fever"]
tokens = ["<bos>", "the", "patient", "reports", "chest", "pain"]
ids = np.array([vocab.index(w) for w in tokens])
d, h = 8, 2; d_k, d_ff, T = d // h, 4 * d, len(ids)
E = rng.normal(scale=0.5, size=(len(vocab), d))                   # embedding matrix |V| x d
def pos_enc(T, d):                                                # sinusoidal encodings
    P = np.zeros((T, d))
    for t in range(T):
        for r in range(d // 2):
            P[t, 2 * r] = np.sin(t / 10000 ** (2 * r / d))
            P[t, 2 * r + 1] = np.cos(t / 10000 ** (2 * r / d))
    return P
X = E[ids] + pos_enc(T, d)                                        # T x d input
print(X.shape)

# 2. Building blocks ----------------------------------------------------------
def softmax_rows(S):
    Z = np.exp(S - S.max(axis=1, keepdims=True)); return Z / Z.sum(axis=1, keepdims=True)
def layer_norm(X):
    mu = X.mean(axis=1, keepdims=True); var = X.var(axis=1, keepdims=True)
    return (X - mu) / np.sqrt(var + 1e-5)
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
def new_block():
    m = lambda r, c: rng.normal(scale=0.3, size=(r, c))
    return dict(W_Q=m(d, d), W_K=m(d, d), W_V=m(d, d), W_O=m(d, d), W_1=m(d, d_ff), W_2=m(d_ff, d))
causal = np.triu(np.full((T, T), -np.inf), k=1)                   # the causal mask

def mha(X, B):                                                    # multi-head attention
    Q, K, V = X @ B["W_Q"], X @ B["W_K"], X @ B["W_V"]
    outs, As = [], []
    for m in range(h):
        cols = slice(m * d_k, (m + 1) * d_k)                      # this head's slice
        A = softmax_rows(Q[:, cols] @ K[:, cols].T / np.sqrt(d_k) + causal)
        outs.append(A @ V[:, cols]); As.append(A)
    return np.hstack(outs) @ B["W_O"], As
def block(X, B):                                                  # pre-LN block (biases and gains omitted)
    att, As = mha(layer_norm(X), B)
    X1 = X + att                                                  # X' = X + MHA(LN(X))
    return X1 + gelu(layer_norm(X1) @ B["W_1"]) @ B["W_2"], As    # Y = X' + FFN(LN(X'))

# 3. Two stacked blocks -------------------------------------------------------
B1, B2 = new_block(), new_block()
Y1, A1 = block(X, B1); Y2, A2 = block(Y1, B2)
print(Y2.shape)                                                   # still T x d
print(np.round(A1[0], 2))                                         # head 1, layer 1: upper triangle is 0
print(A1[0].sum(axis=1))

# 4. Next-token probabilities (weight tying) and cross-entropy ------------------
probs = softmax_rows(layer_norm(Y2) @ E.T)                        # T x |V|
next_ids = ids[1:]                                                # the token that actually follows
nll = -np.log(probs[np.arange(T - 1), next_ids])
print({"mean_cross_entropy": round(float(nll.mean()), 3), "uniform_guess": round(float(np.log(len(vocab))), 3)})

# 5. Weights in one block (with biases and layer-norm gains) ---------------------
print({"attention": 4 * (d * d + d), "ffn": d * d_ff + d_ff + d_ff * d + d, "layernorm": 4 * d})

# 6. Attention of the trained tiny transformer ----------------------------------
att = pd.read_csv("data/precomputed/ch17/attention.csv")
sent = pd.read_csv("data/precomputed/ch17/sentences.csv")
tok1 = sent.loc[sent["sentence"] == 1, "token"].tolist()
a = att[(att["sentence"] == 1) & (att["layer"] == 1) & (att["head"] == 1)]   # att.head would be a method
A_trained = np.zeros((len(tok1), len(tok1)))
A_trained[a["query"], a["key"]] = a["weight"]
last_row = pd.Series(A_trained[-1], index=tok1)
# where the last position looks
print(last_row.sort_values(ascending=False).head(3))
