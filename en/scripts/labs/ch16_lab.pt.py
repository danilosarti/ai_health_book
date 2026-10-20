# Laboratório do capítulo 16
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Carregar e tokenizar as anotações -----------------------------------------
import numpy as np, pandas as pd
notes = pd.read_csv("data/ward7_notes.csv")
toks = lambda s: ["<s>"] + s.split(" ") + ["</s>"]
train = [toks(s) for s in notes.loc[notes["set"] == "train", "text"]]
test = [toks(s) for s in notes.loc[notes["set"] == "test", "text"]]
vocab = sorted({w for doc in train + test for w in doc})
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab) - 1                              # qualquer token exceto <s> pode vir em seguida

# 2. Um modelo de bigramas por contagem (máxima verossimilhança) ---------------
counts = np.zeros((len(vocab), len(vocab)))
for w in train:
    for a, b in zip(w[:-1], w[1:]):
        counts[idx[a], idx[b]] += 1
row = counts[idx["reports"]] / counts[idx["reports"]].sum()
print({vocab[j]: round(row[j], 3) for j in np.argsort(-row)[:5]})

# 3. Suavização add-delta e perplexidade nas anotações de teste ----------------
def perplexity(delta):
    lp = [np.log2((counts[idx[a], idx[b]] + delta) / (counts[idx[a]].sum() + delta * V))
          for w in test for a, b in zip(w[:-1], w[1:])]
    return 2 ** (-np.mean(lp))
print({d: round(perplexity(d), 3) for d in (1, 0.1, 0.01)})

# 4. Gerar anotações por amostragem, com uma temperatura -----------------------
rng = np.random.default_rng(16)
def generate(tau=1.0, max_len=30):
    w, out = "<s>", []
    for _ in range(max_len):
        p = counts[idx[w]] / counts[idx[w]].sum()
        p = p ** (1 / tau); p = p / p.sum()     # temperatura nas probabilidades = nas log-contagens
        w = vocab[rng.choice(len(vocab), p=p)]
        if w == "</s>":
            break
        out.append(w)
    return " ".join(out)
for tau in (0.3, 1, 2):
    print("tau =", tau, ":", generate(tau))

# 5. Um pequeno modelo de linguagem neural (janela de 3 tokens), treinado do zero
def windows(docs):
    rows = []
    for w in docs:
        w = ["<s>", "<s>"] + w                  # preenche para que todo token tenha 3 antecessores
        rows += [[idx[w[i - 3]], idx[w[i - 2]], idx[w[i - 1]], idx[w[i]]] for i in range(3, len(w))]
    return np.array(rows)
Wtr, Wte = windows(train), windows(test)
rng = np.random.default_rng(2003)
nV, d, H, lr = len(vocab), 16, 64, 0.1
E = rng.normal(0, 0.1, (nV, d))
W1 = rng.normal(0, 1 / np.sqrt(3 * d), (3 * d, H)); b1 = np.zeros(H)
W2 = rng.normal(0, 1 / np.sqrt(H), (H, nV)); b2 = np.zeros(nV)
def forward(X):
    x = E[X].reshape(len(X), -1)
    h = np.tanh(x @ W1 + b1)
    z = h @ W2 + b2
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return x, h, p
def bits(W):
    _, _, p = forward(W[:, :3])
    return -np.mean(np.log2(p[np.arange(len(W)), W[:, 3]]))
print("antes do treino:", round(bits(Wte), 3), "bits por token")
for epoch in range(1, 6):
    order = rng.permutation(len(Wtr))
    for s in range(0, len(order), 64):
        X, y = Wtr[order[s:s + 64], :3], Wtr[order[s:s + 64], 3]
        x, h, p = forward(X)
        g = p.copy(); g[np.arange(len(y)), y] -= 1; g /= len(y)   # dPerda/dz
        gh = (g @ W2.T) * (1 - h ** 2)
        gx = (gh @ W1.T).reshape(len(X), 3, d)
        W2 -= lr * h.T @ g; b2 -= lr * g.sum(0)
        W1 -= lr * x.T @ gh; b1 -= lr * gh.sum(0)
        np.add.at(E, X, -lr * gx)                # cada posição atualiza suas linhas de E
    print("época", epoch, ": entropia cruzada de teste", round(bits(Wte), 3), "bits, perplexidade", round(2 ** bits(Wte), 2))
_, _, p = forward(np.array([[idx["started"], idx["vancomycin"], idx["for"]]]))
print({vocab[j]: round(p[0, j], 3) for j in np.argsort(-p[0])[:5]})

# 6. Os embeddings skip-gram pré-calculados: vizinhos e uma analogia -----------
wv = pd.read_csv("data/precomputed/ch16/word_vectors.csv")
U = wv.filter(regex=r"^d[0-9]+$").to_numpy()
U = U / np.linalg.norm(U, axis=1, keepdims=True)    # comprimento unitário: cosseno = produto escalar
words = list(wv["word"])
def nearest(v, k=5, exclude=()):
    s = U @ (v / np.linalg.norm(v))
    return [(words[j], round(s[j], 3)) for j in np.argsort(-s) if words[j] not in exclude][:k]
print(nearest(U[words.index("vancomycin")], exclude=("vancomycin",)))
u = lambda w: U[words.index(w)]
print(nearest(u("nephrology") - u("kidney") + u("heart"), exclude=("nephrology", "kidney", "heart")))
