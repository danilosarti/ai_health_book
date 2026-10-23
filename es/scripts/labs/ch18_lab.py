# Chapter 18 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. The book, the tokenizer and the split -----------------------------------------
import sys, time
import numpy as np
sys.path.insert(0, "scripts/gpt")
import numpy_gpt as G
text = open("data/notes_on_nursing.txt", encoding="utf-8").read()
tok = G.CharTokenizer(text)
data = tok.encode(text)
n_train = int(0.9 * len(data))
train_ids, val_ids = data[:n_train], data[n_train:]
print(len(data), tok.vocab_size, len(train_ids), len(val_ids))

# 2. A batch: inputs and targets shifted by one --------------------------------------
x, y = G.get_batch(train_ids, 32, 4, np.random.default_rng(1))
print(repr(tok.decode(x[0])), "|", repr(tok.decode(y[0])))

# 3. Baselines on the validation text (bits per character) ---------------------------
V = tok.vocab_size
p_uni = (np.bincount(train_ids, minlength=V) + 1) / (len(train_ids) + V)
pair = np.zeros((V, V)); np.add.at(pair, (train_ids[:-1], train_ids[1:]), 1)
p_bi = (pair + 1) / (pair.sum(axis=1, keepdims=True) + V)
bl = {"uniform": np.log2(V), "unigram": -np.mean(np.log2(p_uni[val_ids[1:]])),
      "bigram": -np.mean(np.log2(p_bi[val_ids[:-1], val_ids[1:]]))}
print({k: (round(v, 3), round(2 ** v, 2)) for k, v in bl.items()})

# 4. Train the browser GPT for 200 steps (a minute or two in a browser) -------------
p, cfg = G.init_params(V, d=64, n_layer=2, n_head=4, block_size=32, seed=0)
print(G.count_params(p, cfg))
opt, hist = G.train(p, cfg, train_ids, steps=200, batch_size=16, lr=3e-3, seed=1, print_every=50)
print("validation loss:", round(G.estimate_loss(p, cfg, val_ids), 3), "nats; bigram:",
      round(bl["bigram"] * np.log(2), 3), "nats")

# 5. Load the larger pretrained GPT and check it against the precomputed logits -------
pb, cb, tokb = G.load_pretrained("data/precomputed/ch18/gpt_nightingale.json")
import json
nxt = json.load(open("data/precomputed/ch18/next_char.json", encoding="utf-8"))
z_py = np.array([c["logits"] for c in nxt["contexts"] if c["prompt"] == "The nurse"][0])
logits, _, _ = G.forward(pb, cb, tokb.encode("The nurse")[None, :])
print("max difference:", np.abs(logits[0, -1] - z_py).max())

# 6. Generate with temperature and top-k ----------------------------------------------
for tau in (0.5, 1.0):
    ids = G.generate(pb, cb, tokb.encode("Fresh air"), 120, temperature=tau, top_k=10, seed=1)
    print(f"tau = {tau}:", repr(tokb.decode(ids)))
