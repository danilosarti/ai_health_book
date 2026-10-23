# Chapter 19 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Load the guideline chunks and build a TF-IDF index ------------------------
import re
import numpy as np
import pandas as pd
guidelines = pd.read_csv("data/ch19_guidelines.csv")
stop_words = set("""a an and are as at be by can do does for from has have how i if in is it its not of on or our
should so that the their them then there these they this to was we what when which while who will with without
you been into than after before more most also all any every each may only other such whenever where one two""".split())
def tokens(x):
    return [w for w in re.split(r"[^a-z0-9]+", x.lower()) if len(w) >= 2 and w not in stop_words]
docs = [tokens(s + " " + t) for s, t in zip(guidelines["section"], guidelines["text"])]
vocab = sorted(set(w for d in docs for w in d)); pos = {w: i for i, w in enumerate(vocab)}
tf = np.zeros((len(docs), len(vocab)))
for j, d in enumerate(docs):
    for w in d:
        tf[j, pos[w]] += 1
idf = np.log((1 + len(docs)) / (1 + (tf > 0).sum(axis=0))) + 1
X = tf * idf
X = X / np.sqrt((X ** 2).sum(axis=1, keepdims=True))
def embed(q):
    v = np.zeros(len(vocab))
    for w in tokens(q):
        if w in pos:
            v[pos[w]] += 1
    v = v * idf
    n = np.sqrt((v ** 2).sum())
    return v / n if n > 0 else v
def retrieve(q, k=3):
    s = X @ embed(q); o = np.argsort(-s, kind="stable")[:k]
    return pd.DataFrame({"chunk_id": guidelines["chunk_id"].iloc[o].values,
                         "score": s[o].round(3), "text": guidelines["text"].iloc[o].values})
print({"chunks": X.shape[0], "terms": X.shape[1]})

# 2. Retrieve for Seán's question ------------------------------------------------
question = ("Mrs Byrne is on vancomycin and piperacillin-tazobactam, and her creatinine rose from "
            "1.0 to 1.4 mg/dL in 36 hours. Should we change the vancomycin, and how often should "
            "we check creatinine?")
hits = retrieve(question, k=3)
print(hits[["chunk_id", "score"]])

# 3. Assemble the grounded prompt (this is what would be sent to the model) ------
prompt = "\n".join(["You are the Ward 7 guideline assistant. Answer ONLY from the numbered passages below.",
                    "Cite the passage id for every statement. If the passages do not contain the answer, say so.", ""]
                   + [f"[{c}] {t}" for c, t in zip(hits["chunk_id"], hits["text"])] + ["", "Question: " + question])
print(prompt)

# 4. A crude grounding check of a frozen answer ----------------------------------
answer = ["Do not stop vancomycin without discussing it with the microbiology team.",
          "Consider an alternative beta-lactam to piperacillin-tazobactam and monitor creatinine daily.",
          "Mrs Byrne's risk of needing dialysis is about 10 percent."]
Xhits = X[guidelines["chunk_id"].isin(hits["chunk_id"]).to_numpy()]
support = [float((Xhits @ embed(s)).max()) for s in answer]
print(pd.DataFrame({"sentence": [s[:45] for s in answer], "best_support": np.round(support, 2),
                    "flag": [s < 0.2 for s in support]}))

# 5. Evaluation: accuracy and hallucination by configuration ---------------------
ev = pd.read_csv("data/ch19_eval.csv")
def wilson(k, n, z=1.96):
    p = k / n; c0 = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return c0 - h, c0 + h
rows = []
for cfg, d in ev.groupby("config"):
    lo, hi = wilson(d["hallucinated"].sum(), len(d))
    rows.append({"config": cfg, "accuracy": d["correct"].mean(), "hallucinated": d["hallucinated"].mean(),
                 "hall_lo": lo, "hall_hi": hi, "abstained": d["abstained"].mean()})
print(pd.DataFrame(rows).round(3))
print(ev.pivot_table(index="category", columns="config", values="correct", aggfunc="mean").round(2))

# 6. Calibration and selective answering -----------------------------------------
print((ev.groupby("config")["stated_conf"].mean() - ev.groupby("config")["correct"].mean()).round(3))
rag = ev[ev["config"] == "rag"]
thr = np.round(np.arange(0.5, 0.951, 0.05), 2)
print(pd.DataFrame({"threshold": thr,
                    "coverage": [(rag["stated_conf"] >= t).mean() for t in thr],
                    "accuracy": [rag.loc[rag["stated_conf"] >= t, "correct"].mean() for t in thr]}).round(3))

# 7. Rentosertib Phase 2a: a rough comparison from the published intervals --------
ci60, cipl = np.array([10.9, 185.9]), np.array([-116.1, 75.6])   # FVC change at 12 weeks (mL)
se = np.array([np.diff(ci60)[0], np.diff(cipl)[0]]) / (2 * 1.96)
diff_fvc = ci60.mean() - cipl.mean(); se_diff = np.sqrt((se ** 2).sum())
rough = {"difference": diff_fvc, "lower": diff_fvc - 1.96 * se_diff, "upper": diff_fvc + 1.96 * se_diff}
print({k: round(float(v), 1) for k, v in rough.items()})
