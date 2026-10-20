# =====================================================================
# Chapter 15 · precomputed results for the text-as-data chapter
# Reads data/clinical_notes.csv and data/negation_sentences.csv (made by
# scripts/simulate/sim_ch15_text.py) and writes small CSV files to
# data/precomputed/ch15/:
#   negex_grid.csv            accuracy of NegEx-lite for every setting of the widget
#   model_comparison.csv      five ways of flagging adverse drug reactions, test set
#   test_predictions.csv      test-set probabilities of the chosen classifier
#   top_terms.csv             largest positive and negative coefficients
#   tonight_predictions.csv   probabilities for Aoife's, Tomás's and Priya's notes
# Light: a few seconds on one CPU.
# Usage (from the book folder): python3 scripts/precompute/ch15_text.py
# =====================================================================
import itertools
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = "data/precomputed/ch15"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------
# NegEx-lite (same rules as the widget in ch15.qmd and the Lab)
# ---------------------------------------------------------------------
PRE = ["no evidence of", "negative for", "absence of", "rule out", "denies", "denied",
       "without", "not", "no"]
POST = ["was ruled out", "ruled out", "is absent", "excluded"]
PSEUDO = ["no increase", "no change", "not only", "no further"]
TERM = ["but", "however", "although", "except", ";"]


def tokens(s):
    return re.findall(r"[a-z0-9]+|;", s.lower())


def find(tok, phrase, start=0):
    p = phrase.split()
    for i in range(start, len(tok) - len(p) + 1):
        if tok[i:i + len(p)] == p:
            return i
    return -1


def negex(sentence, finding, window=5, use_pseudo=True, use_term=True, use_post=True):
    tok = tokens(sentence)
    ts = find(tok, finding)
    if ts < 0:
        return None
    te = ts + len(finding.split()) - 1
    i = 0
    while i < ts:
        if use_pseudo and any(find(tok[i:], q) == 0 for q in PSEUDO):
            i += 2
            continue
        hit = next((t for t in PRE if find(tok[i:], t) == 0), None)
        if hit:
            e = i + len(hit.split()) - 1
            if e < ts and ts - e <= window and not (use_term and any(x in TERM for x in tok[e + 1:ts])):
                return True
            i = e + 1
            continue
        i += 1
    if use_post:
        for t in POST:
            s = find(tok, t, te + 1)
            if s > te and s - te <= window and not (use_term and any(x in TERM for x in tok[te + 1:s])):
                return True
    return False


neg = pd.read_csv("data/negation_sentences.csv")
grid = []
for w, p, t, q in itertools.product(range(1, 11), [0, 1], [0, 1], [0, 1]):
    pred = [negex(s, f, w, bool(p), bool(t), bool(q)) for s, f in zip(neg.sentence, neg.finding)]
    acc = np.mean([int(bool(a)) == b for a, b in zip(pred, neg.negated)])
    grid.append(dict(window=w, pseudo=p, termination=t, post=q, accuracy=round(acc, 4)))
grid = pd.DataFrame(grid)
grid.to_csv(f"{OUT}/negex_grid.csv", index=False)
print("keyword only:", round(1 - neg.negated.mean(), 3),
      "| default (5, all on):", grid.query("window==5 and pseudo==1 and termination==1 and post==1").accuracy.item(),
      "| best:", grid.accuracy.max())

# ---------------------------------------------------------------------
# Adverse-drug-reaction classifier
# ---------------------------------------------------------------------
notes = pd.read_csv("data/clinical_notes.csv", keep_default_na=False)
lab = notes[notes.set.isin(["train", "test"])].copy()
lab["adr"] = lab.adr.astype(int)
train, test = lab[lab.set == "train"], lab[lab.set == "test"]
tonight = notes[notes.set == "tonight"]

REACTION = ["rash", "hives", "urticaria", "itching", "itchy", "nausea", "vomiting", "diarrhoea",
            "neutropenia", "neuropathy", "mucositis", "ringing in the ears", "reaction",
            "nephrotoxicity", "constipation", "headache", "hiccups"]


def sentences(text):
    return [x for x in re.split(r"(?<=[.;])\s+", text) if x]


def keyword_rule(text, use_negex=False):
    for s in sentences(text):
        for f in REACTION:
            if find(tokens(s), f) >= 0:
                if not use_negex or not negex(s, f):
                    return 1
    return 0


def mark_negations(text):
    """Replace every negated reaction word by NEG_word, so 'no rash' and 'rash' differ."""
    out = []
    for s in sentences(text):
        for f in REACTION:
            if find(tokens(s), f) >= 0 and negex(s, f):
                s = re.sub(r"\b" + re.escape(f) + r"\b", "NEG_" + f.replace(" ", "_"), s, flags=re.I)
        out.append(s)
    return " ".join(out)


def metrics(y, score, thr=0.5):
    pred = (np.asarray(score) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return dict(sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                ppv=tp / (tp + fp) if tp + fp else np.nan, tp=tp, fp=fp, fn=fn, tn=tn)


rows = []
y = test.adr.values
for name, use in [("Keyword list", False), ("Keyword list + NegEx-lite", True)]:
    sc = np.array([keyword_rule(t, use) for t in test.text])
    rows.append(dict(method=name, auc=np.nan, ap=np.nan, **metrics(y, sc)))


def fit_lr(train_text, test_text, ngram=(1, 1)):
    vec = TfidfVectorizer(ngram_range=ngram, min_df=2)
    Xtr = vec.fit_transform(train_text)
    clf = LogisticRegression(C=10, max_iter=2000)
    clf.fit(Xtr, train.adr.values)
    return vec, clf, clf.predict_proba(vec.transform(test_text))[:, 1]


fits = {}
for name, ngram, prep in [("TF-IDF words + logistic regression", (1, 1), False),
                          ("TF-IDF words and word pairs + logistic regression", (1, 2), False),
                          ("TF-IDF words, negations marked + logistic regression", (1, 1), True)]:
    tr = train.text.map(mark_negations) if prep else train.text
    te = test.text.map(mark_negations) if prep else test.text
    vec, clf, p = fit_lr(tr, te, ngram)
    fits[name] = (vec, clf, p, prep)
    rows.append(dict(method=name, auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                     **metrics(y, p)))
cmp_ = pd.DataFrame(rows)
cmp_.round(3).to_csv(f"{OUT}/model_comparison.csv", index=False)
print(cmp_.round(3).to_string())

CHOSEN = "TF-IDF words and word pairs + logistic regression"
vec, clf, p, prep = fits[CHOSEN]
pd.DataFrame(dict(doc_id=test.doc_id, source=test.source, adr=test.adr, p=np.round(p, 4),
                  text=test.text)).to_csv(f"{OUT}/test_predictions.csv", index=False)
coef = pd.DataFrame(dict(term=vec.get_feature_names_out(), coef=clf.coef_[0]))
top = pd.concat([coef.nlargest(12, "coef"), coef.nsmallest(8, "coef")])
top.round(3).to_csv(f"{OUT}/top_terms.csv", index=False)
tt = tonight.text.map(mark_negations)
pd.DataFrame(dict(doc_id=tonight.doc_id, name=["Aoife", "Tomás", "Priya"],
                  p=np.round(clf.predict_proba(vec.transform(tt))[:, 1], 3))).to_csv(
    f"{OUT}/tonight_predictions.csv", index=False)
print(pd.read_csv(f"{OUT}/tonight_predictions.csv"))
print(top.round(2).to_string())
