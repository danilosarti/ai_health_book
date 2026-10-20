# Laboratorio del capitolo 15
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Caricare il corpus -----------------------------------------------------------
import re
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
notes = pd.read_csv("data/clinical_notes.csv", keep_default_na=False)
print(pd.crosstab(notes["source"], notes["set"]))

# 2. Espressioni regolari: dosi e date -----------------------------------------------
w7 = notes[(notes["source"] == "ward7_discharge") & (notes["set"] != "tonight")]
print(Counter(d for t in w7["text"] for d in re.findall(r"[0-9]+ mg", t)).most_common(6))
print({"iso_dates": int(w7["text"].str.contains(r"\b\d{4}-\d{2}-\d{2}\b").sum()),
       "slash_dates": int(w7["text"].str.contains(r"\b\d{2}/\d{2}/\d{4}\b").sum())})

# 3. NegEx-lite ------------------------------------------------------------------------
PRE = ["no evidence of", "negative for", "absence of", "rule out", "denies", "denied", "without", "not", "no"]
POST = ["was ruled out", "ruled out", "is absent", "excluded"]
PSEUDO = ["no increase", "no change", "not only", "no further"]
TERM = ["but", "however", "although", "except", ";"]

def find(tok, phrase, start=0):
    p = phrase.split()
    for i in range(start, len(tok) - len(p) + 1):
        if tok[i:i + len(p)] == p:
            return i
    return -1

def negex(sentence, finding, window=8):
    tok = re.findall(r"[a-z0-9]+|;", sentence.lower())
    ts = find(tok, finding)
    if ts < 0:
        return None
    te = ts + len(finding.split()) - 1
    blocked = lambda a, b: any(x in TERM for x in tok[a:b])
    i = 0
    while i < ts:
        if any(find(tok[i:], q) == 0 for q in PSEUDO):
            i += 2
            continue
        hit = next((t for t in PRE if find(tok[i:], t) == 0), None)
        if hit:
            e = i + len(hit.split()) - 1
            if e < ts and ts - e <= window and not blocked(e + 1, ts):
                return True
            i = e + 1
            continue
        i += 1
    for t in POST:
        s = find(tok, t, te + 1)
        if s > te and s - te <= window and not blocked(te + 1, s):
            return True
    return False

neg = pd.read_csv("data/negation_sentences.csv")
neg["pred"] = [bool(negex(s, f)) for s, f in zip(neg["sentence"], neg["finding"])]
print("Accuratezza di NegEx-lite:", round((neg["pred"] == (neg["negated"] == 1)).mean(), 3))
print(neg.loc[neg["pred"] != (neg["negated"] == 1), ["sentence", "finding", "negated"]])

# 4. Matrice TF-IDF (vocabolario dalle note di addestramento) ----------------------------
lab = notes[notes["set"].isin(["train", "test"])].copy()
lab["adr"] = lab["adr"].astype(int)
train, test = lab[lab["set"] == "train"], lab[lab["set"] == "test"]
vec = TfidfVectorizer(min_df=2)                      # parole in almeno due note di addestramento
X_train = vec.fit_transform(train["text"])
X_test = vec.transform(test["text"])
print("vocabolario:", len(vec.vocabulary_))

# 5. Regressione logistica ridge -----------------------------------------------------------
clf = LogisticRegression(C=10, max_iter=2000).fit(X_train, train["adr"])
p = clf.predict_proba(X_test)[:, 1]
print("AUC sul test:", round(roc_auc_score(test["adr"], p), 3))
coef = pd.Series(clf.coef_[0], index=vec.get_feature_names_out())
print(coef.sort_values(ascending=False).head(8).round(2))

# 6. Una coda di revisione -----------------------------------------------------------------
yt = test["adr"].to_numpy(); t_low, t_high = 0.10, 0.65
print({"cleared": int((p < t_low).sum()), "missed": int(((p < t_low) & (yt == 1)).sum()),
       "review": int(((p >= t_low) & (p < t_high)).sum()),
       "flagged": int((p >= t_high).sum()), "false_reports": int(((p >= t_high) & (yt == 0)).sum())})
