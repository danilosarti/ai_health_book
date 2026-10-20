"""
Ward 7 short clinical notes — synthetic corpus for Chapter 16 (language models and embeddings).

Every note is generated from a small grammar of templates, so the corpus describes no real
person and its statistical regularities are known by construction:

* drugs of the same class appear in the same kinds of context (what an embedding should pick up);
* each laboratory marker goes with one organ and one specialty (creatinine-kidney-nephrology,
  bilirubin-liver-hepatology, troponin-heart-cardiology, lipase-pancreas-gastroenterology);
* staff roles are followed ("the nurse says she ...") by "she" or "he" with deliberately unequal frequencies, mimicking the
  skew found in real historical text (so that the embeddings inherit a bias we can measure);
* "meropenem" never follows "started" (a gap that a count-based model cannot fill);
* some notes say which drug was started and, a clause or two later, ask to check the level of
  that same drug (a long-range dependence that a fixed window cannot see).

Tokens are lower-case words and the punctuation tokens "," "." ":" separated by single spaces,
so splitting on spaces recovers them. Output: data/ward7_notes.csv (note_id, set, text),
with 90 % of notes in the training set and 10 % in the test set.
Run from the book folder: python3 scripts/simulate/sim_ch16_ward7_notes.py
"""
import csv
import numpy as np

rng = np.random.default_rng(1607)
N_NOTES = 8000

ABX = ["vancomycin", "gentamicin", "piperacillin", "meropenem", "ceftriaxone"]
ANALG = ["paracetamol", "morphine", "oxycodone"]
DIUR = ["furosemide", "bumetanide"]
PRESS = ["noradrenaline", "vasopressin"]
ANTICOAG = ["heparin", "apixaban"]
INDICATION = {**{d: ["sepsis", "infection", "pneumonia"] for d in ABX},
              **{d: ["pain"] for d in ANALG},
              **{d: ["fluid overload", "oedema"] for d in DIUR},
              **{d: ["hypotension", "shock"] for d in PRESS},
              **{d: ["clot prevention"] for d in ANTICOAG},
              "insulin": ["high glucose"]}
ALL_DRUGS = list(INDICATION)
STARTABLE = [d for d in ALL_DRUGS if d != "meropenem"]          # meropenem never follows "started"
LEVEL_DRUGS = ["vancomycin", "gentamicin"]                       # therapeutic drug monitoring
RENAL_ADJUST = ["vancomycin", "gentamicin", "piperacillin", "meropenem", "apixaban"]

SYMPTOMS = ["nausea", "vomiting", "fatigue", "dizziness", "breathlessness", "fever",
            "confusion", "headache"]
SITES = ["chest", "abdominal", "back", "leg"]
TIMES = ["this morning", "last night", "yesterday", "overnight"]

ORGANS = [("creatinine", "kidney", "nephrology"), ("bilirubin", "liver", "hepatology"),
          ("troponin", "heart", "cardiology"), ("lipase", "pancreas", "gastroenterology")]
CONDITIONS = ["aki", "sepsis", "bleeding", "infection", "delirium"]
# probability that the pronoun after the role is "she" (deliberately skewed)
ROLES = {"nurse": 0.88, "physiotherapist": 0.75, "pharmacist": 0.5, "doctor": 0.35,
         "consultant": 0.2, "surgeon": 0.12}
ACTIONS = ["will recheck later", "updated the family", "adjusted the plan",
           "will review tomorrow"]
PLANS = ["repeat bloods tomorrow", "review in the morning", "continue current treatment",
         "discuss with family", "monitor urine output"]


def pick(xs):
    return xs[rng.integers(len(xs))]


def symptom_clause():
    r = rng.random()
    if r < 0.35:
        what = pick(SITES) + " pain"
    else:
        what = pick(SYMPTOMS)
    verb = pick(["patient reports", "patient reports", "patient denies", "patient complains of"])
    s = f"{verb} {what}"
    if rng.random() < 0.4:
        s += " since " + pick(TIMES)
    return s


def drug_clause():
    r = rng.random()
    if r < 0.45:
        d = pick(STARTABLE)
        return f"started {d} for {pick(INDICATION[d])}"
    if r < 0.65:
        d = pick(ALL_DRUGS)
        return f"{d} given for {pick(INDICATION[d])}"
    if r < 0.8:
        d = pick(RENAL_ADJUST)
        return f"{d} dose adjusted for kidney function"
    if r < 0.9:
        d = pick([x for x in ABX if x != "meropenem"])
        return f"switched from {d} to meropenem"
    d = pick(LEVEL_DRUGS)
    return f"{d} level checked"


def organ_clause():
    marker, organ, spec = ORGANS[rng.integers(len(ORGANS))]
    r = rng.random()
    if r < 0.3:
        return f"{marker} rising , {organ} function worsening"
    if r < 0.5:
        return f"{marker} stable , {organ} function improving"
    if r < 0.7:
        return f"{spec} review requested for {organ} concerns"
    if r < 0.85:
        return f"{marker} raised , {spec} informed"
    return f"{organ} function normal , {marker} within range"


def negation_clause():
    c = pick(CONDITIONS)
    return pick([f"no evidence of {c}", f"no signs of {c}", f"{c} ruled out", f"{c} confirmed"])


def staff_clause():
    role = pick(list(ROLES))
    pron = "she" if rng.random() < ROLES[role] else "he"
    return f"the {role} says {pron} {pick(ACTIONS)}"


def plan_clause():
    return "plan : " + pick(PLANS)


def long_range_note():
    d = pick(LEVEL_DRUGS)
    fillers = [symptom_clause() for _ in range(rng.integers(0, 3))]
    middle = "".join(f" , {f}" for f in fillers)
    return f"started {d} today{middle} , check {d} level tomorrow ."


GENERATORS = [(symptom_clause, 0.22), (drug_clause, 0.22), (organ_clause, 0.18),
              (negation_clause, 0.10), (staff_clause, 0.20), (plan_clause, 0.08)]
gens, w = zip(*GENERATORS)
w = np.array(w) / sum(w)


def note():
    if rng.random() < 0.08:
        return long_range_note()
    k = 1 if rng.random() < 0.6 else 2
    clauses = [gens[rng.choice(len(gens), p=w)]() for _ in range(k)]
    return " . ".join(clauses) + " ."


rows = []
for i in range(N_NOTES):
    rows.append({"note_id": f"N{i + 1:04d}", "set": "test" if rng.random() < 0.10 else "train",
                 "text": note()})

with open("data/ward7_notes.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=["note_id", "set", "text"])
    wr.writeheader()
    wr.writerows(rows)

ntok = sum(len(r["text"].split()) for r in rows)
vocab = sorted({t for r in rows for t in r["text"].split()})
print(f"{len(rows)} notes, {ntok} tokens, vocabulary {len(vocab)}")
