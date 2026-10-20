# =====================================================================
# Chapter 15 · Text as data — a small, fully SYNTHETIC clinical corpus
# ---------------------------------------------------------------------
# Every sentence below is invented from templates. No real patient, note,
# report or medicine leaflet was used. Patient identifiers are those of the
# simulated Ward 7 (icu_aki.csv) and ONCO-1 (onco1.csv) datasets, so each
# note inherits the book-wide train/test split of its patient
# (data/splits/ward7_split.csv, data/splits/onco1_split.csv).
#
# Output
#   data/clinical_notes.csv      one row per document
#     doc_id     W7-0001 … (Ward 7 discharge summary), AE-0001 … (ONCO-1
#                adverse-event note), PIL-001 … (patient-leaflet sentence),
#                W7-AOIFE / W7-TOMAS / W7-PRIYA (tonight's admission notes)
#     source     ward7_discharge | onco1_ae | leaflet | ward7_tonight
#     patient_id patient of the note (empty for leaflets)
#     set        train | test (from the shared split files) | tonight | none
#     adr        1 = the note reports a suspected adverse drug reaction,
#                0 = it does not (empty for leaflets and tonight's notes)
#     drug       suspected drug when adr = 1, drug described by a leaflet
#     text       the document
#   data/negation_sentences.csv  short sentences for the negation widget
#     sent_id, sentence, finding (the term to look for),
#     negated (1 = the finding is denied / absent), pattern (why it is there)
#
# Usage (from the book folder): python3 scripts/simulate/sim_ch15_text.py
# =====================================================================
import csv
import numpy as np
import pandas as pd

rng = np.random.default_rng(1515)


def pick(x):
    return x[rng.integers(len(x))]


MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def a_date():
    """The same kind of fact written in four different ways (regex section)."""
    d, m = int(rng.integers(1, 29)), int(rng.integers(1, 13))
    style = rng.integers(4)
    if style == 0:
        return f"{d:02d}/{m:02d}/2026"
    if style == 1:
        return f"2026-{m:02d}-{d:02d}"
    if style == 2:
        return f"{d} {MONTHS[m - 1]} 2026"
    suf = "th" if 11 <= d <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
    return f"{d}{suf} of {MONTHS[m - 1]}"


# ---------------------------------------------------------------------
# 1. Ward 7 discharge summaries
# ---------------------------------------------------------------------
icu = pd.read_csv("data/icu_aki.csv")
w7split = pd.read_csv("data/splits/ward7_split.csv").set_index("patient_id")
w7 = icu.sample(n=450, random_state=1515).sort_values("patient_id")

REASONS = ["septic shock from a urinary source", "community-acquired pneumonia",
           "abdominal sepsis after bowel surgery", "sepsis of unknown source",
           "a severe soft-tissue infection of the leg", "pneumonia with septic shock"]
ANTIBIOTICS = [("vancomycin", ["1000 mg twice daily", "1500 mg twice daily", "1250 mg twice daily"]),
               ("piperacillin-tazobactam", ["4.5 g every 8 hours"]),
               ("meropenem", ["1 g every 8 hours"]),
               ("gentamicin", ["5 mg/kg once daily", "7 mg/kg once daily"]),
               ("amoxicillin-clavulanate", ["1.2 g every 8 hours"])]
SUPPORT = ["furosemide 40 mg", "paracetamol 1 g", "ondansetron 4 mg", "pantoprazole 40 mg",
           "enoxaparin 40 mg", "a noradrenaline infusion", "hydrocortisone 50 mg"]

# ---------------------------------------------------------------------
# The "event" sentence is built from parts, so the same words appear in
# reports of drug reactions and in notes that are not about one:
#   symptom  x  attribution (drug / not drug / unclear)  x  action
# Unclear attributions are labelled at random (probability 0.5): two
# pharmacists reading them could disagree. This is irreducible uncertainty.
# ---------------------------------------------------------------------
SYMPT_W7 = [("an itchy rash on the trunk", None), ("hives on the chest and arms", None),
            ("flushing and itching of the face", "vancomycin"), ("ringing in the ears", "gentamicin"),
            ("profuse diarrhoea", None), ("a low platelet count", None), ("confusion", None),
            ("a further rise in creatinine", "nephro"), ("nausea and vomiting", None),
            ("a pruritic maculopapular eruption", None), ("swelling of the lips", None)]
SYMPT_ONCO = ["grade {g} nausea", "grade 3 neutropenia", "tingling in both feet",
              "grade {g} mucositis", "vomiting after each infusion", "an itchy rash on the forearms",
              "painful red palms and soles", "grade {g} fatigue", "grade {g} diarrhoea",
              "febrile neutropenia", "loss of appetite", "mouth ulcers"]
ATTR = {
    1: ["probably related to {d}", "attributed to {d}", "thought to be a reaction to {d}",
        "considered an adverse effect of {d}", "likely drug-related", "a known side effect of {d}",
        "suspected to be caused by {d}", "related to {d}"],
    0: ["not related to {d}", "unrelated to treatment", "attributed to the infection",
        "present before {d} was started", "considered part of the underlying disease",
        "due to septic shock rather than any drug", "after a viral illness in the family",
        "not thought to be drug-related", "which settled before {d} was given"],
    -1: ["cause unclear", "relation to {d} uncertain", "possibly drug-related or due to the illness",
         "may be related to {d}", "a drug cause cannot be excluded", "drug effect or infection"],
}
ACTION = {1: ["{D} was stopped.", "Dose reduced.", "Treated with an antihistamine.", "{D} was stopped and an alternative given.",
              "Reported to the pharmacy team.", "Monitored closely."],
          0: ["No change to treatment.", "Monitored closely.", "Treated symptomatically.", "Continue current plan.",
              "Dose reduced."]}


def event_sentence(symptom, drug, kind, k):
    """kind: 1 drug-attributed, 0 not drug-attributed, -1 unclear."""
    attr = pick(ATTR[kind]).format(d=drug)
    act_kind = 1 if kind == 1 else (0 if kind == 0 else int(rng.random() < 0.5))
    act_kind = act_kind if rng.random() < 0.75 else 1 - act_kind
    act = pick(ACTION[act_kind]).format(D=drug[0].upper() + drug[1:])
    sym = symptom[0].upper() + symptom[1:]
    form = rng.integers(3)
    if form == 0:
        return f"{sym} on day {k}, {attr}. {act}"
    if form == 1:
        return f"Developed {symptom} on day {k}; {attr}. {act}"
    return f"{sym} since day {k} ({attr}). {act}"


EASY_NEG_W7 = ["No rash or other drug reactions.", "No known drug allergies.", "Skin intact, no rash.",
               "Denies itching.", "No adverse drug reactions during the admission.",
               "Tolerated all antibiotics without problems.", "No nausea or diarrhoea.",
               "Family history of penicillin allergy; no reaction during this admission.",
               "Drug levels within the target range throughout.",
               "No ringing in the ears or hearing change reported."]
EASY_NEG_ONCO = ["No new adverse events since the last visit.", "No nausea, vomiting or diarrhoea.",
                 "Denies tingling or numbness.", "No mouth ulcers.", "Blood counts within normal limits.",
                 "Appetite good; weight stable.", "No rash.", "Fall at home with a bruised knee."]


def label_of(kind):
    return kind if kind >= 0 else int(rng.random() < 0.5)


def w7_note(row):
    base = row.baseline_creat
    peak = base + max(row.delta_creatinine_48h, 0)
    aki = row.delta_creatinine_48h >= 0.3
    ab = [ANTIBIOTICS[i] for i in rng.choice(len(ANTIBIOTICS), size=2, replace=False)]
    s = [f"Admitted to ICU on {a_date()} with {pick(REASONS)}."]
    s.append(f"Treated with {ab[0][0]} {pick(ab[0][1])} and {ab[1][0]} {pick(ab[1][1])}, "
             f"with {pick(SUPPORT)} as needed.")
    if aki:
        s.append(f"Creatinine rose from {base:.1f} to {peak:.1f} mg/dL within 48 hours.")
    else:
        s.append(f"Creatinine remained stable at around {base:.1f} mg/dL.")
    adr, drug = 0, ""
    u = rng.random()
    if u < 0.62:                                  # an event is described
        kind = int(rng.choice([1, 0, -1], p=[0.45, 0.33, 0.22]))
        options = [x for x in SYMPT_W7 if x[1] != "nephro" or aki]
        symptom, hint = options[rng.integers(len(options))]
        names = [a[0] for a in ab]
        drug = hint if hint in names else ("vancomycin" if hint == "nephro" and "vancomycin" in names else pick(names))
        s.append(event_sentence(symptom, drug, kind, int(rng.integers(2, 7))))
        adr = label_of(kind)
        drug = drug if adr == 1 and kind == 1 else ("unclear" if adr == 1 else "")
    for t in rng.choice(EASY_NEG_W7, size=int(rng.integers(0, 3)), replace=False):
        s.append(t)
    s.append(pick([f"Follow-up in the renal clinic in {int(rng.integers(2, 9))} weeks.",
                   f"Recheck creatinine on {a_date()}.",
                   "Discharged to the ward.", "Stepped down to the ward with daily blood tests.",
                   f"GP to review medication on {a_date()}."]))
    return " ".join(s), adr, drug


rows = []
for k, r in enumerate(w7.itertuples(), 1):
    text, adr, drug = w7_note(r)
    rows.append(dict(doc_id=f"W7-{k:04d}", source="ward7_discharge", patient_id=r.patient_id,
                     set=w7split.loc[r.patient_id, "set"], adr=adr, drug=drug, text=text))

# ---------------------------------------------------------------------
# 2. ONCO-1 adverse-event notes
# ---------------------------------------------------------------------
onco = pd.read_csv("data/onco1.csv")
osplit = pd.read_csv("data/splits/onco1_split.csv").set_index("patient_id")
oc = pd.concat([onco[onco.treatment == "chemo"].sample(n=300, random_state=1515),
                onco[onco.treatment == "no_chemo"].sample(n=100, random_state=1515)]).sort_values("patient_id")


def onco_note(row):
    c = int(rng.integers(1, 7))
    s = [f"ONCO-1 review, cycle {c} day {int(rng.integers(1, 22))}."]
    s.append(f"Target lesion {row.post_tumor_mm:.0f} mm (baseline {row.baseline_tumor_mm:.0f} mm); "
             f"performance status {int(row.performance_score)}.")
    adr, drug = 0, ""
    chemo = row.treatment == "chemo"
    if rng.random() < (0.70 if chemo else 0.45):
        p = [0.50, 0.28, 0.22] if chemo else [0.15, 0.65, 0.20]
        kind = int(rng.choice([1, 0, -1], p=p))
        drug = pick(["chemotherapy", "the study drug", "treatment"]) if chemo else pick(["codeine", "ondansetron", "dexamethasone"])
        sym = pick(SYMPT_ONCO).format(g=int(rng.integers(1, 4)))
        s.append(event_sentence(sym, drug, kind, int(rng.integers(2, 15))))
        adr = label_of(kind)
        drug = drug if adr == 1 and kind == 1 else ("unclear" if adr == 1 else "")
    for t in rng.choice(EASY_NEG_ONCO, size=int(rng.integers(0, 3)), replace=False):
        s.append(t)
    s.append(pick(["Next review in three weeks.", "Continue current plan.",
                   f"Next CT scan on {a_date()}.", "Bloods before next cycle."]))
    return " ".join(s), adr, drug


for k, r in enumerate(oc.itertuples(), 1):
    text, adr, drug = onco_note(r)
    rows.append(dict(doc_id=f"AE-{k:04d}", source="onco1_ae", patient_id=r.patient_id,
                     set=osplit.loc[r.patient_id, "set"], adr=adr, drug=drug, text=text))

# ---------------------------------------------------------------------
# 3. Tonight's three Ward 7 patients (written by hand; label unknown)
# ---------------------------------------------------------------------
rows += [
    # Admission/progress notes written TONIGHT: they reveal no ICU outcome, no creatinine change and no
    # discharge (the outcomes of tonight's patients stay unknown throughout the book).
    dict(doc_id="W7-AOIFE", source="ward7_tonight", patient_id="W7-AOIFE", set="tonight", adr="", drug="",
         text=("Mrs Aoife Byrne, 71, admitted to ICU this morning with septic shock from a urinary source. "
               "Piperacillin-tazobactam 4.5 g every 8 hours started this afternoon; vancomycin 1000 mg loading dose given tonight. "
               "Mild itch on the forearms since the first antibiotic dose; cause unclear. No rash. "
               "Vancomycin level and bloods to be checked in the morning.")),
    dict(doc_id="W7-TOMAS", source="ward7_tonight", patient_id="W7-TOMAS", set="tonight", adr="", drug="",
         text=("Mr Tom\u00e1s Ruiz, 54, admitted to ICU today with community-acquired pneumonia. "
               "Amoxicillin-clavulanate 1.2 g every 8 hours started, with paracetamol 1 g as needed. "
               "No adverse drug reactions so far. Denies itching or rash. Review in the morning.")),
    dict(doc_id="W7-PRIYA", source="ward7_tonight", patient_id="W7-PRIYA", set="tonight", adr="", drug="",
         text=("Ms Priya Nair, 38, admitted to ICU tonight with pneumonia with septic shock. "
               "Vancomycin 1500 mg and gentamicin 5 mg/kg given, with a noradrenaline infusion. "
               "Flushing and itching of the face within an hour of starting vancomycin, probably related to vancomycin. "
               "Infusion slowed and treated with an antihistamine. Reported to the pharmacy team.")),
]

# ---------------------------------------------------------------------
# 4. Patient-leaflet sentences (invented; not copied from any real leaflet)
# ---------------------------------------------------------------------
LEAFLET = {
    "vancomycin": ["Vancomycin can affect your kidneys, so your doctor will check your blood regularly.",
                   "If vancomycin is given too quickly you may get flushing and itching of the face and neck.",
                   "Tell your nurse if you notice ringing in the ears while receiving vancomycin.",
                   "Vancomycin is an antibiotic given into a vein for serious infections."],
    "gentamicin": ["Gentamicin can damage the kidneys and the inner ear.",
                   "Tell your doctor straight away if you have ringing in the ears, dizziness or hearing loss while taking gentamicin.",
                   "Your blood levels of gentamicin will be measured to keep the dose safe."],
    "piperacillin-tazobactam": ["Piperacillin-tazobactam can cause diarrhoea; tell your doctor if it is severe or bloody.",
                                "Stop and get help at once if you develop a rash, swelling of the face or difficulty breathing after piperacillin-tazobactam.",
                                "Piperacillin-tazobactam belongs to the penicillin family of antibiotics."],
    "amoxicillin-clavulanate": ["Do not take amoxicillin-clavulanate if you are allergic to penicillin.",
                                "A skin rash is a common side effect of amoxicillin-clavulanate.",
                                "Amoxicillin-clavulanate may cause nausea; take it at the start of a meal."],
    "meropenem": ["Meropenem can rarely cause fits, especially in people with kidney problems.",
                  "Tell your doctor if you get hives or itching after meropenem."],
    "ondansetron": ["Ondansetron prevents nausea and vomiting caused by chemotherapy.",
                    "Headache and constipation are common side effects of ondansetron."],
    "furosemide": ["Furosemide makes you pass more urine; you may feel dizzy when you stand up.",
                   "Your kidney function and salts will be checked while you take furosemide."],
    "codeine": ["Codeine often causes constipation; drink plenty of fluids.",
                "Codeine can make you drowsy; do not drive if you feel sleepy."],
    "chemotherapy": ["Chemotherapy can lower your white blood cells and increase the risk of infection.",
                     "Call the hospital at once if your temperature is 38 degrees or higher during chemotherapy.",
                     "Nausea after chemotherapy can usually be prevented with medicines.",
                     "Some chemotherapy drugs cause tingling or numbness in the hands and feet.",
                     "Mouth ulcers can occur a week after chemotherapy; use a soft toothbrush."],
    "enoxaparin": ["Enoxaparin is injected under the skin to prevent blood clots.",
                   "Tell your doctor about unusual bruising or bleeding while using enoxaparin."],
    "dexamethasone": ["Dexamethasone can raise blood sugar and make it hard to sleep."],
    "noradrenaline": ["Noradrenaline is given through a drip to raise very low blood pressure in intensive care."],
    "hydrocortisone": ["Hydrocortisone is a steroid that may be given in septic shock."],
    "pantoprazole": ["Pantoprazole reduces stomach acid; long use can lower magnesium levels."],
    "paracetamol": ["Do not take more than 4 g of paracetamol in 24 hours.",
                    "Paracetamol in overdose can cause serious liver damage."],
}
k = 0
for drug, sents in LEAFLET.items():
    for t in sents:
        k += 1
        rows.append(dict(doc_id=f"PIL-{k:03d}", source="leaflet", patient_id="", set="none",
                         adr="", drug=drug, text=t))

notes = pd.DataFrame(rows, columns=["doc_id", "source", "patient_id", "set", "adr", "drug", "text"])
notes.to_csv("data/clinical_notes.csv", index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

# ---------------------------------------------------------------------
# 5. Negation sentences (the NegEx-lite widget)
# ---------------------------------------------------------------------
FIND = ["rash", "nausea", "itching", "fever", "diarrhoea", "hives", "chest pain", "tinnitus"]
neg = []


def add(sentence, finding, negated, pattern):
    neg.append(dict(sentence=sentence, finding=finding, negated=negated, pattern=pattern))


OTHERS = ["fever", "cough", "headache", "chest pain", "vomiting", "dizziness", "nausea", "rash",
          "diarrhoea", "itching"]
for f in FIND:
    F = f.capitalize()
    add(pick([f"{F} since day 3.", f"Patient reports {f} after the infusion.",
              f"New {f} on the second day of treatment."]), f, 0, "affirmed")
    add(pick([f"Developed {f} after vancomycin.", f"Marked {f} this morning.",
              f"{F} improving with treatment."]), f, 0, "affirmed")
    add(pick([f"No {f}.", f"No {f} today."]), f, 1, "negation trigger before the finding")
    add(pick([f"Denies {f}.", f"Patient denies any {f}."]), f, 1, "negation trigger before the finding")
    add(pick([f"Negative for {f}.", f"Without {f} after the second dose."]), f, 1, "negation trigger before the finding")
    add(f"No evidence of {f}.", f, 1, "negation trigger before the finding")
    # long list: the finding is 6 or more words after the trigger
    o = [x for x in OTHERS if x != f]
    lst = list(rng.choice(o, size=5, replace=False))
    add(f"No {', '.join(lst)} or {f}.", f, 1, "long list: finding far from the trigger")
    # termination: a second clause changes the subject
    other = pick([x for x in OTHERS if x != f])
    add(pick([f"No {other} but new {f} on the trunk.", f"Denies {other}, however reports {f}.",
              f"No {other}; {f} since this morning."]), f, 0, "scope ends at 'but', 'however' or ';'")
    # pseudo-negation: looks like a negation but is not
    add(pick([f"No increase in {f} since yesterday.", f"No change in {f} overnight.",
              f"Not only {f} but also tiredness."]), f, 0, "pseudo-negation")
    # post-negation: the trigger comes after the finding
    add(pick([f"{F} was ruled out.", f"Drug-related {f} was ruled out by the team."]), f, 1,
        "negation trigger after the finding")
    # negation far before the finding, separated by a semicolon
    add(f"No known drug allergies; developed {f} on day 3.", f, 0, "scope ends at 'but', 'however' or ';'")

# genuinely hard sentences NegEx-lite gets wrong whatever the settings
add("Rash-free since stopping the antibiotic.", "rash", 1, "hard: no trigger word")
add("Free of nausea after switching the antiemetic.", "nausea", 1, "hard: no trigger word")
add("Fever: none since Tuesday.", "fever", 1, "hard: unusual wording")
add("Cannot rule out a drug rash.", "rash", 0, "hard: uncertainty, not negation")
add("Itching resolved within a day.", "itching", 1, "hard: finding no longer present")
negdf = pd.DataFrame(neg)
negdf.insert(0, "sent_id", [f"S{i:03d}" for i in range(1, len(negdf) + 1)])
negdf.to_csv("data/negation_sentences.csv", index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")

print(notes.groupby(["source", "set"]).size())
print(notes[notes.adr != ""].groupby("source").adr.mean())
print(len(negdf), "negation sentences;", negdf.negated.mean().round(2), "negated")
