# =====================================================================
# Chapter 18: data for the "guideline assistant" of Ward 7.
#
# 1. data/ch18_guidelines.csv: a small corpus of ILLUSTRATIVE local
#    guideline paragraphs ("chunks") written for the book. They are a
#    teaching corpus for retrieval, NOT clinical guidance. Most of them
#    concern Ward 7 (acute kidney injury, nephrotoxic drugs, sepsis);
#    a few come from the other running examples (ONCO-1, CellWorks,
#    HomeBeat) so that retrieval has to pick the right document.
# 2. data/ch18_eval.csv: a SIMULATED evaluation of the assistant. The
#    same 150 staff questions are answered by two configurations
#    (model_only: the language model alone; rag: the model with
#    retrieval from the corpus), and each answer is graded as if by two
#    pharmacists. Columns:
#      question_id, category (dosing, monitoring, interaction,
#      out_of_scope), config (model_only, rag),
#      correct (1 = response judged correct; for out-of-scope questions
#      the correct response is to say the guidelines do not cover it),
#      abstained (1 = the assistant said it could not answer),
#      hallucinated (1 = the response contains at least one claim not
#      supported by the guidelines), stated_conf (the confidence the
#      assistant stated for its response, 0-1).
#    The numbers are invented to illustrate typical patterns; they are
#    not results of a real system.
# Usage (from the book folder): python3 scripts/simulate/sim_ch18_assistant.py
# =====================================================================
import csv
import numpy as np

CHUNKS = [
 # --- Ward 7: AKI prevention bundle -----------------------------------------
 ("AKI-1", "Ward 7 AKI prevention bundle", "Scope",
  "This guideline applies to adult patients in the intensive care unit who are at risk of acute kidney injury. The risk of AKI is assessed on admission and reviewed every day by the intensivist and the ward pharmacist."),
 ("AKI-2", "Ward 7 AKI prevention bundle", "Definition of AKI",
  "Acute kidney injury is defined by the KDIGO criteria: a rise in serum creatinine of at least 0.3 mg/dL within 48 hours, or a rise to at least 1.5 times baseline within 7 days, or urine output below 0.5 mL/kg/h for 6 hours."),
 ("AKI-3", "Ward 7 AKI prevention bundle", "Bundle components",
  "The kidney-protective bundle has five components: review and pause nephrotoxic drugs where an alternative exists, optimise volume status, keep mean arterial pressure at or above 65 mmHg, avoid hyperglycaemia, and monitor creatinine and urine output closely."),
 ("AKI-4", "Ward 7 AKI prevention bundle", "Monitoring while the bundle is active",
  "While the bundle is active, serum creatinine is measured every 12 hours and urine output is recorded every hour. The nurse in charge escalates to the intensivist if urine output falls below 0.5 mL/kg/h for two consecutive hours."),
 ("AKI-5", "Ward 7 AKI prevention bundle", "When to start the bundle",
  "Start the bundle when the estimated risk of AKI in the next 48 hours is 25 % or higher, or when creatinine has already risen by 0.3 mg/dL or more. The decision and the risk estimate used are documented in the notes."),
 ("AKI-6", "Ward 7 AKI prevention bundle", "When to stop the bundle",
  "The bundle is reviewed after 48 hours. It can be stopped when creatinine is stable, urine output is adequate and the estimated risk has fallen below the starting threshold. Paused drugs are restarted one at a time."),
 ("AKI-7", "Ward 7 AKI prevention bundle", "Escalation to nephrology",
  "Contact the nephrology team for KDIGO stage 3 acute kidney injury, potassium above 6.0 mmol/L despite treatment, severe metabolic acidosis, or fluid overload that does not respond to diuretics. These may be indications for renal replacement therapy."),
 ("AKI-8", "Ward 7 AKI prevention bundle", "Nephrotoxic drugs to review",
  "Nephrotoxic drugs to review include non-steroidal anti-inflammatory drugs, aminoglycosides, vancomycin, iodinated contrast, ACE inhibitors and angiotensin receptor blockers. Diuretics and metformin are also reviewed in patients with hypotension or rising creatinine."),
 # --- Ward 7: vancomycin -------------------------------------------------------
 ("VAN-1", "Ward 7 antimicrobial guideline: vancomycin", "Loading dose",
  "Vancomycin is started with a loading dose based on actual body weight. The dose is prescribed by the medical team and checked by the ward pharmacist before the first administration."),
 ("VAN-2", "Ward 7 antimicrobial guideline: vancomycin", "Maintenance dose and kidney function",
  "The maintenance dose and interval of vancomycin depend on kidney function. The pharmacist estimates creatinine clearance with the Cockcroft-Gault equation and recommends the regimen; estimates are unreliable when creatinine is changing quickly."),
 ("VAN-3", "Ward 7 antimicrobial guideline: vancomycin", "Therapeutic drug monitoring",
  "Vancomycin dosing is guided by the area under the concentration-time curve (AUC), with a target AUC of 400 to 600 mg h/L. Monitoring by trough level alone is no longer recommended for serious infections."),
 ("VAN-4", "Ward 7 antimicrobial guideline: vancomycin", "Vancomycin when creatinine is rising",
  "If creatinine rises during vancomycin treatment, do not stop vancomycin without discussing it with the microbiology team. The pharmacist reviews the dose and interval within 24 hours and a new vancomycin level is taken."),
 ("VAN-5", "Ward 7 antimicrobial guideline: vancomycin", "Combination with piperacillin-tazobactam",
  "The combination of vancomycin with piperacillin-tazobactam has been associated with a higher rate of acute kidney injury than vancomycin with other beta-lactams. Consider an alternative beta-lactam and monitor creatinine daily."),
 # --- Ward 7: other drugs and procedures ---------------------------------------
 ("GEN-1", "Ward 7 antimicrobial guideline: gentamicin", "Dosing",
  "Gentamicin is given once daily with an extended interval. Levels are taken according to the local nomogram and the course is limited to 72 hours whenever possible."),
 ("GEN-2", "Ward 7 antimicrobial guideline: gentamicin", "Gentamicin and kidney injury",
  "Avoid gentamicin in patients with acute kidney injury or rising creatinine unless there is no suitable alternative, and only after discussion with the microbiology team."),
 ("CON-1", "Ward 7 guideline: iodinated contrast", "Contrast and kidney risk",
  "Iodinated contrast should not be withheld when a scan is essential for diagnosis. Ensure the patient is not hypovolaemic before the scan. The local guideline does not recommend N-acetylcysteine for kidney protection."),
 ("SEP-1", "Ward 7 sepsis guideline", "Lactate",
  "Measure lactate in every patient with suspected sepsis. Repeat the measurement within 2 to 4 hours if the first lactate is above 2 mmol/L."),
 ("SEP-2", "Ward 7 sepsis guideline", "Fluids",
  "Give 30 mL/kg of intravenous crystalloid within the first 3 hours to patients with sepsis-induced hypotension or a lactate of 4 mmol/L or more, reassessing the response after each bolus."),
 ("SEP-3", "Ward 7 sepsis guideline", "Blood pressure target and vasopressors",
  "Target a mean arterial pressure of 65 mmHg. Noradrenaline is the first-line vasopressor and can be started through a peripheral line while central access is obtained."),
 ("SEP-4", "Ward 7 sepsis guideline", "Antimicrobials",
  "Give antimicrobials within one hour for patients with possible septic shock or a high likelihood of sepsis, after taking blood cultures when this causes no delay."),
 ("MED-1", "Ward 7 guideline: medicines during acute illness", "Sick-day rules",
  "During acute illness with hypotension or rising creatinine, hold ACE inhibitors, angiotensin receptor blockers, diuretics, metformin and non-steroidal anti-inflammatory drugs. Restart them when the patient is eating, drinking and haemodynamically stable."),
 ("MED-2", "Ward 7 guideline: medicines during acute illness", "Hyperkalaemia",
  "For potassium above 6.5 mmol/L or ECG changes, give intravenous calcium to protect the heart, then insulin with glucose to shift potassium into cells, and review all drugs that raise potassium."),
 ("LAB-1", "Ward 7 guideline: kidney function tests", "Baseline creatinine",
  "The baseline creatinine is the lowest value in the previous three months. If no earlier value is available, the admission value is used and the AKI stage is reassessed when more results arrive."),
 ("LAB-2", "Ward 7 guideline: kidney function tests", "Creatinine lags behind injury",
  "Serum creatinine rises 24 to 48 hours after the kidney is injured, so a normal value does not exclude early injury. Falling urine output is often the earlier signal."),
 ("LAB-3", "Ward 7 guideline: kidney function tests", "Estimated GFR in unstable patients",
  "Equations that estimate glomerular filtration rate assume a steady state. They are not valid when kidney function is changing quickly, as in acute kidney injury; use them with caution for drug dosing."),
 ("DOC-1", "Ward 7 guideline: documentation", "Recording decisions",
  "Every decision to start, change or stop the bundle, and every dose change of a nephrotoxic drug, is recorded with the reason, the information used and the name of the person responsible."),
 # --- ONCO-1 --------------------------------------------------------------------
 ("ONC-1", "ONCO-1 trial: adverse event reporting", "Grading and reporting",
  "Adverse events in ONCO-1 are graded with the Common Terminology Criteria for Adverse Events. Serious adverse events are reported to the sponsor within 24 hours of the site becoming aware of them."),
 ("ONC-2", "ONCO-1 trial: adverse event reporting", "Kidney function during chemotherapy",
  "Creatinine is measured before every chemotherapy cycle. A rise of more than 50 % from baseline delays the cycle until the treating oncologist has reviewed the patient."),
 # --- CellWorks -----------------------------------------------------------------
 ("CEL-1", "CellWorks batch release procedure", "Potency test",
  "A batch can be released only if it passes the potency test. A batch that fails is quarantined and a deviation investigation is opened within five working days."),
 ("CEL-2", "CellWorks batch release procedure", "Temperature deviations",
  "Any temperature deviation during culture is recorded with its duration. Deviations longer than two hours are reviewed by quality assurance before the batch can be released."),
 # --- HomeBeat ------------------------------------------------------------------
 ("HOM-1", "HomeBeat study: data gaps", "Missing watch data",
  "If no data have been received from a participant's watch for 24 hours, the study team contacts the participant. Gaps are recorded with the reason given, so that analysts can judge why the data are missing."),
]

with open("data/ch18_guidelines.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["chunk_id", "document", "section", "text"])
    for row in CHUNKS:
        w.writerow(row)

# ------------------------------------------------------------------------------
rng = np.random.default_rng(1818)
n_q = 150
cats = rng.choice(["dosing", "monitoring", "interaction", "out_of_scope"], size=n_q,
                  p=[0.30, 0.30, 0.25, 0.15])
p_correct = {"model_only": {"dosing": 0.60, "monitoring": 0.66, "interaction": 0.64, "out_of_scope": 0.15},
             "rag":        {"dosing": 0.84, "monitoring": 0.92, "interaction": 0.80, "out_of_scope": 0.70}}
rows = []
for config in ["model_only", "rag"]:
    for q in range(n_q):
        c = cats[q]
        correct = int(rng.random() < p_correct[config][c])
        if c == "out_of_scope":
            abstained = correct                     # the right response is to abstain
            if not correct and config == "rag":
                abstained = 0
        else:
            # occasionally abstains on an answerable question (judged incorrect)
            abstained = int((not correct) and rng.random() < (0.05 if config == "model_only" else 0.30))
        if abstained:
            hall = 0
        elif correct:
            hall = int(rng.random() < (0.10 if config == "model_only" else 0.03))
        else:
            hall = int(rng.random() < (0.85 if config == "model_only" else 0.55))
        if config == "model_only":   # overconfident: stated confidence barely tracks correctness
            conf = 0.86 + 0.03 * correct + rng.normal(0, 0.05)
        else:                        # better calibrated
            conf = (0.66 + 0.20 * correct) + rng.normal(0, 0.09)
            if not correct and rng.random() < 0.25:   # some confidently wrong answers remain
                conf = 0.91 + rng.normal(0, 0.04)
        conf = float(np.clip(conf, 0.30, 0.99))
        rows.append([f"Q{q + 1:03d}", c, config, correct, abstained, hall, round(conf, 2)])

with open("data/ch18_eval.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["question_id", "category", "config", "correct", "abstained", "hallucinated", "stated_conf"])
    w.writerows(rows)
print("ch18_guidelines.csv:", len(CHUNKS), "chunks; ch18_eval.csv:", len(rows), "rows")
