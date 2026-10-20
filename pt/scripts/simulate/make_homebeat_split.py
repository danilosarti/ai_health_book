"""Fixed HomeBeat train/test split and CV folds (companion of make_splits.R).

Writes data/splits/homebeat_split.csv: 400 training + 100 test participants, training
participants assigned to 5 folds of 80.

Why Python and not make_splits.R: Chapters 8 and 19 already split HomeBeat with
    np.random.default_rng(2026).permutation(500)[:400]   -> training participants
and their published numbers depend on that exact NumPy permutation (R's sample() cannot
reproduce it). This script writes that permutation to disk once, so that every chapter and
the model ledger read the same file instead of re-drawing it.

Columns
    participant_id  as in data/homebeat.csv
    set             "train" / "test"
    fold            1-5 for training rows (position in the permutation modulo 5, plus 1),
                    empty for test rows
    perm_order      0-based position of the participant in the permutation. Chapter 19 takes the
                    training rows IN PERMUTATION ORDER (hb.iloc[perm[:400]]); sorting the training
                    rows by perm_order reproduces that order exactly.

Usage (from the book folder):  python3 scripts/simulate/make_homebeat_split.py
"""
import numpy as np
import pandas as pd

SEED = 2026
hb = pd.read_csv("data/homebeat.csv")
perm = np.random.default_rng(SEED).permutation(len(hb))
order = np.empty(len(hb), int)
order[perm] = np.arange(len(hb))
is_tr = order < 400
fold = np.where(is_tr, order % 5 + 1, 0)
out = pd.DataFrame({"participant_id": hb["participant_id"], "set": np.where(is_tr, "train", "test"),
                    "fold": pd.array(np.where(is_tr, fold, pd.NA), dtype="Int64"), "perm_order": order})
out.to_csv("data/splits/homebeat_split.csv", index=False)

# --- checks: the file reproduces the permutation used by Chapters 8 and 19 ---
chk = np.zeros(len(hb), bool); chk[perm[:400]] = True
assert (chk == is_tr).all() and is_tr.sum() == 400
assert (out.loc[is_tr, "fold"].value_counts() == 80).all()
assert (out.sort_values("perm_order").index.to_numpy() == perm).all()
print("HomeBeat:", int(is_tr.sum()), "train /", int((~is_tr).sum()), "test; folds of",
      out.loc[is_tr, "fold"].value_counts().iloc[0])
