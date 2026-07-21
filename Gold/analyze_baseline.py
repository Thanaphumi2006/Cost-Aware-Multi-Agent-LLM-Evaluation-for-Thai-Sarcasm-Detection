# -*- coding: utf-8 -*-
"""Break the baseline results down by 'item origin' to see self-selection bias without another model

Idea: the sarcastic/not items in gold come from two sources GPT-4o "saw" differently
  - keyword group : the first 102-item gold  -> keyword-filtered, GPT-4o never touched it
  - harvest group : the 25 newly added       -> GPT-4o itself selected them as "probably sarcastic"
If GPT-4o does clearly better on the group it selected = evidence of self-selection bias

Run: python analyze_baseline.py
"""

import os
import random
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, "baseline_preds_gpt.csv")
BACKUP = os.path.join(HERE, "gold_backup.csv")


def prf(y_true, y_pred, pos="1"):
    tp = sum(t == pos and p == pos for t, p in zip(y_true, y_pred))
    fp = sum(t != pos and p == pos for t, p in zip(y_true, y_pred))
    fn = sum(t == pos and p != pos for t, p in zip(y_true, y_pred))
    tn = sum(t != pos and p != pos for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return acc, prec, rec, f1, (tn, fp, fn, tp)


d = pd.read_csv(PRED, dtype=str).fillna("")
d = d[d.pred.isin(["0", "1"])]
b = pd.read_csv(BACKUP, dtype=str).fillna("")
old_texts = set(b.text)
d["stratum"] = ["keyword" if t in old_texts else "harvest" for t in d.text]

print(f"n = {len(d)}  (keyword {sum(d.stratum=='keyword')} / harvest {sum(d.stratum=='harvest')})\n")

# ---------- 1) compare to dumb baselines ----------
yt = d.label.tolist()
print("== [1] does GPT-4o actually beat random guessing ==")
for name, yp in [
    ("guess 'not sarcastic' for all", ["0"] * len(d)),
    ("guess 'sarcastic' for all", ["1"] * len(d)),
    ("GPT-4o zero-shot", d.pred.tolist()),
]:
    acc, prec, rec, f1, _ = prf(yt, yp)
    print(f"  {name:<26} acc {acc:.3f}  prec {prec:.3f}  rec {rec:.3f}  F1 {f1:.3f}")

# ---------- 2) split by origin ----------
print("\n== [2] self-selection bias: is GPT-4o better on items it selected ==")
for s in ["keyword", "harvest"]:
    g = d[d.stratum == s]
    acc, prec, rec, f1, (tn, fp, fn, tp) = prf(g.label.tolist(), g.pred.tolist())
    npos, nneg = (g.label == "1").sum(), (g.label == "0").sum()
    print(f"\n  [{s}]  sarcastic {npos} / not {nneg}")
    print(f"    recall (caught sarcasm)   : {rec:.3f}   ({tp}/{npos})")
    print(f"    false-positive rate       : {fp/nneg if nneg else 0:.3f}   ({fp}/{nneg} non-sarcastic items predicted sarcastic)")
    print(f"    precision / F1            : {prec:.3f} / {f1:.3f}")

# ---------- 3) control for the source variable (harvest is all wisesight) ----------
print("\n== [3] compare wisesight only (control for length/source) ==")
w = d[d.source == "wisesight"]
for s in ["keyword", "harvest"]:
    g = w[w.stratum == s]
    if not len(g):
        continue
    _, prec, rec, f1, (tn, fp, fn, tp) = prf(g.label.tolist(), g.pred.tolist())
    npos, nneg = (g.label == "1").sum(), (g.label == "0").sum()
    fpr = fp / nneg if nneg else float("nan")
    print(f"  [{s:<7}] sarcastic {npos:>2} / not {nneg:>2} | recall {rec:.3f} ({tp}/{npos}) | FPR {fpr:.3f} ({fp}/{nneg})")

# ---------- 4) bootstrap CI of F1 (avoid over-reading at small n) ----------
print("\n== [4] how noisy is GPT-4o's F1 (bootstrap 2000 rounds) ==")
random.seed(42)
rows = list(zip(yt, d.pred.tolist()))
f1s = []
for _ in range(2000):
    samp = [rows[random.randrange(len(rows))] for _ in rows]
    f1s.append(prf([a for a, _ in samp], [p for _, p in samp])[3])
f1s.sort()
lo, hi = f1s[int(0.025 * len(f1s))], f1s[int(0.975 * len(f1s))]
print(f"  F1 = {prf(yt, d.pred.tolist())[3]:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
print("  -> if multi-agent's F1 stays inside this range, you can't claim it's 'better'")

# ---------- 5) where do the FPs pile up ----------
print("\n== [5] where do the 'sarcastic' mispredictions (false positives) pile up ==")
fp_rows = d[(d.label == "0") & (d.pred == "1")]
print(f"  total {len(fp_rows)} items")
print(fp_rows.groupby(["source", "stratum"]).size().to_string())
