# -*- coding: utf-8 -*-
"""Check whether gold.csv is enough to evaluate: count, 1/0 ratio, and warnings"""
import os
import pandas as pd

# read gold.csv from the same folder as this script (run from anywhere)
GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold.csv")
g = pd.read_csv(GOLD)
g["label"] = g["label"].astype(str).str.strip()

n = len(g)
n1 = (g["label"] == "1").sum()   # sarcasm
n0 = (g["label"] == "0").sum()   # not sarcasm

print(f"total: {n} items")
print(f"  sarcasm (1)     : {n1}  ({n1/n*100:.0f}%)")
print(f"  not sarcasm (0) : {n0}  ({n0/n*100:.0f}%)")
if "source" in g.columns:
    print("\ndata sources:")
    print(g["source"].value_counts().to_string())

print("\n== assessment ==")
ok = True
if n1 < 30:
    print(f"warning: only {n1} sarcasm items (should be >=30-40) -> sarcasm-side F1 will be unstable, collect more sarcasm")
    ok = False
if n0 < 30:
    print(f"warning: only {n0} non-sarcasm items (should be >=30) -> collect more")
    ok = False
ratio = min(n1, n0) / max(n1, n0) if max(n1, n0) else 0
if ratio < 0.3:
    print(f"warning: heavily imbalanced (ratio {ratio:.2f}) -> use F1/precision/recall, not accuracy, when evaluating")
if ok:
    print("OK: enough to start a baseline")
else:
    print("-> tip: check more by targeting high suspect_score items (likely sarcasm) to grow the 1 side")