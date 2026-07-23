# -*- coding: utf-8 -*-
"""Honest version of the cue-confidence dial: the cut-off is chosen on training folds only.

A cue cut-off c means: if |cueScore| >= c the cue answers for free, otherwise the item goes to the LLM.
Selecting c in-sample gave F1 0.818; this measures what it is worth out-of-sample.
"""
import math
import numpy as np
import pandas as pd

from cascade_eval import CUES, GOLD, T_LLM, prf

RNG = np.random.default_rng(0)


def cue_info(text):
    hits = [l for _, rx, l in CUES if rx.search(text)]
    if not hits:
        return np.nan, 0.0
    s = sum(math.log(max(l, 0.05)) for l in hits)
    return (1 if s > 0 else 0), abs(s)


def rd(name):
    d = pd.read_csv(GOLD + name)
    d.columns = [c.lstrip("﻿") for c in d.columns]
    return d


g, l = rd("gold.csv"), rd("frontier_probs_gpt-4.1-mini.csv")
d = g[["text", "label", "source"]].merge(l[["text", "prob"]], on="text").reset_index(drop=True)
d["llm_pred"] = (d.prob >= T_LLM).astype(int)
ci = [cue_info(t) for t in d.text]
d["cp"], d["cs"] = [c for c, _ in ci], [s for _, s in ci]
y = d.label.values
CUTS = sorted(set(np.round(d.cs.values, 6))) + [1e9]


def apply_cut(sub, c):
    """cue answers when it has a hit and |score| >= c; else the LLM does"""
    use = sub.cp.notna() & (sub.cs >= c)
    pred = np.where(use, sub.cp.fillna(0).values, sub.llm_pred.values).astype(int)
    return pred, int((~use).sum())


def f1_at(sub, c):
    return prf(sub.label.values, apply_cut(sub, c)[0])[2]


print("=== 5-fold cross-validated choice of the cue cut-off (repeated 20x) ===")
f1s, costs, chosen = [], [], []
for rep in range(20):
    idx = RNG.permutation(len(d))
    folds = np.array_split(idx, 5)
    pred = np.empty(len(d), dtype=int)
    calls = 0
    for f in folds:
        tr = d.drop(index=f)
        c = max(CUTS, key=lambda c: (f1_at(tr, c), c))     # pick on training folds only
        chosen.append(c)
        p, k = apply_cut(d.loc[f], c)
        pred[f] = p
        calls += k
    f1s.append(prf(y, pred)[2])
    costs.append(calls / len(d))
print(f"  out-of-sample F1 = {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}   "
      f"LLM cost = {100*np.mean(costs):.0f}% of full")
print(f"  cut-off picked: median |cueScore| >= {np.median(chosen):.2f}")

print("\n  reference points (same 127 items)")
for name, pred, calls in [
        ("today: cue answers every hit", apply_cut(d, 0.0)[0], apply_cut(d, 0.0)[1]),
        ("LLM only", d.llm_pred.values, len(d))]:
    pr, rc, f1 = prf(y, pred)
    print(f"    {name:32s} P={pr:.3f} R={rc:.3f} F1={f1:.3f}  calls={calls} ({100*calls/len(d):.0f}%)")

# a fixed, defensible cut-off rather than a tuned one: |score| >= log(2.46) (one strong cue)
for c, why in [(math.log(2.46), "one strong cue (555)"), (1.0, "|score| >= 1.0"), (1.5, "|score| >= 1.5")]:
    p, k = apply_cut(d, c)
    pr, rc, f1 = prf(y, p)
    print(f"    fixed cut {c:.2f} ({why:22s}) P={pr:.3f} R={rc:.3f} F1={f1:.3f}  calls={k} ({100*k/len(d):.0f}%)")

print("\n=== domain split at the fixed cut |score| >= 1.0 ===")
for src, sub in d.groupby("source"):
    for name, c in [("today (cut 0)", 0.0), ("cut 1.0", 1.0)]:
        p, k = apply_cut(sub, c)
        pr, rc, f1 = prf(sub.label.values, p)
        print(f"  {src:10s} {name:14s} P={pr:.3f} R={rc:.3f} F1={f1:.3f}  calls={k}/{len(sub)}")
