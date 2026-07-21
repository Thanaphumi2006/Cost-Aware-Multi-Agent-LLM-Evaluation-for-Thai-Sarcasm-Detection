# -*- coding: utf-8 -*-
"""Compare baseline vs multi-agent correctly (same gold set -> must use a paired test)

Why not "F1 exceeds the baseline's upper CI":
  that compares as if the two systems measured different sets -> too strict.
  The correct way is a paired bootstrap: resample the same items for both, look at the F1 difference per round.
  If the 95% CI of the difference "doesn't cross 0" = a real difference, not chance.
  + McNemar: look only at items where the two systems disagree.

Run: python compare_systems.py
"""
import glob
import os
import random
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))


def f1_of(label, pred, pos="1"):
    tp = sum(t == pos and p == pos for t, p in zip(label, pred))
    fp = sum(t != pos and p == pos for t, p in zip(label, pred))
    fn = sum(t == pos and p != pos for t, p in zip(label, pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0, prec, rec


def load(path):
    d = pd.read_csv(path, dtype=str).fillna("")
    d = d[d.pred.isin(["0", "1"])][["text", "label", "pred"]]
    return d.set_index("text")


EVAL_DIR = os.environ.get("EVAL_DIR", HERE)  # can point to another results folder (e.g. v2_results)
base = load(os.path.join(EVAL_DIR, "baseline_preds_gpt.csv"))
variants = {}
for p in sorted(glob.glob(os.path.join(EVAL_DIR, "multiagent_preds_gpt*.csv"))):
    name = os.path.basename(p).replace("multiagent_preds_gpt", "").replace(".csv", "").strip("_") or "v?"
    variants[name] = load(p)

# system ③ WangchanBERTa (out-of-fold preds for all 127 items -> pairs the same way)
wcb = os.path.join(EVAL_DIR, "wangchanberta_preds.csv")
if os.path.exists(wcb):
    variants["wangchanberta"] = load(wcb)

print(f"baseline: {len(base)} items | multi-agent variants: {list(variants)}\n")
bf1, bp, br = f1_of(base.label, base.pred)
print(f"{'system':<16}{'F1':>8}{'prec':>8}{'recall':>8}")
print(f"{'baseline':<16}{bf1:>8.3f}{bp:>8.3f}{br:>8.3f}")
for name, d in variants.items():
    f1, pr, rc = f1_of(d.label, d.pred)
    print(f"{name:<16}{f1:>8.3f}{pr:>8.3f}{rc:>8.3f}")

random.seed(42)
N = 5000
for name, d in variants.items():
    # align to the same items (paired)
    common = base.index.intersection(d.index)
    b = base.loc[common]
    m = d.loc[common]
    lab = b.label.tolist()
    bp_, mp_ = b.pred.tolist(), m.pred.tolist()

    f1b, _, _ = f1_of(lab, bp_)
    f1m, _, _ = f1_of(lab, mp_)

    # paired bootstrap of the F1 difference
    idx = list(range(len(lab)))
    diffs = []
    for _ in range(N):
        s = [random.randrange(len(idx)) for _ in idx]
        L = [lab[i] for i in s]
        diffs.append(f1_of(L, [mp_[i] for i in s])[0] - f1_of(L, [bp_[i] for i in s])[0])
    diffs.sort()
    lo, hi = diffs[int(0.025 * N)], diffs[int(0.975 * N)]
    p_worse = sum(1 for x in diffs if x <= 0) / N

    # McNemar: only items where correct/wrong differs
    b_correct = [p == t for p, t in zip(bp_, lab)]
    m_correct = [p == t for p, t in zip(mp_, lab)]
    b_only = sum(bc and not mc for bc, mc in zip(b_correct, m_correct))  # base right, multi wrong
    m_only = sum(mc and not bc for bc, mc in zip(b_correct, m_correct))  # multi right, base wrong

    print(f"\n== {name}  vs baseline (paired, n={len(common)}) ==")
    print(f"  F1: baseline {f1b:.3f} -> {name} {f1m:.3f}  (diff {f1m-f1b:+.3f})")
    print(f"  95% CI of the F1 difference: [{lo:+.3f}, {hi:+.3f}]")
    if lo > 0:
        print(f"  -> CI does not cross 0 = {name} is significantly better than baseline ok")
    else:
        print(f"  -> CI crosses 0 (chance {name} is not better/worse = {p_worse:.0%}) = can't conclude a win")
    print(f"  McNemar: multi-right-base-wrong {m_only} | base-right-multi-wrong {b_only}")
