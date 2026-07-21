# -*- coding: utf-8 -*-
"""Measure predict.py on a "new domain" — answering the open question: does it transfer across domains?

Why it matters: all of gold is Wongnai (restaurant reviews) + Wisesight (tweets).
There is no evidence the model works on other domains (news/politics/tech products/YouTube comments, etc.).
This is "the biggest risk" at real deployment — this file is the tool to close that gap.

*** You need data first: a new-domain CSV that a "Thai human has labeled" (not model-labeled) ***
    columns: text, label (1=sarcasm, 0=not)  — at least ~30 sarcastic for a meaningful CI
    label by the same criteria as gold (pretense — see labeling_rubric.md) or it's not comparable

Usage:
  export OPENAI_API_KEY=sk-...
  python eval_domain.py newsdomain.csv                 # measure balanced (gpt-4.1-mini)
  python eval_domain.py newsdomain.csv --op high_recall
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))

# results on gold (the original domain), to compare "how much it drops when crossing domains"
GOLD_REF = {"balanced": dict(P=0.68, R=0.83, F1=0.75),
            "high_recall": dict(P=0.43, R=1.00, F1=0.61)}


def metrics(y, p):
    y, p = np.array(y), np.array(p)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    P = tp/(tp+fp) if tp+fp else 0.0; R = tp/(tp+fn) if tp+fn else 0.0
    F = 2*P*R/(P+R) if P+R else 0.0
    return P, R, F, (tp, fp, fn, tn)


def boot_f1_ci(y, p, n=5000, seed=0):
    y, p = np.array(y), np.array(p); rng = np.random.default_rng(seed); N = len(y); out = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        _, _, f, _ = metrics(y[i], p[i]); out.append(f)
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="new-domain CSV (columns text,label) — human-labeled")
    ap.add_argument("--op", default="balanced", choices=["balanced", "high_recall"])
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    a = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY required")

    df = pd.read_csv(a.csv, dtype=str).fillna("")
    for c in (a.text_col, a.label_col):
        if c not in df.columns:
            sys.exit(f"no column '{c}' (have: {list(df.columns)})")
    df[a.label_col] = df[a.label_col].str.strip()
    df = df[df[a.label_col].isin(["0", "1"])].reset_index(drop=True)
    y = df[a.label_col].astype(int).tolist()
    npos = sum(y)
    if npos < 10:
        print(f"warning: only {npos} sarcastic items — too few, the CI will be very wide (recommend >=30)")

    import predict
    det = predict.SarcasmDetector(operating=a.op)
    print(f"measuring {len(df)} items (sarcastic {npos}) · {det.model} · operating point {a.op}\n", flush=True)
    preds = []
    for n, t in enumerate(df[a.text_col], 1):
        preds.append(det.predict(t).get("label"))
        print(f"  {n}/{len(df)}", end="\r", flush=True)
    ok = [(yy, pp) for yy, pp in zip(y, preds) if pp in (0, 1)]
    y2, p2 = [a_ for a_, _ in ok], [b_ for _, b_ in ok]
    P, R, F, (tp, fp, fn, tn) = metrics(y2, p2)
    lo, hi = boot_f1_ci(y2, p2)
    g = GOLD_REF[a.op]

    print("\n" + "=" * 60)
    print(f"new domain ({os.path.basename(a.csv)}): P {P:.3f} · R {R:.3f} · F1 {F:.3f}  [95% CI {lo:.3f}–{hi:.3f}]")
    print(f"  TP {tp} FP {fp} FN {fn} TN {tn} · cache hit {det.hits}/{det.hits+det.misses}")
    print(f"original gold (Wongnai/Wisesight domain): P {g['P']:.2f} · R {g['R']:.2f} · F1 {g['F1']:.2f}")
    drop = g["F1"] - F
    print("-" * 60)
    if drop > 0.10:
        print(f"warning: F1 dropped {drop:+.3f} crossing domains — large · this model 'should not be trusted' outside the original domain")
    elif drop > 0.05:
        print(f"F1 dropped {drop:+.3f} — some domain gap · usable but be careful / re-tune the threshold per domain")
    else:
        print(f"F1 differs {drop:+.3f} — transfers well (but check the CI: if wide, it's not conclusive)")
    print("=" * 60)


if __name__ == "__main__":
    main()
