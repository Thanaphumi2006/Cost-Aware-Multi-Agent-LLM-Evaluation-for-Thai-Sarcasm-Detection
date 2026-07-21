# -*- coding: utf-8 -*-
"""The fair comparison: every system vs. the "tuned" baseline (not the raw baseline)

Why this file exists (the most important point of the project):
  compare_systems.py compares everything to baseline @argmax (F1 0.690), which "throws away the logprob" already paid for
  -> that makes multi-agent fight a handicapped opponent -> credits multi-agent too much.
  The correct comparison point is baseline + threshold (F1 0.725, same cost $0.094, no added calls).

This file makes no API calls -- it reuses the saved predictions of every system.
Run: python compare_fair.py
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REF_FILE = "multiagent_preds_gpt_threshold.csv"   # the tuned baseline = the fair comparison point
REF_NAME = "baseline+threshold"
N_BOOT = 5000
RNG = np.random.default_rng(0)

# cost per system (from RESULTS.md), a reminder of "what the extra spend buys"
COST = {"baseline": 0.094, "baseline+threshold": 0.094, "conservative": 0.169,
        "v1aggressive": 0.157, "debate": 0.695, "hybrid": 0.407, "cascade": 0.124}


def load(path):
    d = pd.read_csv(path, dtype=str).fillna("")
    d["label"] = d["label"].str.strip()
    d = d[d["pred"].isin(["0", "1"])]
    return d.set_index("text")[["label", "pred"]]


def f1(y, p):
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main():
    ref = load(os.path.join(HERE, REF_FILE))
    systems = {"baseline": "baseline_preds_gpt.csv"}
    for p in sorted(glob.glob(os.path.join(HERE, "multiagent_preds_gpt_*.csv"))):
        nm = os.path.basename(p).replace("multiagent_preds_gpt_", "").replace(".csv", "")
        if nm == "threshold":
            continue                      # this is the ref itself
        systems[nm] = p
    systems["wangchanberta"] = "wangchanberta_preds.csv"

    print(f"comparison point (fair) = {REF_NAME}: F1 {f1((ref['label']=='1').values,(ref['pred']=='1').values):.3f} "
          f"| ${COST[REF_NAME]:.3f} | 127 calls\n")
    print(f"{'system':<16}{'F1':>6}{'diff':>8}{'95% CI':>20}{'P(not better)':>15}{'McNemar':>12}{'cost':>8}")
    print("-" * 86)

    for name, fn_ in systems.items():
        s = load(os.path.join(HERE, fn_))
        common = ref.index.intersection(s.index)
        r, ss = ref.loc[common], s.loc[common]
        y = (r["label"].values == "1")
        pr = (r["pred"].values == "1")      # ref (tuned baseline)
        ps = (ss["pred"].values == "1")     # the system being compared
        n = len(common)

        diffs = np.array([f1(y[i], ps[i]) - f1(y[i], pr[i])
                          for i in (RNG.integers(0, n, n) for _ in range(N_BOOT))])
        d0 = f1(y, ps) - f1(y, pr)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_not = (diffs <= 0).mean() * 100
        win = int(((ps == y) & (pr != y)).sum())   # system right, ref wrong
        los = int(((pr == y) & (ps != y)).sum())   # ref right, system wrong
        c = COST.get(name, float("nan"))
        star = "  <-" if (lo <= 0 <= hi) else ""    # CI crosses 0 = indistinguishable from ref
        print(f"{name:<16}{f1(y,ps):>6.3f}{d0:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]{p_not:>12.0f}%"
              f"{f'{win}-{los}':>12}{f'${c:.3f}':>8}{star}")

    print("\nHow to read:")
    print("  'diff' = F1(system) - F1(baseline+threshold) ; positive = system beats the fair comparison point")
    print("  '<-' = CI crosses 0 = can't separate from the tuned baseline (paying extra for noise)")
    print("  McNemar = (system-right-baseline+threshold-wrong) - (vice versa) ; near-tie = no real advantage")


if __name__ == "__main__":
    main()
