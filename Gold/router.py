# -*- coding: utf-8 -*-
"""System ⑧ — 3-way micro-router: sarcastic / not sarcastic / **unsure** (escalate to the LLM)

cascade.py uses *one* threshold -> every item WCB calls "sarcastic" goes to the verifier
router.py uses *two* thresholds -> keep only the items WCB is "unsure" about for the LLM
  prob < lo          -> decide "not sarcastic" itself  (free, 0 calls)
  prob >= hi         -> decide "sarcastic" itself       (free, 0 calls)
  lo <= prob < hi    -> **unsure** -> send to GPT to decide (pay only for these)

Why it should work: finding 14 showed the free methods (regex 555 = 0.590, kNN = 0.588,
WCB = 0.620) beat every 7-8B open model -> the "free layer" at ~0.6 is real.
The remaining question is **how much you must pay to climb from 0.62 to 0.727 (the GPT bot)**.
This script answers by sweeping the escalation budget b = 0%..100% and plotting the frontier.

How to choose thresholds without cheating: leave-fold-out, exactly like cascade.py
  items in fold k use (tau, delta, tau_gpt) chosen from "the other 4 folds only"
  -> no item helps set the threshold that judges it
b = 0   -> WCB only (all free)
b = 1   -> GPT only (pay on every item) = the same endpoint as the GPT bot
the middle is what this script measures

Joining files: **join by position, not by text** (see HANDOFF.md -- score_lm_eval.py
rewrites newlines to spaces, so a text-join silently drops multi-line items)
every file derives from gold.csv in order -> assert the labels match first, then index by position

Run:  python router.py              (free, no API -- uses precomputed probs)
      python router.py --seeds 42 7 2024
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from baseline import PRICE_PER_MTOK, metrics

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OOF_CSV = os.path.join(HERE, "wcb_oof_probs.csv")
GPT_CSV = os.path.join(HERE, "frontier_probs_gpt-4.1-mini.csv")
OUT_CSV = os.path.join(HERE, "router_frontier.csv")
PRED_CSV = os.path.join(HERE, "multiagent_preds_gpt_router.csv")

IN_P, OUT_P = PRICE_PER_MTOK["gpt"]
# real average of one v2 call (391 in / 7 out) -- same value as cascade.py so they compare
COST_PER_CALL = 391 / 1e6 * IN_P + 7 / 1e6 * OUT_P
BUDGETS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]


def f1_at(probs, y, tau):
    pred = (probs >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def best_tau(probs, y):
    """the threshold with the highest F1 on the given data (used only on 'the other 4 folds')"""
    cands = np.unique(np.concatenate([probs, [0.0, 1.0]]))
    return float(max(cands, key=lambda t: f1_at(probs, y, t)))


def route(df, seed, budget, gpt_prob, tau_mode="fixed"):
    """Return (pred[], band[], n_escalated) -- band: 0=not sarcastic(free) 1=sarcastic(free) 2=unsure(paid)

    delta = the uncertainty radius around tau. Items with |prob - tau| < delta count as 'unsure'
    choose delta from the other 4 folds to get escalate fraction = budget (standard uncertainty sampling)

    tau_mode: "fixed" = use 0.5 directly | "tuned" = tune tau leave-fold-out
      default is fixed because **it was measured that tuning makes it worse** at n=127 (see --tau-mode in the output)
      tuning on the other 4 folds gives F1 0.556 / no tuning gives 0.590 -> here tuning is pure overfitting
    """
    probs = df[f"prob_seed{seed}"].values.astype(float)
    folds = df[f"fold_seed{seed}"].values
    y = df["label"].astype(int).values

    pred = np.zeros(len(df), dtype=int)
    band = np.zeros(len(df), dtype=int)
    for k in sorted(set(folds)):
        held = folds == k
        rest = ~held
        tau = 0.5 if tau_mode == "fixed" else best_tau(probs[rest], y[rest])
        # delta: quantile of the distance from tau on the other 4 folds -> escalate fraction ~= budget
        d_rest = np.abs(probs[rest] - tau)
        delta = float(np.quantile(d_rest, budget)) if budget > 0 else -1.0
        tau_gpt = best_tau(gpt_prob[rest], y[rest])   # GPT's threshold is also chosen leave-fold-out

        d_held = np.abs(probs[held] - tau)
        unsure = d_held < delta
        auto = (probs[held] >= tau).astype(int)
        esc = (gpt_prob[held] >= tau_gpt).astype(int)

        pred[held] = np.where(unsure, esc, auto)
        band[held] = np.where(unsure, 2, auto)
    return pred, band, int((band == 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    ap.add_argument("--tau-mode", choices=["fixed", "tuned"], default="fixed",
                    help="fixed=0.5 (default, better) | tuned=leave-fold-out (overfits at this n)")
    a = ap.parse_args()

    for p in (OOF_CSV, GPT_CSV):
        if not os.path.exists(p):
            sys.exit(f"{p} not found")
    df = pd.read_csv(OOF_CSV, encoding="utf-8-sig")
    gpt = pd.read_csv(GPT_CSV, encoding="utf-8-sig")

    # join by position -- first verify the two files are aligned (never join by text)
    if len(df) != len(gpt):
        sys.exit(f"row counts differ: oof {len(df)} vs gpt {len(gpt)}")
    a_lab = df["label"].astype(str).str.strip().tolist()
    b_lab = gpt["label"].astype(str).str.strip().tolist()
    if a_lab != b_lab:
        sys.exit(f"label order mismatch -> can't join by position ({sum(x!=y for x,y in zip(a_lab,b_lab))} differ)")

    df["label"] = df["label"].astype(str).str.strip()
    y = df["label"].tolist()
    gpt_prob = gpt["prob"].values.astype(float)
    n = len(df)

    print(f"3-way micro-router | gold {n} items | seeds {a.seeds} | tau-mode {a.tau_mode}")
    print(f"verifier cost ${COST_PER_CALL:.5f}/call (391 in / 7 out -- real from v2)\n")
    print(f"{'budget':>7} {'escalate':>9} {'%':>6} {'F1':>6} {'prec':>6} {'rec':>6} "
          f"{'calls':>6} {'cost':>8} {'vs GPT':>7}")
    print("-" * 72)

    rows = []
    preds_at = {}
    for b in BUDGETS:
        f1s, precs, recs, escs = [], [], [], []
        for s in a.seeds:
            pred, band, n_esc = route(df, s, b, gpt_prob, a.tau_mode)
            p = [str(v) for v in pred]
            _, prec, rec, f1, _ = metrics(y, p)
            f1s.append(f1); precs.append(prec); recs.append(rec); escs.append(n_esc)
            if s == a.seeds[0]:
                preds_at[b] = (pred, band)
        mf1, mprec, mrec = np.mean(f1s), np.mean(precs), np.mean(recs)
        mesc = float(np.mean(escs))
        c = mesc * COST_PER_CALL
        rows.append(dict(budget=b, escalated=mesc, pct=100 * mesc / n, f1=mf1,
                         prec=mprec, rec=mrec, calls=mesc, cost=c, f1_sd=np.std(f1s)))
        print(f"{b:>7.2f} {mesc:>9.1f} {100*mesc/n:>5.1f}% {mf1:>6.3f} {mprec:>6.3f} "
              f"{mrec:>6.3f} {mesc:>6.0f} {'$%.4f' % c:>8} {'%.0f%%' % (100*mf1/0.727):>7}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # reported point = the *lowest* budget that recovers >= half the F1 gain GPT-only provides
    # (don't use "F1 per call max" -- that criterion always favors the smallest budget, tells you nothing)
    lo_f1, hi_f1 = rows[0]["f1"], rows[-1]["f1"]
    need = lo_f1 + 0.5 * (hi_f1 - lo_f1)
    reached = [r for r in rows if r["escalated"] > 0 and r["f1"] >= need]
    knee = min(reached, key=lambda r: r["escalated"]) if reached else rows[-1]
    pred, band = preds_at[knee["budget"]]
    pr = df[["text", "label"]].copy()
    pr["pred"] = [str(v) for v in pred]
    pr["band"] = ["not_sarcastic(free)" if b == 0 else "sarcastic(free)" if b == 1 else "unsure->LLM"
                  for b in band]
    pr.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")

    base_f1 = rows[0]["f1"]
    full_f1 = rows[-1]["f1"]
    print("\n" + "=" * 72)
    print(f"b=0 (WCB only, free)   F1 {base_f1:.3f} | 0 calls    | $0")
    print(f"b=1 (GPT only)         F1 {full_f1:.3f} | {n} calls  | ${n*COST_PER_CALL:.4f}")
    print(f"sweet spot b={knee['budget']:.2f}       F1 {knee['f1']:.3f} | {knee['escalated']:.0f} calls "
          f"| ${knee['cost']:.4f}  ({100*knee['escalated']/n:.0f}% of all items)")
    if full_f1 > base_f1:
        recovered = 100 * (knee["f1"] - base_f1) / (full_f1 - base_f1)
        print(f"-> pay {100*knee['escalated']/n:.0f}% of the full price, get {recovered:.0f}% of the F1 gain GPT-only provides")
    print(f"saved -> {os.path.basename(OUT_CSV)} , {os.path.basename(PRED_CSV)}")
    print("=" * 72)
    print("next: python compare_systems.py  (bootstrap + McNemar vs the other systems)")


if __name__ == "__main__":
    main()
