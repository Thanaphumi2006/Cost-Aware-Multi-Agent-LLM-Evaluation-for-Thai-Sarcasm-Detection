# -*- coding: utf-8 -*-
"""3-tier cascade experiment: does WangchanBERTa earn a slot between the cue screener and the LLM?

Compares, on the 127-item gold set, entirely offline (cached probs, no API calls, no training):
    A. cue -> LLM              (what the live demo does today)
    B. WCB -> LLM              (Finding 6's shape, as a confidence band)
    C. cue -> WCB -> LLM       (the proposal)
    D. LLM only                (ceiling reference)
    E. cue -> WCB              (C with the LLM tier switched off)
    N. cue -> [always say 0] -> LLM   (control: WCB replaced by a free constant, same ordering)

WCB probs are the leak-free out-of-fold ones from Finding 6.
LLM probs are cached gpt-4.1-mini P(sarcastic); threshold 0.095 = the deployed 'balanced' operating point.
"""
import math
import os
import re
import sys

import numpy as np
import pandas as pd

GOLD = os.path.dirname(os.path.abspath(__file__)) + "/"   # this Gold/ folder, wherever the repo lives
T_LLM = 0.095                       # OPERATING["balanced"]["t"] in predict.py
SEEDS = ["seed42", "seed7", "seed2024"]
RNG = np.random.default_rng(0)

# ---- the cue model, ported verbatim from app.html / space/app.py -------------
CUES = [("555", re.compile(r"555"), 2.46),
        ("??", re.compile(r"[?]{2,}"), 2.54),
        ("ตัวอักษรยืด", re.compile(r"(.)\1{2,}"), 1.69),
        ("จ้า", re.compile(r"จ้า"), 1.32),
        ("ค่ะ", re.compile(r"ค่ะ"), 0.40),
        ("นะคะ", re.compile(r"นะคะ"), 0.22),
        ("ครับ", re.compile(r"ครับ"), 0.05)]


def cue_pred(text):
    """-> 1 / 0 / None(abstain), matching verdictOf() in app.html"""
    hits = [lift for _, rx, lift in CUES if rx.search(text)]
    if not hits:
        return None
    return 1 if sum(math.log(max(l, 0.05)) for l in hits) > 0 else 0


# ---- metrics ----------------------------------------------------------------
def prf(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1


def best_threshold(prob, y):
    """threshold maximising F1 -- deliberately generous to WCB (chosen in-sample)"""
    cand = sorted(set(np.round(prob, 6)))
    best = (0.0, 0.5)
    for t in cand:
        f1 = prf(y, (prob >= t).astype(int))[2]
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


# ---- data -------------------------------------------------------------------
def load():
    def rd(name):
        d = pd.read_csv(GOLD + name)
        d.columns = [c.lstrip("﻿") for c in d.columns]
        return d

    g, w, l = rd("gold.csv"), rd("wcb_oof_probs.csv"), rd("frontier_probs_gpt-4.1-mini.csv")
    d = g[["text", "label", "source"]].merge(
        w[["text"] + [f"prob_{s}" for s in SEEDS]], on="text").merge(
        l[["text", "prob"]].rename(columns={"prob": "llm"}), on="text")
    assert len(d) == len(g), f"join lost rows: {len(d)} vs {len(g)}"
    d["wcb"] = d[[f"prob_{s}" for s in SEEDS]].mean(axis=1)
    d["cue"] = [cue_pred(t) for t in d.text]
    d["llm_pred"] = (d.llm >= T_LLM).astype(int)
    return d


# ---- cascade ----------------------------------------------------------------
def run_cascade(d, use_cue, use_wcb, n_defer, t_wcb, mid_pred="wcb"):
    """returns (preds, n_llm_calls).

    tier 1: cue answers items with a cue hit          (free)
    tier 2: among the leftovers, the n_defer LEAST WCB-confident go on; WCB answers the rest (free)
    tier 3: LLM answers whatever is left              (paid, 1 call each)
    """
    pred = np.full(len(d), -1)
    idx = np.arange(len(d))

    if use_cue:
        for i in idx:
            c = d.cue.iloc[i]
            if pd.notna(c):
                pred[i] = int(c)
    left = idx[pred == -1]

    if use_wcb and len(left):
        conf = np.abs(d.wcb.values[left] - t_wcb)
        order = left[np.argsort(conf)]           # least confident first
        deferred = set(order[:n_defer].tolist())
        for i in left:
            if i not in deferred:
                pred[i] = int(d.wcb.iloc[i] >= t_wcb) if mid_pred == "wcb" else 0
        left = idx[pred == -1]

    for i in left:
        pred[i] = d.llm_pred.iloc[i]
    return pred, len(left)


def row(name, d, pred, calls):
    pr, rc, f1 = prf(d.label.values, pred)
    return dict(config=name, P=pr, R=rc, F1=f1, calls=calls,
                cost_pct=100 * calls / len(d))


def boot_delta(d, pa, pb, n=4000):
    """paired bootstrap of F1(b) - F1(a); returns (mean, lo, hi, P(b>a))"""
    y = d.label.values
    n_ = len(y)
    out = []
    for _ in range(n):
        s = RNG.integers(0, n_, n_)
        out.append(prf(y[s], pb[s])[2] - prf(y[s], pa[s])[2])
    out = np.array(out)
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5), float((out > 0).mean())


def main():
    d = load()
    y = d.label.values
    n_cue_hit = int(d.cue.notna().sum())
    n_open = len(d) - n_cue_hit
    t_wcb = best_threshold(d.wcb.values, y)

    print(f"n={len(d)}  positives={int(y.sum())}  cue answers {n_cue_hit}, leaves {n_open} open")
    print(f"t_llm={T_LLM} (deployed)   t_wcb={t_wcb:.4f} (F1-optimal in-sample, generous to WCB)")
    print(f"WCB alone @t: F1={prf(y, (d.wcb.values >= t_wcb).astype(int))[2]:.3f}   "
          f"LLM alone: F1={prf(y, d.llm_pred.values)[2]:.3f}\n")

    rows, preds = [], {}

    p, c = run_cascade(d, False, False, 0, t_wcb); preds["D. LLM only"] = p
    rows.append(row("D. LLM only", d, p, c))

    p, c = run_cascade(d, True, False, 0, t_wcb); preds["A. cue -> LLM (today)"] = p
    rows.append(row("A. cue -> LLM (today)", d, p, c))

    p, c = run_cascade(d, False, True, 0, t_wcb); preds["WCB only"] = p
    rows.append(row("WCB only", d, p, c))

    p, c = run_cascade(d, True, True, 0, t_wcb); preds["E. cue -> WCB (no LLM)"] = p
    rows.append(row("E. cue -> WCB (no LLM)", d, p, c))

    # --- C: full defer sweep -------------------------------------------------
    sweep = []
    for k in range(0, n_open + 1):
        p, c = run_cascade(d, True, True, k, t_wcb)
        pr, rc, f1 = prf(y, p)
        sweep.append((k, c, pr, rc, f1))
    best_k, best_calls, bp, br, bf1 = max(sweep, key=lambda r: (r[4], -r[1]))
    p_best, c_best = run_cascade(d, True, True, best_k, t_wcb)
    preds["C. cue -> WCB -> LLM (best band)"] = p_best
    rows.append(row(f"C. cue -> WCB -> LLM (defer {best_k}/{n_open})", d, p_best, c_best))

    # --- B: WCB -> LLM, sweep ------------------------------------------------
    bsweep = []
    for k in range(0, len(d) + 1):
        p, c = run_cascade(d, False, True, k, t_wcb)
        bsweep.append((k, c, prf(y, p)[2]))
    bk, bc, bf = max(bsweep, key=lambda r: (r[2], -r[1]))
    p, c = run_cascade(d, False, True, bk, t_wcb)
    preds["B. WCB -> LLM (best band)"] = p
    rows.append(row(f"B. WCB -> LLM (defer {bk}/{len(d)})", d, p, c))

    # --- N: control, WCB's decisions replaced by constant 0 -------------------
    p, c = run_cascade(d, True, True, best_k, t_wcb, mid_pred="zero")
    preds["N. control (free 'not sarcastic')"] = p
    rows.append(row("N. control: cue -> always-0 -> LLM", d, p, c))

    t = pd.DataFrame(rows)[["config", "P", "R", "F1", "calls", "cost_pct"]]
    print("=== headline (127-item gold) ===")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== C: full sweep of the WCB defer band (cue -> WCB -> LLM) ===")
    print(" defer  LLMcalls   P      R      F1")
    for k, c, pr, rc, f1 in sweep:
        mark = "  <-- best" if k == best_k else ("  == config A" if k == n_open else "")
        if k % max(1, n_open // 20) == 0 or k in (best_k, n_open):
            print(f"  {k:3d}     {c:3d}    {pr:.3f}  {rc:.3f}  {f1:.3f}{mark}")

    # --- is C's gain over A real? -------------------------------------------
    a = preds["A. cue -> LLM (today)"]
    print("\n=== paired bootstrap vs A (cue -> LLM, today) ===")
    for name in ["C. cue -> WCB -> LLM (best band)", "B. WCB -> LLM (best band)",
                 "E. cue -> WCB (no LLM)", "N. control (free 'not sarcastic')", "D. LLM only"]:
        m, lo, hi, pw = boot_delta(d, a, preds[name])
        print(f"  {name:38s} dF1={m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(better)={pw:.2f}")

    # --- domain split --------------------------------------------------------
    print("\n=== by domain (out-of-domain sanity check) ===")
    for src, sub in d.groupby("source"):
        m = d.source.values == src
        print(f"\n  {src}  n={m.sum()}  positives={int(y[m].sum())}")
        for name in ["A. cue -> LLM (today)", "C. cue -> WCB -> LLM (best band)",
                     "E. cue -> WCB (no LLM)", "D. LLM only"]:
            pr, rc, f1 = prf(y[m], preds[name][m])
            print(f"    {name:36s} P={pr:.3f} R={rc:.3f} F1={f1:.3f}")

    # --- per-seed stability of C --------------------------------------------
    print("\n=== C's best band, per WCB seed (is the middle tier stable?) ===")
    for s in SEEDS:
        d2 = d.copy(); d2["wcb"] = d2[f"prob_{s}"]
        t2 = best_threshold(d2.wcb.values, y)
        p2, c2 = run_cascade(d2, True, True, best_k, t2)
        pr, rc, f1 = prf(y, p2)
        print(f"  {s:8s} t={t2:.4f}  F1={f1:.3f}  calls={c2}")


if __name__ == "__main__":
    sys.exit(main())
