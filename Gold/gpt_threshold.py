# -*- coding: utf-8 -*-
"""System ⑧ — extract a "score" (not just 0/1) from the GPT screener and tune a threshold

Why: every system in this project answers a hard 0/1 at a single operating point
  -> a real user can't choose their own operating point (screening wants recall / auto-reply wants precision)
  -> and we don't even know how "close" the baseline came to answering the other way

Method: call the exact same DETECT_SYS (same prompt, same json_object) but also request logprobs,
  then read P("1") at the label-token position -> a continuous score 0..1.
  **Don't touch the prompt** -> still directly comparable to the original baseline (the only added variable is "reading out the score").

How this differs from cascade (finding 6): finding 6 proved WCB "can't rank" -> can't be a screener.
  This file asks the same question of GPT: **can GPT rank?** If so -> tune the threshold to buy precision for free.
  (Truly free: no extra calls; the score comes from a call already paid for.)

How to choose the threshold without cheating: leave-fold-out, like cascade.py.
  Items in fold k use a threshold chosen from the other 4 folds only -> no item sets the threshold that judges it.

Run:
  python gpt_threshold.py --score       fire GPT 127 times to collect scores (~$0.05) -> gpt_screener_probs.csv
  python gpt_threshold.py --sweep       analyze thresholds from the saved scores (free, no re-firing the API)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import multiagent
from baseline import PRICE_PER_MTOK, metrics

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
# override with the same env as baseline.py: GOLD_CSV + EVAL_DIR
EVAL_DIR = os.environ.get("EVAL_DIR", HERE)
os.makedirs(EVAL_DIR, exist_ok=True)
GOLD_CSV = os.environ.get("GOLD_CSV", os.path.join(HERE, "gold.csv"))
PROB_CSV = os.path.join(EVAL_DIR, "gpt_screener_probs.csv")
# this name lets compare_systems.py (glob multiagent_preds_gpt*.csv) pick it up for bootstrap/McNemar
OUT_CSV = os.path.join(EVAL_DIR, "multiagent_preds_gpt_threshold.csv")
IN_P, OUT_P = PRICE_PER_MTOK["gpt"]
N_FOLDS, SEED = 5, 42


def cost(i, o):
    return i / 1e6 * IN_P + o / 1e6 * OUT_P


def score_one(client, text):
    """Call the same DETECT_SYS + request logprobs -> return (P(sarcastic), in_tok, out_tok)

    The output is JSON {"label": "1"} -> the label-value token is a bare '1' or '0'.
    Read top_logprobs at that position, then normalize over just '0' and '1' (the rest is noise, discardable)."""
    r = client.chat.completions.create(
        model=multiagent.MODELS["gpt"], max_tokens=20,
        response_format={"type": "json_object"},
        logprobs=True, top_logprobs=20,
        messages=[{"role": "system", "content": multiagent.DETECT_SYS},
                  {"role": "user", "content": f"ข้อความ: {text}"}],
    )
    in_tok, out_tok = r.usage.prompt_tokens, r.usage.completion_tokens
    content = r.choices[0].logprobs.content or []

    # find the token that is the label "value" (a bare digit) -- not the key, not punctuation
    for tok in content:
        if tok.token.strip().strip('"') not in ("0", "1"):
            continue
        p0 = p1 = 0.0
        for alt in tok.top_logprobs:
            t = alt.token.strip().strip('"')
            if t == "1":
                p1 += math.exp(alt.logprob)
            elif t == "0":
                p0 += math.exp(alt.logprob)
        if p0 + p1 > 0:
            return p1 / (p0 + p1), in_tok, out_tok

    # fallback: couldn't read it -> use the hard answer as 0/1 (rare, but don't crash the whole run)
    try:
        v = str(json.loads(r.choices[0].message.content or "{}").get("label", "")).strip()
        return (1.0 if v == "1" else 0.0), in_tok, out_tok
    except json.JSONDecodeError:
        return float("nan"), in_tok, out_tok


def do_score():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY required")
    df = pd.read_csv(GOLD_CSV, dtype=str).fillna("")
    df["label"] = df["label"].str.strip()
    df = df[df["label"].isin(["0", "1"])].reset_index(drop=True)

    client = multiagent._make_client()
    probs, ti, to = [], 0, 0
    t0 = time.time()
    for n, text in enumerate(df["text"], 1):
        p, i, o = score_one(client, text)
        probs.append(p)
        ti += i; to += o
        print(f"  {n}/{len(df)}  P(sarcastic)={p:.3f}", end="\r", flush=True)

    out = df[["text", "label"]].copy()
    out["prob"] = probs
    out.to_csv(PROB_CSV, index=False, encoding="utf-8-sig")
    print(f"\nfired {len(df)} calls | {ti} in / {to} out tokens | ${cost(ti, to):.4f} | {time.time()-t0:.0f}s")
    print(f"saved -> {PROB_CSV}")
    print("next: python gpt_threshold.py --sweep   (free)")


def at(probs, y, tau):
    pred = (probs >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp, fp, fn


def do_sweep():
    if not os.path.exists(PROB_CSV):
        sys.exit(f"{PROB_CSV} not found -- run --score first")
    df = pd.read_csv(PROB_CSV, dtype={"text": str, "label": str}).fillna("")
    df["label"] = df["label"].str.strip()
    y = df["label"].astype(int).values
    probs = df["prob"].astype(float).values

    print(f"gold {len(df)} items | sarcastic {int(y.sum())}")
    print(f"score distribution: min {probs.min():.3f} · p25 {np.percentile(probs,25):.3f} · "
          f"median {np.percentile(probs,50):.3f} · p75 {np.percentile(probs,75):.3f} · max {probs.max():.3f}")
    n_extreme = int(((probs < 0.01) | (probs > 0.99)).sum())
    print(f"extreme scores (<0.01 or >0.99): {n_extreme}/{len(df)} items "
          f"-> if many, GPT is 'overconfident' and the threshold has little to work with\n")

    # ---- full PR curve (not the final result -- this picks the threshold on the same data = cheating) ----
    print(f"{'tau':>6} {'prec':>6} {'recall':>7} {'F1':>6} {'TP':>3} {'FP':>3} {'FN':>3}")
    print("-" * 42)
    for tau in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        prec, rec, f1, tp, fp, fn = at(probs, y, tau)
        print(f"{tau:>6.2f} {prec:>6.3f} {rec:>7.3f} {f1:>6.3f} {tp:>3} {fp:>3} {fn:>3}")

    best_tau = max(np.unique(probs), key=lambda t: at(probs, y, t)[2])
    bp, br, bf, *_ = at(probs, y, best_tau)
    print(f"\nbest threshold (chosen on all of gold = the 'cheating' number, don't report):")
    print(f"  tau {best_tau:.3f} -> F1 {bf:.3f} (prec {bp:.3f} recall {br:.3f})")

    # ---- the reportable number: leave-fold-out ----
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pred = np.zeros(len(df), dtype=int)
    taus = []
    for tr, te in skf.split(probs, y):
        t = max(np.unique(probs[tr]), key=lambda x: at(probs[tr], y[tr], x)[2])  # choose from 4 folds
        pred[te] = (probs[te] >= t).astype(int)                                   # apply to the held-out fold
        taus.append(t)
    _, prec, rec, f1, (tn, fp, fn, tp) = metrics(df["label"].tolist(), [str(p) for p in pred])

    out = df[["text", "label"]].copy()
    out["pred"] = [str(p) for p in pred]
    out["prob"] = probs
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'='*62}")
    print(f"GPT screener + threshold tuned leave-fold-out (the reportable number)")
    print(f"  tau per fold : {', '.join(f'{t:.3f}' for t in taus)}")
    print(f"  F1 {f1:.3f} | prec {prec:.3f} | recall {rec:.3f} | TP {tp} FP {fp} FN {fn}")
    print(f"  LLM calls 127 (exactly like baseline) | cost same as baseline $0.094")
    print(f"compare: baseline @argmax F1 0.690 | v2 (2 agents) F1 0.744 $0.169")
    print(f"saved -> {OUT_CSV}")
    print(f"{'='*62}")
    print("next: python compare_systems.py  (bootstrap + McNemar)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="fire GPT to collect scores (costs ~$0.05)")
    ap.add_argument("--sweep", action="store_true", help="analyze thresholds (free)")
    a = ap.parse_args()
    if a.score:
        do_score()
    elif a.sweep:
        do_sweep()
    else:
        ap.print_help()
