# -*- coding: utf-8 -*-
"""System ⑦ — Cascade: WangchanBERTa (free) screens -> GPT verifier rejects

Idea: take the winning architecture (a high-recall screener -> a "reject-only" verifier)
and change only the "screener" from GPT (pay on every item) to WangchanBERTa (free, offline, 26ms).

  v2 (original): GPT screener on every item (127 calls) + GPT verifier only on the sarcastic ones (56) = 183 calls
  cascade:       WCB screener on every item (0 calls, free) + GPT verifier only on WCB's sarcastic ones = N calls
  -> the price floor once paid on every item "disappears entirely"

Why WCB's F1 0.620 doesn't mean it can't be a screener:
  a screener needn't be accurate -- it just needs to "not let sarcasm slip" (high recall); the verifier trims the over-flagging
  argmax @0.5 gives recall only 0.700 -> too low to be the first stage
  but "lowering the threshold" raises recall, trading precision (which the verifier can repair)

How to choose the threshold without cheating (very important):
  don't choose the threshold from all of gold and measure on the same gold -- that's leak
  here we use leave-fold-out: items in fold k use a threshold chosen from "the other 4 folds only"
  -> no item helps set the threshold that judges it

An inescapable ceiling: cascade's recall <= the WCB screener's recall (the verifier can only reject,
can't add new sarcasm) -- any sarcasm WCB lets slip at the first stage is lost permanently

Run:
  python cascade.py --dry-run          see how many each threshold would flag / expected cost (free, no API)
  python cascade.py --target-recall 0.95   run for real (needs OPENAI_API_KEY)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

import multiagent
from baseline import PRICE_PER_MTOK, metrics

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OOF_CSV = os.path.join(HERE, "wcb_oof_probs.csv")
# named so compare_systems.py (which globs multiagent_preds_gpt*.csv) picks it up automatically
OUT_CSV = os.path.join(HERE, "multiagent_preds_gpt_cascade.csv")
IN_P, OUT_P = PRICE_PER_MTOK["gpt"]
SEEDS_DEFAULT = [42, 7, 2024]


def cost(i, o):
    return i / 1e6 * IN_P + o / 1e6 * OUT_P


def pick_threshold(probs, y, target_recall):
    """the "highest" threshold that still has recall >= target on the given data
    highest = flags the fewest = cheapest, subject to the required recall
    (recall is non-increasing in the threshold -> only need to scan the prob values of the "actually sarcastic" items)"""
    pos = np.sort(probs[y == 1])[::-1]          # probs of true positives, high->low
    if len(pos) == 0:
        return 0.0
    k = int(np.ceil(target_recall * len(pos)))  # must catch at least k true positives
    k = min(max(k, 1), len(pos))
    return float(pos[k - 1])                    # set the threshold at the kth item -> catches exactly k


def screen(df, seed, target_recall):
    """Return (flag[], tau per fold) -- flag=1 means send onward to the verifier
    fold k's threshold is chosen from "other folds" only (leave-fold-out), not from fold k itself"""
    probs = df[f"prob_seed{seed}"].values
    folds = df[f"fold_seed{seed}"].values
    y = df["label"].astype(int).values

    flag = np.zeros(len(df), dtype=int)
    taus = {}
    for k in sorted(set(folds)):
        held = folds == k
        rest = ~held
        tau = pick_threshold(probs[rest], y[rest], target_recall)   # chosen from the other 4 folds
        flag[held] = (probs[held] >= tau).astype(int)               # applied to the held-out fold
        taus[int(k)] = tau
    return flag, taus


def screener_stats(df, seed, target_recall):
    flag, taus = screen(df, seed, target_recall)
    y = df["label"].astype(int).values
    tp = int(((flag == 1) & (y == 1)).sum())
    fp = int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return dict(seed=seed, flagged=int(flag.sum()), tp=tp, fp=fp, fn=fn,
                recall=rec, prec=prec, taus=taus, flag=flag)


def dry_run(df, seeds, targets):
    """No API calls -- answers one question: "with WCB as the screener, how many items are left for GPT to check?" """
    # real average from v2: one verifier call ~391 in / 7 out tokens
    per_call = cost(391, 7)
    print(f"estimated verifier cost: ${per_call:.5f}/call (391 in / 7 out tokens -- real from v2)")
    print(f"reference: v2 = 183 calls, $0.169 | baseline = 127 calls, $0.094\n")
    print(f"{'target':>7} {'seed':>5} {'flag':>5} {'recall':>7} {'prec':>6} {'FN':>3} "
          f"{'calls':>6} {'~cost':>8} {'vs v2':>7}")
    print("-" * 62)
    for t in targets:
        rows = [screener_stats(df, s, t) for s in seeds]
        for r in rows:
            c = r["flagged"] * per_call
            print(f"{t:>7.2f} {r['seed']:>5} {r['flagged']:>5} {r['recall']:>7.3f} {r['prec']:>6.3f} "
                  f"{r['fn']:>3} {r['flagged']:>6} {'$%.3f' % c:>8} {'%.2fx' % (c/0.169):>7}")
        mr = np.mean([r["recall"] for r in rows])
        mf = np.mean([r["flagged"] for r in rows])
        print(f"{'':>7} {'mean':>5} {mf:>5.1f} {mr:>7.3f} {'':>6} {'':>3} {mf:>6.0f} "
              f"{'$%.3f' % (mf*per_call):>8} {'%.2fx' % (mf*per_call/0.169):>7}\n")
    print("How to read: this recall is cascade's 'ceiling' -- the verifier can only reject")
    print("           can't raise it further; low precision is fine, the verifier's job is to repair it")
    print("           FN = sarcasm the first stage let slip = lost permanently")


def verify(client, texts, cache):
    """GPT verifier -- exactly v2's prompt (multiagent.VERIFY_SYS); the only variable changed is 'who screens'
    cache avoids paying twice for the same text (3 seeds flag many of the same items -- fire once)"""
    n_in = n_out = calls = 0
    for t in texts:
        if t in cache:
            continue
        v, i, o = multiagent._ask(client, multiagent.VERIFY_SYS, multiagent.VERIFY_SCHEMA, "verdict", t)
        cache[t] = v if v in ("0", "1") else "1"     # malformed -> keep (matches the "unsure = keep as sarcasm" rule)
        n_in += i; n_out += o; calls += 1
        print(f"    verifier {calls}/{len(texts)}  -> {cache[t]}", end="\r", flush=True)
    return n_in, n_out, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no API -- just see how many it would flag")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    a = ap.parse_args()

    if not os.path.exists(OOF_CSV):
        sys.exit(f"{OOF_CSV} not found -- run wcb_oof_probs.py first (free)")
    df = pd.read_csv(OOF_CSV, dtype={"text": str, "label": str}).fillna("")
    df["label"] = df["label"].str.strip()

    print(f"cascade: WangchanBERTa (free) -> GPT verifier | gold {len(df)} items\n")
    if a.dry_run:
        dry_run(df, a.seeds, [0.85, 0.90, 0.95, 1.00])
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY required (export OPENAI_API_KEY=sk-...)")
    client = multiagent._make_client()

    y = df["label"].tolist()
    cache = {}
    t0 = time.time()
    per_seed, preds_by_seed = [], {}

    for seed in a.seeds:
        st = screener_stats(df, seed, a.target_recall)
        flag = st["flag"]
        idx = np.where(flag == 1)[0]
        print(f"seed {seed}: WCB flagged {len(idx)} items (recall {st['recall']:.3f}) -> sent to verifier")

        n_in, n_out, ncalls = verify(client, [df["text"].iloc[i] for i in idx], cache)

        # verifier can only reject: pred=1 iff (WCB flagged) and (verifier confirms)
        pred = ["0"] * len(df)
        for i in idx:
            pred[i] = cache[df["text"].iloc[i]]
        preds_by_seed[seed] = pred

        _, prec, rec, f1, (tn, fp, fn, tp) = metrics(y, pred)
        killed = sum(1 for i in idx if cache[df["text"].iloc[i]] == "0")
        per_seed.append(dict(seed=seed, f1=f1, prec=prec, rec=rec, tp=tp, fp=fp, fn=fn,
                             calls=int(flag.sum()), killed=killed,
                             cost=flag.sum() * cost(391, 7)))
        print(f"  -> F1 {f1:.3f} | prec {prec:.3f} | recall {rec:.3f} | TP {tp} FP {fp} FN {fn} "
              f"| verifier rejected {killed} items | calls {int(flag.sum())}\n")

    f1s = [r["f1"] for r in per_seed]
    med_i = int(np.argsort(f1s)[len(f1s) // 2])          # representative seed = median, not the best (cherry-pick)
    rep = per_seed[med_i]["seed"]

    out = df[["text", "label"]].copy()
    out["pred"] = preds_by_seed[rep]
    for s in a.seeds:
        out[f"pred_seed{s}"] = preds_by_seed[s]
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    mean_calls = float(np.mean([r["calls"] for r in per_seed]))
    mean_cost = float(np.mean([r["cost"] for r in per_seed]))
    print("=" * 66)
    print(f"target recall (set)      : {a.target_recall}")
    print(f"F1 per seed              : {', '.join(f'{f:.3f}' for f in f1s)}")
    print(f"F1 mean                  : {np.mean(f1s):.3f}  (SD {np.std(f1s):.3f})")
    print(f"representative seed      : {rep} (median)")
    print(f"mean LLM calls           : {mean_calls:.0f}   (v2 = 183, baseline = 127)")
    print(f"mean cost                : ${mean_cost:.3f}  (v2 = $0.169, baseline = $0.094)")
    print(f"actual API calls         : {len(cache)} (cache avoids paying twice across seeds)")
    print(f"total time               : {(time.time()-t0)/60:.1f} min")
    print(f"saved -> {OUT_CSV}")
    print("=" * 66)
    print("next: python compare_systems.py  (bootstrap + McNemar vs the other systems)")


if __name__ == "__main__":
    main()
