# -*- coding: utf-8 -*-
"""After expanding gold to 157 items (pos 45): have the 2 main systems predict the 30 new items, then re-compare paired

Systems compared (the main pair from finding 8):
  - baseline+threshold (single agent reading the logprob) : continues from frontier_probs_gpt-4o.csv
  - v2 multi-agent (detector->verifier)                   : continues from multiagent_preds_gpt_conservative.csv
Fire only the new items without a pred -> cheap (the original 127 reuse existing results)

*** Bias to declare: the 30 new items were chosen by model score (P>0.2), not random ***
    -> slightly favors the logprob-threshold system · absolute F1 on this set isn't comparable to the original 127
    -> read only "paired on the same 157-item set" and allow for a lean toward threshold

Run: python expand_rerun.py            (fire the new items ~$0.06 then compare)
     python expand_rerun.py --compare  (compare only, if already fired)
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import multiagent
from baseline import PRICE_PER_MTOK, metrics

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold_expanded.csv")                    # 157 -- doesn't touch canonical gold.csv
# separate pred files for the expansion -- seed from the canonical 127 then add the 30 new (no overwrite)
THRESH_CANON = os.path.join(HERE, "frontier_probs_gpt-4o.csv")
V2_CANON = os.path.join(HERE, "multiagent_preds_gpt_conservative.csv")
THRESH_PROBS = os.path.join(HERE, "frontier_probs_gpt-4o_expanded.csv")
V2_PREDS = os.path.join(HERE, "multiagent_preds_gpt_conservative_expanded.csv")
MODEL = "gpt-4o"


def _seed(canon, expanded):
    """build the expanded file from the canonical one -> fire only the new items, not the original 127"""
    if not os.path.exists(expanded) and os.path.exists(canon):
        pd.read_csv(canon, dtype=str).fillna("").to_csv(expanded, index=False, encoding="utf-8-sig")
IN_P, OUT_P = PRICE_PER_MTOK["gpt"]


def gold_df():
    g = pd.read_csv(GOLD, dtype=str).fillna("")
    g["label"] = g["label"].str.strip()
    return g[g["label"].isin(["0", "1"])].reset_index(drop=True)


def score_logprob(client, text):
    r = client.chat.completions.create(
        model=MODEL, max_tokens=20, response_format={"type": "json_object"},
        logprobs=True, top_logprobs=20,
        messages=[{"role": "system", "content": multiagent.DETECT_SYS},
                  {"role": "user", "content": f"ข้อความ: {text}"}])
    for tok in (r.choices[0].logprobs.content or []):
        if tok.token.strip().strip('"') not in ("0", "1"):
            continue
        p0 = p1 = 0.0
        for alt in tok.top_logprobs:
            t = alt.token.strip().strip('"')
            if t == "1": p1 += math.exp(alt.logprob)
            elif t == "0": p0 += math.exp(alt.logprob)
        if p0 + p1 > 0:
            return p1 / (p0 + p1)
    return float("nan")


def do_score():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY required")
    g = gold_df()
    _seed(THRESH_CANON, THRESH_PROBS)      # copy the original 127 as a base, fire only the 30 new
    _seed(V2_CANON, V2_PREDS)
    from openai import OpenAI
    client = OpenAI(timeout=30.0, max_retries=3)

    # --- 1) threshold: add logprob for the new items ---
    tp = pd.read_csv(THRESH_PROBS, dtype=str).fillna("")
    have = set(tp["text"])
    new = [t for t in g["text"] if t not in have]
    print(f"threshold: have {len(have)} · to fire {len(new)}")
    rows = []
    for n, t in enumerate(new, 1):
        rows.append({"text": t, "label": g.loc[g["text"] == t, "label"].iloc[0],
                     "prob": score_logprob(client, t)})
        print(f"  logprob {n}/{len(new)}", end="\r", flush=True)
    if rows:
        tp = pd.concat([tp, pd.DataFrame(rows)], ignore_index=True)
        tp.to_csv(THRESH_PROBS, index=False, encoding="utf-8-sig")
    print(f"\n  -> {THRESH_PROBS} has {len(tp)} items")

    # --- 2) v2 multi-agent: add pred for the new items ---
    vp = pd.read_csv(V2_PREDS, dtype=str).fillna("")
    have = set(vp["text"])
    new = [t for t in g["text"] if t not in have]
    print(f"v2: have {len(have)} · to fire {len(new)}")
    rows = []
    for n, t in enumerate(new, 1):
        r = multiagent.run_pipeline(client, t)
        rows.append({"text": t, "label": g.loc[g["text"] == t, "label"].iloc[0], "pred": r["pred"]})
        print(f"  v2 {n}/{len(new)}", end="\r", flush=True)
    if rows:
        keep = [c for c in vp.columns if c in ("text", "label", "pred")]
        vp = pd.concat([vp[keep], pd.DataFrame(rows)], ignore_index=True)
        vp.to_csv(V2_PREDS, index=False, encoding="utf-8-sig")
    print(f"\n  -> {V2_PREDS} has {len(vp)} items")


def loo_pred(probs, y):
    def fa(p, yy, t):
        pr = (p >= t).astype(int); tp = ((pr == 1) & (yy == 1)).sum()
        fp = ((pr == 1) & (yy == 0)).sum(); fn = ((pr == 0) & (yy == 1)).sum()
        P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
        return 2*P*R/(P+R) if P+R else 0
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    pred = np.zeros(len(probs), int)
    for tr, te in skf.split(probs, y):
        t = max(np.unique(probs[tr]), key=lambda x: fa(probs[tr], y[tr], x))
        pred[te] = (probs[te] >= t).astype(int)
    return pred


def f1(y, p):
    tp = (y & p).sum(); fp = (~y & p).sum(); fn = (y & ~p).sum()
    return 2*tp/(2*tp+fp+fn) if tp else 0.0


def do_compare():
    g = gold_df()
    tp = pd.read_csv(THRESH_PROBS, dtype=str).fillna("")
    tp = tp[tp["text"].isin(set(g["text"]))]
    y = tp["label"].str.strip().astype(int).values
    thr = pd.Series(loo_pred(tp["prob"].astype(float).values, y).astype(bool), index=tp["text"].values)

    vp = pd.read_csv(V2_PREDS, dtype=str).fillna("")
    vp = vp[vp["pred"].isin(["0", "1"]) & vp["text"].isin(set(g["text"]))]
    v2 = pd.Series(vp["pred"].values == "1", index=vp["text"].values)
    lab = pd.Series(g["label"].values == "1", index=g["text"].values)

    common = thr.index.intersection(v2.index)
    yy = lab.loc[common].values; pa = thr.loc[common].values; pb = v2.loc[common].values
    n = len(common); npos = int(yy.sum())
    rng = np.random.default_rng(0)
    d = np.array([f1(yy[i], pb[i]) - f1(yy[i], pa[i]) for i in (rng.integers(0, n, n) for _ in range(5000))])
    lo, hi = np.percentile(d, [2.5, 97.5])
    nb = int(((pb == yy) & (pa != yy)).sum()); na = int(((pa == yy) & (pb != yy)).sum())

    print("=" * 64)
    print(f"re-compared on expanded gold: n={n} (sarcastic {npos})   [original n=127 sarcastic 30]")
    print(f"  baseline+threshold  F1 {f1(yy,pa):.3f}")
    print(f"  v2 multi-agent      F1 {f1(yy,pb):.3f}")
    print(f"  diff (v2-thr) {f1(yy,pb)-f1(yy,pa):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"P(v2 not better)={np.mean(d<=0)*100:.0f}%")
    print(f"  McNemar: v2-right-thr-wrong {nb} | thr-right-v2-wrong {na}")
    cross = "crosses 0 (still indistinguishable)" if lo <= 0 <= hi else "does not cross 0 (separated!)"
    print(f"  -> CI {cross}")
    print("=" * 64)
    print("bias: the 30 new items were chosen by model score -> leans toward threshold · absolute F1 not comparable to the original 127")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="compare only, no firing")
    a = ap.parse_args()
    if not a.compare:
        do_score()
    do_compare()
