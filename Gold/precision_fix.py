# -*- coding: utf-8 -*-
"""Fix precision at the source: add a few-shot "balanced review = not sarcasm" to the detector prompt

Motivation (finding 1): the bottleneck is precision 0.526 -- 27 FPs, mostly "balanced reviews" (real praise + real criticism)
Every prior system tried to clean up these FPs *after* the detector fired -- nobody fixed the detector itself
Hypothesis: telling the detector directly "real praise + real criticism = not sarcasm" should cut FPs at the source, for free (no extra calls)

*** Important: the few-shot examples below are all "synthetic" -- not a single one copied from gold ***
    (using gold as few-shot = leak = cheating) the examples only teach the "boundary", not memorized answers

Fair comparison: run on gpt-4o like baseline+threshold (finding 7, F1 0.725) -- only the prompt differs
threshold chosen leave-fold-out as before

Run:
  python precision_fix.py --score       call gpt-4o 127 times (~$0.09)
  python precision_fix.py --report      analyze + paired comparison vs plain prompt (free)
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
from baseline import metrics

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_CSV = os.path.join(HERE, "gold.csv")
PROB_CSV = os.path.join(HERE, "precision_fix_probs.csv")
PLAIN_CSV = os.path.join(HERE, "frontier_probs_gpt-4o.csv")  # original prompt, gpt-4o, F1 0.725
MODEL = "gpt-4o"
N_FOLDS, SEED = 5, 42

# original prompt + an explicit "pretense" rule + synthetic few-shot (not gold) targeting balanced-review FPs
DETECT_V2 = """ตัดสินว่าข้อความภาษาไทยนี้ "ประชด/เสียดสี" หรือไม่
ประชด = เจตนาจริงตรงข้ามกับความหมายผิวเผิน เพื่อเหน็บหรือแสดงความไม่พอใจ

**หัวใจ: ประชดต้องมี "การเสแสร้ง" -- แกล้งชม/แกล้งขอบคุณทั้งที่จริงไม่พอใจ**
กรณีที่คน "มักเข้าใจผิดว่าประชด" แต่จริงๆ ไม่ใช่ (ตอบ 0):
  - รีวิวสมดุล: ชมจริงบางจุด + ติจริงบางจุด อยู่ด้วยกันตามจริง = **ไม่ประชด (ไม่มีการเสแสร้ง)**
  - บ่น/ตำหนิตรงๆ ล้วนๆ ไม่มีแกล้งชม = ไม่ประชด
  - เล่าเหตุการณ์/อ้างคำพูดคนอื่น เฉยๆ = ไม่ประชด

ตัวอย่าง (สังเคราะห์ ไม่ใช่ข้อสอบ):
  "อาหารอร่อยดี รสชาติใช้ได้ แต่ที่จอดรถหายากไปหน่อย" -> 0  (ชมจริง+ติจริง = รีวิวสมดุล)
  "พนักงานบริการดีมากจริงใจ แต่คิวยาวรอนาน" -> 0  (สมดุล ไม่ได้เสแสร้ง)
  "ขอบคุณมากที่ทำให้รอ 2 ชั่วโมง บริการเยี่ยมจริงๆ นะ" -> 1  (แกล้งขอบคุณ/แกล้งชม = เสแสร้ง)
  "ร้านนี้แย่มาก ช้าและแพง" -> 0  (บ่นตรงๆ ไม่มีแกล้งชม)

ตอบเป็น JSON เท่านั้น: {"label": "1" หรือ "0"}
1 = ประชด, 0 = ไม่ประชด"""


def cost(i, o):
    return i / 1e6 * 2.50 + o / 1e6 * 10.00   # gpt-4o


def score_one(client, text):
    r = client.chat.completions.create(
        model=MODEL, max_tokens=20, response_format={"type": "json_object"},
        logprobs=True, top_logprobs=20,
        messages=[{"role": "system", "content": DETECT_V2},
                  {"role": "user", "content": f"ข้อความ: {text}"}],
    )
    in_tok, out_tok = r.usage.prompt_tokens, r.usage.completion_tokens
    for tok in (r.choices[0].logprobs.content or []):
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
    from openai import OpenAI
    client = OpenAI(timeout=30.0, max_retries=3)

    probs, ti, to = [], 0, 0
    t0 = time.time()
    for n, text in enumerate(df["text"], 1):
        try:
            p, i, o = score_one(client, text)
        except Exception as e:
            print(f"\n  item {n} failed: {type(e).__name__}: {e}")
            p, i, o = float("nan"), 0, 0
        probs.append(p); ti += i; to += o
        print(f"  {n}/{len(df)}", end="\r", flush=True)
    out = df[["text", "label"]].copy()
    out["prob"] = probs
    out.to_csv(PROB_CSV, index=False, encoding="utf-8-sig")
    print(f"\ncalled {len(df)} calls | {ti} in / {to} out | ${cost(ti,to):.4f} | {time.time()-t0:.0f}s")
    print(f"saved -> {PROB_CSV}  |  next: python precision_fix.py --report")


def loo_pred(probs, y):
    """leave-fold-out threshold -> return pred array"""
    def f1_at(p, yy, t):
        pr = (p >= t).astype(int); tp = ((pr == 1) & (yy == 1)).sum()
        fp = ((pr == 1) & (yy == 0)).sum(); fn = ((pr == 0) & (yy == 1)).sum()
        P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
        return 2*P*R/(P+R) if P+R else 0
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    pred = np.zeros(len(probs), dtype=int)
    for tr, te in skf.split(probs, y):
        t = max(np.unique(probs[tr]), key=lambda x: f1_at(probs[tr], y[tr], x))
        pred[te] = (probs[te] >= t).astype(int)
    return pred


def f1(y, p):
    tp = (y & p).sum(); fp = (~y & p).sum(); fn = (y & ~p).sum()
    return 2*tp/(2*tp+fp+fn) if tp else 0.0


def do_report():
    if not os.path.exists(PROB_CSV):
        sys.exit(f"{PROB_CSV} not found -- run --score first")
    fix = pd.read_csv(PROB_CSV, dtype={"text": str, "label": str}).fillna("")
    fix["label"] = fix["label"].str.strip()
    y = fix["label"].astype(int).values
    pf = loo_pred(np.nan_to_num(fix["prob"].astype(float).values, nan=0.0), y)
    _, prec, rec, f1v, (tn, fp, fn, tp) = metrics(fix["label"].tolist(), [str(v) for v in pf])
    print(f"detector + few-shot (balanced review=0) on {MODEL} + threshold leave-fold-out:")
    print(f"  F1 {f1v:.3f} | prec {prec:.3f} | recall {rec:.3f} | TP {tp} FP {fp} FN {fn}\n")

    # paired comparison vs the original prompt (gpt-4o, F1 0.725) -- item by item
    if not os.path.exists(PLAIN_CSV):
        print("(no frontier_probs_gpt-4o.csv -> skipping paired comparison)")
        return
    plain = pd.read_csv(PLAIN_CSV, dtype={"text": str, "label": str}).fillna("")
    plain["label"] = plain["label"].str.strip()
    pp = loo_pred(np.nan_to_num(plain["prob"].astype(float).values,
                                nan=0.0), plain["label"].astype(int).values)
    plain_pred = pd.Series(pp.astype(bool), index=plain["text"].values)
    fix_pred = pd.Series(pf.astype(bool), index=fix["text"].values)
    lab = pd.Series(fix["label"].values == "1", index=fix["text"].values)
    common = plain_pred.index.intersection(fix_pred.index)
    yy = lab.loc[common].values; pa = plain_pred.loc[common].values; pb = fix_pred.loc[common].values
    n = len(common)
    rng = np.random.default_rng(0)
    d = np.array([f1(yy[i], pb[i]) - f1(yy[i], pa[i]) for i in (rng.integers(0, n, n) for _ in range(5000))])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"paired comparison (n={n}) vs the original prompt (gpt-4o):")
    print(f"  plain prompt      F1 {f1(yy,pa):.3f}")
    print(f"  + balanced few-shot   F1 {f1(yy,pb):.3f}")
    print(f"  delta {f1(yy,pb)-f1(yy,pa):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"P(not better)={np.mean(d<=0)*100:.0f}%")
    nb = ((pb == yy) & (pa != yy)).sum(); na = ((pa == yy) & (pb != yy)).sum()
    print(f"  McNemar: few-shot right-plain wrong {nb} | plain right-few-shot wrong {na}")
    print(f"  *** cost is not equal: few-shot makes the prompt longer -> input tokens ~2x ($0.19 vs $0.094)")
    print(f"      same call count (127) but 'free' only in call count, not in dollars")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.score:
        do_score()
    elif a.report:
        do_report()
    else:
        ap.print_help()
