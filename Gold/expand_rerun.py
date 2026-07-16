# -*- coding: utf-8 -*-
"""หลังขยาย gold เป็น 157 ข้อ (pos 45): ให้ 2 ระบบหลักทำนาย 30 ข้อใหม่ แล้วเทียบ paired ใหม่

ระบบที่เทียบ (คู่หลักจาก finding 8):
  - baseline+threshold (เอเจนต์เดี่ยว อ่าน logprob) : ต่อจาก frontier_probs_gpt-4o.csv
  - v2 multi-agent (detector->verifier)             : ต่อจาก multiagent_preds_gpt_conservative.csv
ยิงเฉพาะข้อใหม่ที่ยังไม่มี pred -> ประหยัด (ข้อเดิม 127 ใช้ของเดิม)

*** อคติที่ต้องประกาศ: 30 ข้อใหม่ถูกเลือกด้วยคะแนนโมเดล (P>0.2) ไม่ใช่สุ่ม ***
    -> เข้าข้างระบบ logprob-threshold เล็กน้อย · absolute F1 บนเซ็ตนี้เทียบกับ 127 เดิมไม่ได้
    -> อ่านได้แค่ "paired บนเซ็ต 157 เดียวกัน" และเผื่อใจว่าเอนเข้าหา threshold

รัน: python expand_rerun.py            (ยิงข้อใหม่ ~$0.06 แล้วเทียบ)
     python expand_rerun.py --compare  (เทียบอย่างเดียว ถ้ายิงครบแล้ว)
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
GOLD = os.path.join(HERE, "gold_expanded.csv")                    # 157 -- ไม่แตะ gold.csv canonical
# ไฟล์ pred แยกเฉพาะ expansion -- seed จาก canonical 127 แล้วเติม 30 ข้อใหม่ (ไม่ทับของเดิม)
THRESH_CANON = os.path.join(HERE, "frontier_probs_gpt-4o.csv")
V2_CANON = os.path.join(HERE, "multiagent_preds_gpt_conservative.csv")
THRESH_PROBS = os.path.join(HERE, "frontier_probs_gpt-4o_expanded.csv")
V2_PREDS = os.path.join(HERE, "multiagent_preds_gpt_conservative_expanded.csv")
MODEL = "gpt-4o"


def _seed(canon, expanded):
    """สร้างไฟล์ expanded จาก canonical ครั้งแรก -> ยิงเฉพาะข้อใหม่พอ ไม่ยิงซ้ำ 127 เดิม"""
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
        sys.exit("ต้องมี OPENAI_API_KEY")
    g = gold_df()
    _seed(THRESH_CANON, THRESH_PROBS)      # คัดลอก 127 เดิมมาเป็นฐาน ยิงแค่ 30 ใหม่
    _seed(V2_CANON, V2_PREDS)
    from openai import OpenAI
    client = OpenAI(timeout=30.0, max_retries=3)

    # --- 1) threshold: เติม logprob ให้ข้อใหม่ ---
    tp = pd.read_csv(THRESH_PROBS, dtype=str).fillna("")
    have = set(tp["text"])
    new = [t for t in g["text"] if t not in have]
    print(f"threshold: มี {len(have)} · ต้องยิงใหม่ {len(new)}")
    rows = []
    for n, t in enumerate(new, 1):
        rows.append({"text": t, "label": g.loc[g["text"] == t, "label"].iloc[0],
                     "prob": score_logprob(client, t)})
        print(f"  logprob {n}/{len(new)}", end="\r", flush=True)
    if rows:
        tp = pd.concat([tp, pd.DataFrame(rows)], ignore_index=True)
        tp.to_csv(THRESH_PROBS, index=False, encoding="utf-8-sig")
    print(f"\n  -> {THRESH_PROBS} มี {len(tp)} ข้อ")

    # --- 2) v2 multi-agent: เติม pred ให้ข้อใหม่ ---
    vp = pd.read_csv(V2_PREDS, dtype=str).fillna("")
    have = set(vp["text"])
    new = [t for t in g["text"] if t not in have]
    print(f"v2: มี {len(have)} · ต้องยิงใหม่ {len(new)}")
    rows = []
    for n, t in enumerate(new, 1):
        r = multiagent.run_pipeline(client, t)
        rows.append({"text": t, "label": g.loc[g["text"] == t, "label"].iloc[0], "pred": r["pred"]})
        print(f"  v2 {n}/{len(new)}", end="\r", flush=True)
    if rows:
        keep = [c for c in vp.columns if c in ("text", "label", "pred")]
        vp = pd.concat([vp[keep], pd.DataFrame(rows)], ignore_index=True)
        vp.to_csv(V2_PREDS, index=False, encoding="utf-8-sig")
    print(f"\n  -> {V2_PREDS} มี {len(vp)} ข้อ")


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
    print(f"เทียบใหม่บน gold ขยาย: n={n} (ประชด {npos})   [เดิม n=127 ประชด 30]")
    print(f"  baseline+threshold  F1 {f1(yy,pa):.3f}")
    print(f"  v2 multi-agent      F1 {f1(yy,pb):.3f}")
    print(f"  ต่าง (v2-thr) {f1(yy,pb)-f1(yy,pa):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"P(v2 ไม่ดีกว่า)={np.mean(d<=0)*100:.0f}%")
    print(f"  McNemar: v2 ถูก-thr ผิด {nb} | thr ถูก-v2 ผิด {na}")
    cross = "คร่อม 0 (ยังแยกไม่ออก)" if lo <= 0 <= hi else "ไม่คร่อม 0 (แยกออกแล้ว!)"
    print(f"  -> CI {cross}")
    print("=" * 64)
    print("อคติ: 30 ข้อใหม่เลือกด้วยคะแนนโมเดล -> เอนเข้าหา threshold · absolute F1 เทียบ 127 เดิมไม่ได้")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="เทียบอย่างเดียว ไม่ยิง")
    a = ap.parse_args()
    if not a.compare:
        do_score()
    do_compare()
