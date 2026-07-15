# -*- coding: utf-8 -*-
"""cost-quality frontier: สถาปัตยกรรมที่ดีที่สุด (baseline+threshold) บนหลายโมเดล

ธีสิสของโปรเจกต์ชื่อ "Cost-Aware" แต่ finding 1-8 วัด cost ของ *สถาปัตยกรรม* บนโมเดลเดียว (gpt-4o)
finding 7 บอกแล้วว่า "แกนที่ขยับ cost ได้จริงคือ threshold ไม่ใช่จำนวน agent"
ไฟล์นี้ไปอีกขั้น: **ขยับที่ตัวโมเดล** -- เอาสถาปัตยกรรมที่ถูกสุด+ดีสุด (เอเจนต์เดี่ยว + threshold)
ไปรันบนโมเดลราคาต่างกัน 30 เท่า แล้วดูว่า F1 ตกเท่าไหร่เมื่อประหยัด

วิธี: ต่อโมเดล -> ยิง DETECT_SYS เดิม + logprobs -> P(ประชด) -> เลือก threshold leave-fold-out (เหมือน gpt_threshold.py)
เก็บคะแนนดิบต่อโมเดลไว้ (frontier_probs_<model>.csv) -> รันซ้ำ/เปลี่ยนราคาได้โดยไม่ยิง API อีก

*** ราคาด้านล่างเป็นค่าที่ต้อง "ตรวจสอบกับราคาปัจจุบัน" -- token counts วัดจริงจาก API เป๊ะอยู่แล้ว
    ถ้าราคาเปลี่ยน แก้ที่ PRICE ตัวเดียว cost คำนวณใหม่เองหมด ***

รัน:
  python frontier.py --score        ยิงทุกโมเดล (เสียเงิน ~$0.20) เก็บคะแนน
  python frontier.py --report       วิเคราะห์ F1/cost ต่อโมเดล (ฟรี)
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
N_FOLDS, SEED = 5, 42

# (input, output) $/1M tokens -- *** ตรวจสอบกับราคาปัจจุบันก่อนอ้างอิง ***
PRICE = {
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini":  (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1":      (2.00, 8.00),
    "gpt-4o":       (2.50, 10.00),
}
MODELS = list(PRICE.keys())


def cost(model, i, o):
    ip, op = PRICE[model]
    return i / 1e6 * ip + o / 1e6 * op


def probs_path(model):
    return os.path.join(HERE, f"frontier_probs_{model}.csv")


def score_one(client, model, text):
    """DETECT_SYS เดิม + logprobs -> P(ประชด). เหมือน gpt_threshold.score_one แต่รับ model ได้"""
    r = client.chat.completions.create(
        model=model, max_tokens=20,
        response_format={"type": "json_object"},
        logprobs=True, top_logprobs=20,
        messages=[{"role": "system", "content": multiagent.DETECT_SYS},
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


def do_score(models):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ต้องมี OPENAI_API_KEY")
    df = pd.read_csv(GOLD_CSV, dtype=str).fillna("")
    df["label"] = df["label"].str.strip()
    df = df[df["label"].isin(["0", "1"])].reset_index(drop=True)
    # timeout ต่อ request -> กันไม่ให้ call เดียวค้างแล้วแขวนทั้งรัน (เจอมาแล้วรอบก่อน)
    from openai import OpenAI
    client = OpenAI(timeout=30.0, max_retries=3)

    for model in models:
        if os.path.exists(probs_path(model)):
            print(f"[ข้าม] {model} -- มีคะแนนแล้ว ({os.path.basename(probs_path(model))})")
            continue
        probs, ti, to = [], 0, 0
        t0 = time.time()
        for n, text in enumerate(df["text"], 1):
            try:
                p, i, o = score_one(client, model, text)
            except Exception as e:
                print(f"\n  {model} ข้อ {n} พัง: {type(e).__name__}: {e}")
                p, i, o = float("nan"), 0, 0
            probs.append(p); ti += i; to += o
            print(f"  {model}: {n}/{len(df)}", end="\r", flush=True)
        out = df[["text", "label"]].copy()
        out["prob"] = probs
        out.attrs = {}
        out.to_csv(probs_path(model), index=False, encoding="utf-8-sig")
        # เก็บ token รวมไว้ท้ายไฟล์ผ่านชื่อคอลัมน์แยก (แถวแรกพอ) -- ง่ายกว่าเปิดไฟล์ meta
        meta = pd.DataFrame([{"model": model, "in_tok": ti, "out_tok": to,
                              "cost": cost(model, ti, to)}])
        meta.to_csv(os.path.join(HERE, f"frontier_meta_{model}.csv"), index=False)
        print(f"  {model}: {len(df)} calls | {ti} in / {to} out | ${cost(model,ti,to):.4f} | {time.time()-t0:.0f}s")
    print("ต่อไป: python frontier.py --report")


def loo_f1(probs, y):
    """เลือก threshold แบบ leave-fold-out -> คืน (f1, prec, rec, taus) ; กัน threshold leak"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    def f1_at(p, yy, t):
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (yy == 1)).sum()); fp = int(((pred == 1) & (yy == 0)).sum())
        fn = int(((pred == 0) & (yy == 1)).sum())
        pr = tp/(tp+fp) if tp+fp else 0.0; rc = tp/(tp+fn) if tp+fn else 0.0
        return (2*pr*rc/(pr+rc) if pr+rc else 0.0)

    pred = np.zeros(len(probs), dtype=int)
    taus = []
    for tr, te in skf.split(probs, y):
        t = max(np.unique(probs[tr]), key=lambda x: f1_at(probs[tr], y[tr], x))
        pred[te] = (probs[te] >= t).astype(int)
        taus.append(float(t))
    _, prec, rec, f1, _ = metrics([str(v) for v in y], [str(v) for v in pred])
    return f1, prec, rec, taus


def do_report(models):
    rows = []
    for model in models:
        pp = probs_path(model)
        if not os.path.exists(pp):
            continue
        d = pd.read_csv(pp, dtype={"text": str, "label": str}).fillna("")
        d = d[d["prob"] != ""]
        y = d["label"].str.strip().astype(int).values
        probs = d["prob"].astype(float).values
        if np.isnan(probs).any():
            probs = np.nan_to_num(probs, nan=0.0)
        f1, prec, rec, taus = loo_f1(probs, y)
        mp = os.path.join(HERE, f"frontier_meta_{model}.csv")
        c = pd.read_csv(mp)["cost"].iloc[0] if os.path.exists(mp) else float("nan")
        rows.append(dict(model=model, f1=f1, prec=prec, rec=rec, cost=c,
                         cheap=PRICE[model][0]))
    if not rows:
        sys.exit("ยังไม่มีคะแนน -- รัน --score ก่อน")

    rows.sort(key=lambda r: r["cheap"])
    print(f"cost-quality frontier | เอเจนต์เดี่ยว + threshold (leave-fold-out) | gold 127 ข้อ\n")
    print(f"{'model':<15}{'F1':>7}{'prec':>7}{'recall':>8}{'$/รัน':>9}{'$/1M in':>9}")
    print("-" * 56)
    best = max(rows, key=lambda r: r["f1"])
    for r in rows:
        star = "  <- F1 สูงสุด" if r is best else ""
        cheapest = "  (ถูกสุด)" if r is rows[0] else ""
        print(f"{r['model']:<15}{r['f1']:>7.3f}{r['prec']:>7.3f}{r['rec']:>8.3f}"
              f"{('$%.4f'%r['cost']) if not math.isnan(r['cost']) else '  n/a':>9}"
              f"{'$%.2f'%r['cheap']:>9}{star or cheapest}")

    gpt4o = next((r for r in rows if r["model"] == "gpt-4o"), None)
    if gpt4o and rows[0] is not gpt4o:
        c0, cg = rows[0], gpt4o
        df1 = cg["f1"] - c0["f1"]
        xcost = cg["cost"] / c0["cost"] if c0["cost"] else float("nan")
        print(f"\nถูกสุด ({c0['model']}) vs gpt-4o: F1 ต่าง {df1:+.3f} | gpt-4o แพงกว่า {xcost:.1f}x")
        print("อ่านยังไง: ถ้า F1 ต่างน้อยแต่ราคาต่างหลายเท่า -> โมเดลถูกคือจุดที่คุ้มบน frontier")
    print("\n(ราคาใน PRICE = ค่าที่ต้องตรวจกับราคาปัจจุบัน; token counts วัดจริงจาก API)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--models", nargs="+", default=MODELS)
    a = ap.parse_args()
    if a.score:
        do_score(a.models)
    elif a.report:
        do_report(a.models)
    else:
        ap.print_help()
