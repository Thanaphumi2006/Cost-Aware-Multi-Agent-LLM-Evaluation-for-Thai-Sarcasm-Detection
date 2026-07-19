# -*- coding: utf-8 -*-
"""ระบบ ⑧ — ดึง "คะแนน" (ไม่ใช่แค่ 0/1) ออกจาก GPT screener แล้วปรับ threshold

ทำไม: ทุกระบบในโปรเจกต์นี้ตอบ 0/1 ตายตัวที่จุดทำงานจุดเดียว
  -> คนเอาไปใช้จริงเลือกจุดทำงานเองไม่ได้ (งานคัดกรองอยากได้ recall / งานออโต้รีพลายอยากได้ precision)
  -> และเราไม่รู้ด้วยซ้ำว่า baseline "เกือบ" ตอบอีกทางแค่ไหน

วิธี: เรียก DETECT_SYS ตัวเดิมเป๊ะ (prompt เดิม, json_object เดิม) แต่ขอ logprobs มาด้วย
  แล้วอ่าน P("1") ตรงตำแหน่ง token ที่เป็นค่า label -> ได้คะแนนต่อเนื่อง 0..1
  **ไม่แตะ prompt** -> ยังเทียบกับ baseline เดิมได้ตรงๆ (ตัวแปรเดียวที่เพิ่มคือ "อ่านคะแนนออกมา")

ต่างจาก cascade (ข้อ 6) ตรงไหน: ข้อ 6 พิสูจน์ว่า WCB "จัดอันดับไม่เป็น" -> เป็นด่านคัดกรองไม่ได้
  ไฟล์นี้ถามคำถามเดียวกันกับ GPT: **GPT จัดอันดับเป็นไหม** ถ้าเป็น -> ปรับ threshold ซื้อ precision ได้ฟรี
  (ฟรีจริง: ไม่ต้องเพิ่ม call เลย คะแนนได้มาจาก call เดิมที่จ่ายไปแล้ว)

เลือก threshold ยังไงไม่ให้โกง: leave-fold-out เหมือน cascade.py
  ข้อใน fold k ใช้ threshold ที่เลือกจากอีก 4 folds เท่านั้น -> ไม่มีข้อไหนกำหนด threshold ที่ตัดสินตัวเอง

รัน:
  python gpt_threshold.py --score       ยิง GPT 127 ครั้งเก็บคะแนน (~$0.05) -> gpt_screener_probs.csv
  python gpt_threshold.py --sweep       วิเคราะห์ threshold จากคะแนนที่เก็บไว้ (ฟรี ไม่ยิง API ซ้ำ)
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
# override ได้ด้วย env เดียวกับ baseline.py: GOLD_CSV + EVAL_DIR
EVAL_DIR = os.environ.get("EVAL_DIR", HERE)
os.makedirs(EVAL_DIR, exist_ok=True)
GOLD_CSV = os.environ.get("GOLD_CSV", os.path.join(HERE, "gold.csv"))
PROB_CSV = os.path.join(EVAL_DIR, "gpt_screener_probs.csv")
# ชื่อนี้ compare_systems.py (glob multiagent_preds_gpt*.csv) จะเก็บไปเทียบ bootstrap/McNemar ให้เอง
OUT_CSV = os.path.join(EVAL_DIR, "multiagent_preds_gpt_threshold.csv")
IN_P, OUT_P = PRICE_PER_MTOK["gpt"]
N_FOLDS, SEED = 5, 42


def cost(i, o):
    return i / 1e6 * IN_P + o / 1e6 * OUT_P


def score_one(client, text):
    """เรียก DETECT_SYS ตัวเดิม + ขอ logprobs -> คืน (P(ประชด), in_tok, out_tok)

    ผลลัพธ์เป็น JSON {"label": "1"} -> token ที่เป็นค่า label คือตัว '1' หรือ '0' โดดๆ
    อ่าน top_logprobs ตรงตำแหน่งนั้น แล้ว normalize เฉพาะ '0' กับ '1' (มวลที่เหลือเป็น noise ทิ้งได้)"""
    r = client.chat.completions.create(
        model=multiagent.MODELS["gpt"], max_tokens=20,
        response_format={"type": "json_object"},
        logprobs=True, top_logprobs=20,
        messages=[{"role": "system", "content": multiagent.DETECT_SYS},
                  {"role": "user", "content": f"ข้อความ: {text}"}],
    )
    in_tok, out_tok = r.usage.prompt_tokens, r.usage.completion_tokens
    content = r.choices[0].logprobs.content or []

    # หา token ที่เป็น "ค่า" ของ label (ตัวเลขโดดๆ) -- ไม่ใช่ key ไม่ใช่เครื่องหมายวรรคตอน
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

    # fallback: อ่านไม่ได้ -> ใช้คำตอบ hard เป็น 0/1 (เกิดยาก แต่อย่าให้พังทั้งรัน)
    try:
        v = str(json.loads(r.choices[0].message.content or "{}").get("label", "")).strip()
        return (1.0 if v == "1" else 0.0), in_tok, out_tok
    except json.JSONDecodeError:
        return float("nan"), in_tok, out_tok


def do_score():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ต้องมี OPENAI_API_KEY")
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
        print(f"  {n}/{len(df)}  P(ประชด)={p:.3f}", end="\r", flush=True)

    out = df[["text", "label"]].copy()
    out["prob"] = probs
    out.to_csv(PROB_CSV, index=False, encoding="utf-8-sig")
    print(f"\nยิง {len(df)} calls | {ti} in / {to} out tokens | ${cost(ti, to):.4f} | {time.time()-t0:.0f}s")
    print(f"บันทึก -> {PROB_CSV}")
    print("ต่อไป: python gpt_threshold.py --sweep   (ฟรี)")


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
        sys.exit(f"ไม่พบ {PROB_CSV} -- รัน --score ก่อน")
    df = pd.read_csv(PROB_CSV, dtype={"text": str, "label": str}).fillna("")
    df["label"] = df["label"].str.strip()
    y = df["label"].astype(int).values
    probs = df["prob"].astype(float).values

    print(f"gold {len(df)} ข้อ | ประชด {int(y.sum())}")
    print(f"การกระจายคะแนน: min {probs.min():.3f} · p25 {np.percentile(probs,25):.3f} · "
          f"median {np.percentile(probs,50):.3f} · p75 {np.percentile(probs,75):.3f} · max {probs.max():.3f}")
    n_extreme = int(((probs < 0.01) | (probs > 0.99)).sum())
    print(f"คะแนนที่สุดขั้ว (<0.01 หรือ >0.99): {n_extreme}/{len(df)} ข้อ "
          f"-> ถ้าเยอะแปลว่า GPT 'มั่นใจเกิน' และ threshold แทบไม่มีอะไรให้ปรับ\n")

    # ---- PR curve เต็มเส้น (ยังไม่ใช่ผลสุดท้าย -- อันนี้เลือก threshold บนข้อมูลชุดเดียวกัน = โกง) ----
    print(f"{'tau':>6} {'prec':>6} {'recall':>7} {'F1':>6} {'TP':>3} {'FP':>3} {'FN':>3}")
    print("-" * 42)
    for tau in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        prec, rec, f1, tp, fp, fn = at(probs, y, tau)
        print(f"{tau:>6.2f} {prec:>6.3f} {rec:>7.3f} {f1:>6.3f} {tp:>3} {fp:>3} {fn:>3}")

    best_tau = max(np.unique(probs), key=lambda t: at(probs, y, t)[2])
    bp, br, bf, *_ = at(probs, y, best_tau)
    print(f"\nthreshold ที่ดีที่สุด (เลือกบน gold ทั้งชุด = ตัวเลข 'โกง' ห้ามรายงาน):")
    print(f"  tau {best_tau:.3f} -> F1 {bf:.3f} (prec {bp:.3f} recall {br:.3f})")

    # ---- ตัวเลขที่รายงานได้จริง: leave-fold-out ----
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pred = np.zeros(len(df), dtype=int)
    taus = []
    for tr, te in skf.split(probs, y):
        t = max(np.unique(probs[tr]), key=lambda x: at(probs[tr], y[tr], x)[2])  # เลือกจาก 4 folds
        pred[te] = (probs[te] >= t).astype(int)                                   # ใช้กับ fold ที่กันไว้
        taus.append(t)
    _, prec, rec, f1, (tn, fp, fn, tp) = metrics(df["label"].tolist(), [str(p) for p in pred])

    out = df[["text", "label"]].copy()
    out["pred"] = [str(p) for p in pred]
    out["prob"] = probs
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'='*62}")
    print(f"GPT screener + threshold ปรับแบบ leave-fold-out (ตัวเลขที่รายงานได้)")
    print(f"  tau ต่อ fold : {', '.join(f'{t:.3f}' for t in taus)}")
    print(f"  F1 {f1:.3f} | prec {prec:.3f} | recall {rec:.3f} | TP {tp} FP {fp} FN {fn}")
    print(f"  LLM calls 127 (เท่า baseline เป๊ะ) | ค่าใช้จ่ายเท่า baseline $0.094")
    print(f"เทียบ: baseline @argmax F1 0.690 | v2 (2 agents) F1 0.744 $0.169")
    print(f"บันทึก -> {OUT_CSV}")
    print(f"{'='*62}")
    print("ต่อไป: python compare_systems.py  (bootstrap + McNemar)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="ยิง GPT เก็บคะแนน (เสียเงิน ~$0.05)")
    ap.add_argument("--sweep", action="store_true", help="วิเคราะห์ threshold (ฟรี)")
    a = ap.parse_args()
    if a.score:
        do_score()
    elif a.sweep:
        do_sweep()
    else:
        ap.print_help()
