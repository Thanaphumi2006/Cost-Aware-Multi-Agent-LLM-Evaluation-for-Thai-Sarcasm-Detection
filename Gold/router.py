# -*- coding: utf-8 -*-
"""ระบบ ⑧ — Micro-router 3 ทาง: ประชด / ไม่ประชด / **ไม่แน่ใจ** (ส่งต่อให้ LLM)

cascade.py ใช้ threshold *เดียว* -> ทุกข้อที่ WCB ว่า "ประชด" ถูกส่งไป verifier
router.py ใช้ threshold *สองตัว* -> เก็บเฉพาะข้อที่ WCB "ไม่มั่นใจ" ไว้ให้ LLM
  prob < lo          -> ตัดสินเอง "ไม่ประชด"  (ฟรี, 0 calls)
  prob >= hi         -> ตัดสินเอง "ประชด"     (ฟรี, 0 calls)
  lo <= prob < hi    -> **ไม่แน่ใจ** -> ส่งให้ GPT ตัดสิน (จ่ายเฉพาะข้อพวกนี้)

ทำไมถึงควรได้ผล: finding 14 วัดไว้แล้วว่าวิธีฟรี (regex 555 = 0.590, kNN = 0.588,
WCB = 0.620) ชนะโมเดลเปิด 7-8B ทุกตัว -> "ชั้นฟรี" ที่ ~0.6 มีจริง
คำถามที่เหลือคือ **ต้องจ่ายเท่าไหร่ถึงจะไต่จาก 0.62 ไป 0.727 (GPT bot) ได้**
สคริปต์นี้ตอบด้วยการกวาด escalation budget b = 0%..100% แล้ววาด frontier

เลือก threshold ยังไงไม่ให้โกง: leave-fold-out เหมือน cascade.py เป๊ะ
  ข้อใน fold k ใช้ (tau, delta, tau_gpt) ที่เลือกจาก "อีก 4 folds เท่านั้น"
  -> ไม่มีข้อไหนมีส่วนกำหนด threshold ที่ตัดสินตัวมันเอง
b = 0   -> WCB ล้วน (ฟรีทั้งหมด)
b = 1   -> GPT ล้วน (จ่ายทุกข้อ) = ปลายทางเดียวกับ GPT bot
ระหว่างกลางคือของใหม่ที่สคริปต์นี้วัด

การ join ไฟล์: **join ด้วยตำแหน่ง ไม่ใช่ด้วย text** (ดู HANDOFF.md -- score_lm_eval.py
เขียนขึ้นบรรทัดใหม่ทับเป็น space ทำให้ join ด้วย text หล่นข้อ multi-line ทิ้งเงียบๆ)
ทุกไฟล์ derive จาก gold.csv ตามลำดับ -> assert ว่า label ตรงกันก่อน แล้วค่อย index ตามตำแหน่ง

รัน:  python router.py              (ฟรี ไม่ยิง API -- ใช้ prob ที่คำนวณไว้แล้ว)
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
# ค่าเฉลี่ยจริงของ 1 call จาก v2 (391 in / 7 out) -- ใช้ตัวเดียวกับ cascade.py จะได้เทียบกันได้
COST_PER_CALL = 391 / 1e6 * IN_P + 7 / 1e6 * OUT_P
BUDGETS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]


def f1_at(probs, y, tau):
    pred = (probs >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def best_tau(probs, y):
    """threshold ที่ให้ F1 สูงสุดบนข้อมูลที่ให้มา (ใช้เฉพาะกับ 'อีก 4 folds')"""
    cands = np.unique(np.concatenate([probs, [0.0, 1.0]]))
    return float(max(cands, key=lambda t: f1_at(probs, y, t)))


def route(df, seed, budget, gpt_prob, tau_mode="fixed"):
    """คืน (pred[], band[], n_escalated) -- band: 0=ไม่ประชด(ฟรี) 1=ประชด(ฟรี) 2=ไม่แน่ใจ(จ่าย)

    delta = รัศมีความไม่มั่นใจรอบ tau. ข้อที่ |prob - tau| < delta ถือว่า 'ไม่แน่ใจ'
    เลือก delta จากอีก 4 folds ให้ได้สัดส่วน escalate = budget (uncertainty sampling มาตรฐาน)

    tau_mode: "fixed" = ใช้ 0.5 ตรงๆ | "tuned" = จูน tau แบบ leave-fold-out
      ดีฟอลต์เป็น fixed เพราะ **วัดแล้วว่าจูนแล้วแย่ลง** ที่ n=127 (ดู --tau-mode ในผลลัพธ์)
      จูนบนอีก 4 folds ได้ F1 0.556 / ไม่จูนเลยได้ 0.590 -> การจูนคือการ overfit ล้วนๆ ตรงนี้
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
        # delta: ควอนไทล์ของระยะห่างจาก tau บนอีก 4 folds -> สัดส่วนที่ escalate ~= budget
        d_rest = np.abs(probs[rest] - tau)
        delta = float(np.quantile(d_rest, budget)) if budget > 0 else -1.0
        tau_gpt = best_tau(gpt_prob[rest], y[rest])   # เกณฑ์ของ GPT ก็เลือก leave-fold-out เช่นกัน

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
                    help="fixed=0.5 (ดีฟอลต์ ดีกว่า) | tuned=จูน leave-fold-out (overfit ที่ n นี้)")
    a = ap.parse_args()

    for p in (OOF_CSV, GPT_CSV):
        if not os.path.exists(p):
            sys.exit(f"ไม่พบ {p}")
    df = pd.read_csv(OOF_CSV, encoding="utf-8-sig")
    gpt = pd.read_csv(GPT_CSV, encoding="utf-8-sig")

    # join ตามตำแหน่ง -- ยืนยันก่อนว่าสองไฟล์เรียงตรงกันจริง (ห้าม join ด้วย text)
    if len(df) != len(gpt):
        sys.exit(f"จำนวนแถวไม่ตรง: oof {len(df)} vs gpt {len(gpt)}")
    a_lab = df["label"].astype(str).str.strip().tolist()
    b_lab = gpt["label"].astype(str).str.strip().tolist()
    if a_lab != b_lab:
        sys.exit(f"ลำดับ label ไม่ตรงกัน -> join ตามตำแหน่งไม่ได้ ({sum(x!=y for x,y in zip(a_lab,b_lab))} ข้อต่างกัน)")

    df["label"] = df["label"].astype(str).str.strip()
    y = df["label"].tolist()
    gpt_prob = gpt["prob"].values.astype(float)
    n = len(df)

    print(f"micro-router 3 ทาง | gold {n} ข้อ | seeds {a.seeds} | tau-mode {a.tau_mode}")
    print(f"ราคา verifier ${COST_PER_CALL:.5f}/call (391 in / 7 out -- ค่าจริงจาก v2)\n")
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

    # จุดที่รายงาน = budget *ต่ำสุด* ที่กู้กำไร F1 ได้ >= ครึ่งหนึ่งของที่ GPT ล้วนให้
    # (ห้ามใช้ "F1 ต่อ 1 call สูงสุด" -- เกณฑ์นั้นเข้าข้าง budget เล็กที่สุดเสมอ ไม่ได้บอกอะไร)
    lo_f1, hi_f1 = rows[0]["f1"], rows[-1]["f1"]
    need = lo_f1 + 0.5 * (hi_f1 - lo_f1)
    reached = [r for r in rows if r["escalated"] > 0 and r["f1"] >= need]
    knee = min(reached, key=lambda r: r["escalated"]) if reached else rows[-1]
    pred, band = preds_at[knee["budget"]]
    pr = df[["text", "label"]].copy()
    pr["pred"] = [str(v) for v in pred]
    pr["band"] = ["ไม่ประชด(ฟรี)" if b == 0 else "ประชด(ฟรี)" if b == 1 else "ไม่แน่ใจ->LLM"
                  for b in band]
    pr.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")

    base_f1 = rows[0]["f1"]
    full_f1 = rows[-1]["f1"]
    print("\n" + "=" * 72)
    print(f"b=0 (WCB ล้วน ฟรี)     F1 {base_f1:.3f} | 0 calls    | $0")
    print(f"b=1 (GPT ล้วน)         F1 {full_f1:.3f} | {n} calls  | ${n*COST_PER_CALL:.4f}")
    print(f"จุดคุ้มสุด b={knee['budget']:.2f}        F1 {knee['f1']:.3f} | {knee['escalated']:.0f} calls "
          f"| ${knee['cost']:.4f}  ({100*knee['escalated']/n:.0f}% ของข้อทั้งหมด)")
    if full_f1 > base_f1:
        recovered = 100 * (knee["f1"] - base_f1) / (full_f1 - base_f1)
        print(f"-> จ่าย {100*knee['escalated']/n:.0f}% ของราคาเต็ม ได้ {recovered:.0f}% ของกำไร F1 ที่ GPT ล้วนให้")
    print(f"บันทึก -> {os.path.basename(OUT_CSV)} , {os.path.basename(PRED_CSV)}")
    print("=" * 72)
    print("ต่อไป: python compare_systems.py  (bootstrap + McNemar เทียบกับระบบอื่น)")


if __name__ == "__main__":
    main()
