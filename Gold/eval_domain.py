# -*- coding: utf-8 -*-
"""วัด predict.py บนข้อมูล "โดเมนใหม่" — ตอบคำถามที่ยังไม่รู้: มันข้ามโดเมนได้ไหม

ทำไมสำคัญ: gold ทั้งหมดคือ Wongnai (รีวิวร้าน) + Wisesight (ทวีต)
ยังไม่มีหลักฐานเลยว่าโมเดลใช้ได้กับโดเมนอื่น (ข่าว/การเมือง/สินค้าเทค/คอมเมนต์ YouTube ฯลฯ)
นี่คือ "ความเสี่ยงที่ใหญ่สุด" ตอนเอาไป deploy จริง — ไฟล์นี้คือเครื่องมือปิดช่องนั้น

*** ต้องมีข้อมูลก่อน: CSV โดเมนใหม่ที่ "คนไทย label แล้ว" (ไม่ใช่โมเดล label) ***
    คอลัมน์: text, label (1=ประชด, 0=ไม่)  — อย่างน้อย ~30 ประชด ถึงจะมี CI ที่มีความหมาย
    label ด้วยเกณฑ์เดียวกับ gold (การเสแสร้ง — ดู labeling_rubric.md) ไม่งั้นเทียบไม่ได้

ใช้:
  export OPENAI_API_KEY=sk-...
  python eval_domain.py newsdomain.csv                 # วัด balanced (gpt-4.1-mini)
  python eval_domain.py newsdomain.csv --op high_recall
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))

# ผลบน gold (โดเมนเดิม) ไว้เทียบว่า "ตกไปเท่าไหร่เมื่อข้ามโดเมน"
GOLD_REF = {"balanced": dict(P=0.68, R=0.83, F1=0.75),
            "high_recall": dict(P=0.43, R=1.00, F1=0.61)}


def metrics(y, p):
    y, p = np.array(y), np.array(p)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    P = tp/(tp+fp) if tp+fp else 0.0; R = tp/(tp+fn) if tp+fn else 0.0
    F = 2*P*R/(P+R) if P+R else 0.0
    return P, R, F, (tp, fp, fn, tn)


def boot_f1_ci(y, p, n=5000, seed=0):
    y, p = np.array(y), np.array(p); rng = np.random.default_rng(seed); N = len(y); out = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        _, _, f, _ = metrics(y[i], p[i]); out.append(f)
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV โดเมนใหม่ (คอลัมน์ text,label) — label โดยคน")
    ap.add_argument("--op", default="balanced", choices=["balanced", "high_recall"])
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    a = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ต้องมี OPENAI_API_KEY")

    df = pd.read_csv(a.csv, dtype=str).fillna("")
    for c in (a.text_col, a.label_col):
        if c not in df.columns:
            sys.exit(f"ไม่มีคอลัมน์ '{c}' (มี: {list(df.columns)})")
    df[a.label_col] = df[a.label_col].str.strip()
    df = df[df[a.label_col].isin(["0", "1"])].reset_index(drop=True)
    y = df[a.label_col].astype(int).tolist()
    npos = sum(y)
    if npos < 10:
        print(f"⚠ ประชดแค่ {npos} ข้อ — น้อยเกินไป CI จะกว้างมาก (แนะนำ ≥30)")

    import predict
    det = predict.SarcasmDetector(operating=a.op)
    print(f"วัด {len(df)} ข้อ (ประชด {npos}) · {det.model} · จุดทำงาน {a.op}\n", flush=True)
    preds = []
    for n, t in enumerate(df[a.text_col], 1):
        preds.append(det.predict(t).get("label"))
        print(f"  {n}/{len(df)}", end="\r", flush=True)
    ok = [(yy, pp) for yy, pp in zip(y, preds) if pp in (0, 1)]
    y2, p2 = [a_ for a_, _ in ok], [b_ for _, b_ in ok]
    P, R, F, (tp, fp, fn, tn) = metrics(y2, p2)
    lo, hi = boot_f1_ci(y2, p2)
    g = GOLD_REF[a.op]

    print("\n" + "=" * 60)
    print(f"โดเมนใหม่ ({os.path.basename(a.csv)}): P {P:.3f} · R {R:.3f} · F1 {F:.3f}  [95% CI {lo:.3f}–{hi:.3f}]")
    print(f"  TP {tp} FP {fp} FN {fn} TN {tn} · cache hit {det.hits}/{det.hits+det.misses}")
    print(f"gold เดิม (โดเมน Wongnai/Wisesight): P {g['P']:.2f} · R {g['R']:.2f} · F1 {g['F1']:.2f}")
    drop = g["F1"] - F
    print("-" * 60)
    if drop > 0.10:
        print(f"⚠ F1 ตก {drop:+.3f} เมื่อข้ามโดเมน — มาก · โมเดลนี้ยัง 'ไม่ควรเชื่อ' นอกโดเมนเดิม")
    elif drop > 0.05:
        print(f"F1 ตก {drop:+.3f} — พอมี domain gap · ใช้ได้แต่ควรระวัง/ตั้ง threshold ใหม่ต่อโดเมน")
    else:
        print(f"F1 ต่าง {drop:+.3f} — ข้ามโดเมนได้ดี (แต่ดู CI: ถ้ากว้างแปลว่ายังฟันธงไม่ได้)")
    print("=" * 60)


if __name__ == "__main__":
    main()
