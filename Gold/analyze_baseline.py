# -*- coding: utf-8 -*-
"""แยกผล baseline ตาม 'ที่มาของข้อ' เพื่อดู self-selection bias โดยไม่ต้องใช้โมเดลเจ้าอื่น

แนวคิด: ประชด/ไม่ประชดใน gold มาจากสองแหล่งที่ GPT-4o "เคยเห็น" ต่างกัน
  - กลุ่ม keyword : gold รุ่นแรก 102 ข้อ  -> คัดด้วยคีย์เวิร์ด GPT-4o ไม่เคยแตะ
  - กลุ่ม harvest : 25 ข้อที่เพิ่งเติม     -> GPT-4o เป็นคนคัดมาเองว่า "น่าจะประชด"
ถ้า GPT-4o ทำได้ดีกว่าอย่างชัดเจนบนกลุ่มที่ตัวเองคัด = หลักฐาน self-selection bias

รัน: python analyze_baseline.py
"""

import os
import random
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, "baseline_preds_gpt.csv")
BACKUP = os.path.join(HERE, "gold_backup.csv")


def prf(y_true, y_pred, pos="1"):
    tp = sum(t == pos and p == pos for t, p in zip(y_true, y_pred))
    fp = sum(t != pos and p == pos for t, p in zip(y_true, y_pred))
    fn = sum(t == pos and p != pos for t, p in zip(y_true, y_pred))
    tn = sum(t != pos and p != pos for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return acc, prec, rec, f1, (tn, fp, fn, tp)


d = pd.read_csv(PRED, dtype=str).fillna("")
d = d[d.pred.isin(["0", "1"])]
b = pd.read_csv(BACKUP, dtype=str).fillna("")
old_texts = set(b.text)
d["stratum"] = ["keyword" if t in old_texts else "harvest" for t in d.text]

print(f"n = {len(d)}  (keyword {sum(d.stratum=='keyword')} / harvest {sum(d.stratum=='harvest')})\n")

# ---------- 1) เทียบกับเส้นฐานโง่ๆ ----------
yt = d.label.tolist()
print("== [1] GPT-4o ชนะการเดามั่วจริงไหม ==")
for name, yp in [
    ("ทายว่า 'ไม่ประชด' ทุกข้อ", ["0"] * len(d)),
    ("ทายว่า 'ประชด' ทุกข้อ", ["1"] * len(d)),
    ("GPT-4o zero-shot", d.pred.tolist()),
]:
    acc, prec, rec, f1, _ = prf(yt, yp)
    print(f"  {name:<26} acc {acc:.3f}  prec {prec:.3f}  rec {rec:.3f}  F1 {f1:.3f}")

# ---------- 2) แยกตามที่มา ----------
print("\n== [2] self-selection bias: GPT-4o เก่งกว่าบนข้อที่ตัวเองคัดไหม ==")
for s in ["keyword", "harvest"]:
    g = d[d.stratum == s]
    acc, prec, rec, f1, (tn, fp, fn, tp) = prf(g.label.tolist(), g.pred.tolist())
    npos, nneg = (g.label == "1").sum(), (g.label == "0").sum()
    print(f"\n  [{s}]  ประชด {npos} / ไม่ประชด {nneg}")
    print(f"    recall (จับประชดได้)      : {rec:.3f}   ({tp}/{npos})")
    print(f"    false-positive rate       : {fp/nneg if nneg else 0:.3f}   ({fp}/{nneg} ข้อที่ไม่ประชดแต่ถูกทายว่าประชด)")
    print(f"    precision / F1            : {prec:.3f} / {f1:.3f}")

# ---------- 3) คุมตัวแปรแหล่งข้อมูล (harvest เป็น wisesight ล้วน) ----------
print("\n== [3] เทียบเฉพาะ wisesight (คุมความยาว/แหล่ง) ==")
w = d[d.source == "wisesight"]
for s in ["keyword", "harvest"]:
    g = w[w.stratum == s]
    if not len(g):
        continue
    _, prec, rec, f1, (tn, fp, fn, tp) = prf(g.label.tolist(), g.pred.tolist())
    npos, nneg = (g.label == "1").sum(), (g.label == "0").sum()
    fpr = fp / nneg if nneg else float("nan")
    print(f"  [{s:<7}] ประชด {npos:>2} / ไม่ประชด {nneg:>2} | recall {rec:.3f} ({tp}/{npos}) | FPR {fpr:.3f} ({fp}/{nneg})")

# ---------- 4) bootstrap CI ของ F1 (กัน n เล็กแล้วอ่านเกินจริง) ----------
print("\n== [4] F1 ของ GPT-4o มั่วแค่ไหน (bootstrap 2000 รอบ) ==")
random.seed(42)
rows = list(zip(yt, d.pred.tolist()))
f1s = []
for _ in range(2000):
    samp = [rows[random.randrange(len(rows))] for _ in rows]
    f1s.append(prf([a for a, _ in samp], [p for _, p in samp])[3])
f1s.sort()
lo, hi = f1s[int(0.025 * len(f1s))], f1s[int(0.975 * len(f1s))]
print(f"  F1 = {prf(yt, d.pred.tolist())[3]:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
print("  -> ถ้า multi-agent ได้ F1 ไม่หลุดออกนอกช่วงนี้ ยังอ้างว่า 'ดีกว่า' ไม่ได้")

# ---------- 5) FP ไปกองอยู่ที่ไหน ----------
print("\n== [5] ข้อที่ทายผิดว่า 'ประชด' (false positive) กองอยู่ที่ไหน ==")
fp_rows = d[(d.label == "0") & (d.pred == "1")]
print(f"  ทั้งหมด {len(fp_rows)} ข้อ")
print(fp_rows.groupby(["source", "stratum"]).size().to_string())
