# -*- coding: utf-8 -*-
"""เทียบ baseline vs multi-agent อย่างถูกสถิติ (วัด gold ชุดเดียวกัน -> ต้องใช้ paired test)

ทำไมไม่ใช้ "F1 เกิน CI บนของ baseline":
  นั่นเทียบราวกับสองระบบวัดคนละชุด -> เข้มเกินไป
  ที่ถูกคือ paired bootstrap: resample ข้อเดียวกันทั้งคู่ แล้วดูผลต่าง F1 ต่อรอบ
  ถ้า 95% CI ของผลต่าง "ไม่คร่อม 0" = ต่างจริง ไม่ใช่บังเอิญ
  + McNemar: ดูเฉพาะข้อที่สองระบบไม่ตรงกัน

รัน: python compare_systems.py
"""
import glob
import os
import random
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))


def f1_of(label, pred, pos="1"):
    tp = sum(t == pos and p == pos for t, p in zip(label, pred))
    fp = sum(t != pos and p == pos for t, p in zip(label, pred))
    fn = sum(t == pos and p != pos for t, p in zip(label, pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0, prec, rec


def load(path):
    d = pd.read_csv(path, dtype=str).fillna("")
    d = d[d.pred.isin(["0", "1"])][["text", "label", "pred"]]
    return d.set_index("text")


EVAL_DIR = os.environ.get("EVAL_DIR", HERE)  # ชี้โฟลเดอร์ผลชุดอื่น (เช่น v2_results) ได้
base = load(os.path.join(EVAL_DIR, "baseline_preds_gpt.csv"))
variants = {}
for p in sorted(glob.glob(os.path.join(EVAL_DIR, "multiagent_preds_gpt*.csv"))):
    name = os.path.basename(p).replace("multiagent_preds_gpt", "").replace(".csv", "").strip("_") or "v?"
    variants[name] = load(p)

# ระบบ ③ WangchanBERTa (out-of-fold preds ครบ 127 ข้อ -> เทียบ paired ได้เหมือนกัน)
wcb = os.path.join(EVAL_DIR, "wangchanberta_preds.csv")
if os.path.exists(wcb):
    variants["wangchanberta"] = load(wcb)

print(f"baseline: {len(base)} ข้อ | multi-agent variants: {list(variants)}\n")
bf1, bp, br = f1_of(base.label, base.pred)
print(f"{'ระบบ':<16}{'F1':>8}{'prec':>8}{'recall':>8}")
print(f"{'baseline':<16}{bf1:>8.3f}{bp:>8.3f}{br:>8.3f}")
for name, d in variants.items():
    f1, pr, rc = f1_of(d.label, d.pred)
    print(f"{name:<16}{f1:>8.3f}{pr:>8.3f}{rc:>8.3f}")

random.seed(42)
N = 5000
for name, d in variants.items():
    # จัดให้เรียงข้อเดียวกัน (paired)
    common = base.index.intersection(d.index)
    b = base.loc[common]
    m = d.loc[common]
    lab = b.label.tolist()
    bp_, mp_ = b.pred.tolist(), m.pred.tolist()

    f1b, _, _ = f1_of(lab, bp_)
    f1m, _, _ = f1_of(lab, mp_)

    # paired bootstrap ของผลต่าง F1
    idx = list(range(len(lab)))
    diffs = []
    for _ in range(N):
        s = [random.randrange(len(idx)) for _ in idx]
        L = [lab[i] for i in s]
        diffs.append(f1_of(L, [mp_[i] for i in s])[0] - f1_of(L, [bp_[i] for i in s])[0])
    diffs.sort()
    lo, hi = diffs[int(0.025 * N)], diffs[int(0.975 * N)]
    p_worse = sum(1 for x in diffs if x <= 0) / N

    # McNemar: เฉพาะข้อที่ถูก/ผิดต่างกัน
    b_correct = [p == t for p, t in zip(bp_, lab)]
    m_correct = [p == t for p, t in zip(mp_, lab)]
    b_only = sum(bc and not mc for bc, mc in zip(b_correct, m_correct))  # base ถูก multi ผิด
    m_only = sum(mc and not bc for bc, mc in zip(b_correct, m_correct))  # multi ถูก base ผิด

    print(f"\n== {name}  vs baseline (paired, n={len(common)}) ==")
    print(f"  F1: baseline {f1b:.3f} -> {name} {f1m:.3f}  (ต่าง {f1m-f1b:+.3f})")
    print(f"  95% CI ของผลต่าง F1: [{lo:+.3f}, {hi:+.3f}]")
    if lo > 0:
        print(f"  -> CI ไม่คร่อม 0 = {name} ดีกว่า baseline อย่างมีนัยสำคัญ ✓")
    else:
        print(f"  -> CI คร่อม 0 (โอกาส {name} ไม่ดีกว่า/แย่กว่า = {p_worse:.0%}) = ยังสรุปว่าชนะไม่ได้")
    print(f"  McNemar: multi ถูก-base ผิด {m_only} ข้อ | base ถูก-multi ผิด {b_only} ข้อ")
