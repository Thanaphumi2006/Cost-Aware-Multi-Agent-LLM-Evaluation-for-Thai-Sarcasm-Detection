# -*- coding: utf-8 -*-
"""Distillation ขั้น 2 — เทรน WangchanBERTa (student) ด้วย gold + silver แล้ววัดผลแบบ "ไม่โกง"

คำถามที่ตอบ: "multi-agent/GPT teacher สอน student ตัวเล็กให้เก่งขึ้นได้ไหม" (finding ที่สองของเปเปอร์)

*** โปรโตคอลวัดผลต้องไม่ leak (สำคัญสุด) ***
เราวัดผลบน gold เท่านั้น (ป้ายคนจริง) แต่ silver เอาไปช่วย "เทรน" ได้
  -> ใช้ 5-fold OOF เดียวกับ wangchanberta.py: fold ที่ te ใช้วัด, ส่วน silver + gold-train-folds เอาไปเทรน
  -> ทุกข้อ gold ถูกทำนายโดยโมเดลที่ไม่เคยเห็นข้อนั้น -> เทียบกับ baseline WCB (F1~0.62) ได้ตรงๆ
silver ใส่เข้า "ทุก training fold" (มันคือข้อมูลเสริม ไม่ใช่ชุดวัดผล)

reuse train_one_fold จาก wangchanberta.py -> พฤติกรรมเทรนเหมือน baseline เป๊ะ ต่างแค่ "ข้อมูลเทรน"

รัน (ใช้เวลา -- CPU, 5 fold × 3 seed):
  python distill_train_eval.py --silver silver.csv
  python distill_train_eval.py --silver silver.csv --out wcb_distill_oof.csv
"""
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# reuse ชิ้นส่วนเทรน/ค่าคอนฟิกชุดเดียวกับ baseline WCB (ห้าม divergent จะเทียบไม่ได้)
from wangchanberta import DS, make_collate, train_one_fold, MODEL_NAME, SEEDS, GOLD_CSV  # noqa: F401
try:
    from wangchanberta import N_FOLDS
except ImportError:
    N_FOLDS = 5


def prf(yt, yp):
    TP = sum(t == 1 and p == 1 for t, p in zip(yt, yp))
    FP = sum(t == 0 and p == 1 for t, p in zip(yt, yp))
    FN = sum(t == 1 and p == 0 for t, p in zip(yt, yp))
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F1, TP, FP, FN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", required=True, help="silver.csv จาก distill_label.py (text, silver_label)")
    ap.add_argument("--out", default="wcb_distill_oof.csv", help="เขียน OOF prediction (เทียบ/bootstrap ได้)")
    a = ap.parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from transformers import AutoTokenizer

    # gold = ชุดวัดผล (ป้ายคน)
    g = pd.read_csv(GOLD_CSV, dtype=str).fillna("")
    g["label"] = g["label"].str.strip()
    g = g[g["label"].isin(["0", "1"])].reset_index(drop=True)
    gy = np.array(g["label"].astype(int)); gt = g["text"].tolist()

    # silver = ข้อมูลเสริม (ป้าย teacher)
    s = pd.read_csv(a.silver, dtype=str).fillna("")
    s["silver_label"] = s["silver_label"].astype(str).str.strip()
    s = s[s["silver_label"].isin(["0", "1"])].reset_index(drop=True)
    st = s["text"].astype(str).tolist(); sy = s["silver_label"].astype(int).tolist()
    print(f"gold {len(gt)} (ประชด {int(gy.sum())}) + silver {len(st)} (ประชด {sum(sy)}) "
          f"| วัดผลบน gold แบบ {N_FOLDS}-fold OOF × {len(SEEDS)} seeds (CPU -- ใช้เวลา)\n")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    f1s, oof_all = [], {}
    for seed in SEEDS:
        oof = np.zeros(len(gt), dtype=int)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(gt, gy), 1):
            tr_texts = [gt[i] for i in tr] + st            # gold-train + silver ทั้งหมด
            tr_y = [int(gy[i]) for i in tr] + list(sy)
            n_pos = sum(tr_y); n_neg = len(tr_y) - n_pos
            pos_weight = n_neg / max(n_pos, 1)             # imbalance ของชุดเทรนรวม silver
            ts = time.time()
            oof[te] = train_one_fold(tr_texts, tr_y, [gt[i] for i in te], tok, pos_weight, seed)
            print(f"  seed {seed} fold {k}/{N_FOLDS}  ({time.time() - ts:.0f}s)")
        P, R, F1, *_ = prf(list(gy), list(oof))
        f1s.append(F1); oof_all[seed] = oof.copy()
        print(f"  -> seed {seed}: F1 {F1:.3f} | prec {P:.3f} | recall {R:.3f}\n")

    # seed ตัวแทน = median (ไม่ cherry-pick อันดีสุด) -- โปรโตคอลเดียวกับ wangchanberta.py
    order = sorted(range(len(f1s)), key=lambda i: f1s[i])
    rep = SEEDS[order[len(f1s) // 2]]
    g["distill_pred"] = oof_all[rep]
    for sd in SEEDS:
        g[f"distill_pred_seed{sd}"] = oof_all[sd]
    g.to_csv(a.out, index=False, encoding="utf-8-sig")

    P, R, F1, TP, FP, FN = prf(list(gy), list(oof_all[rep]))
    print("=" * 48)
    print(f"F1 ต่อ seed : {', '.join(f'{f:.3f}' for f in f1s)}  (seed ตัวแทน = median {rep})")
    print(f"distilled WCB (median): F1 {F1:.3f} | prec {P:.3f} | recall {R:.3f}  (TP{TP} FP{FP} FN{FN})")
    print(f"baseline WCB (ไม่มี silver): F1 ~0.62   |  teacher pipeline: F1 ~0.74")
    print(f"เขียน OOF -> {a.out}")
    print("อ่านผล: ปิดช่องว่างได้เท่าไร? ถ้า F1 ขยับเข้าใกล้ 0.74 = teacher สอน student ได้จริง")
    print("(ถ้า precision ตก = silver positive ปนเปื้อนตามที่เตือน -> เพิ่ม --pos-conf ตอน distill_label)")


if __name__ == "__main__":
    main()
