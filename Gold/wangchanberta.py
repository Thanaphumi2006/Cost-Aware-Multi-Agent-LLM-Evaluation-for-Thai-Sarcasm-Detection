# -*- coding: utf-8 -*-
"""ระบบ ③ — WangchanBERTa (โมเดลเล็กเทรนเอง) เทียบกับ LLM

คำถามที่ระบบนี้ตอบ: "ต้องใช้ LLM จริงไหม หรือโมเดลเล็กฟรีๆ ก็พอ"

ทำไมต้อง 5-fold CV ไม่ใช่ train/test split:
  gold มีแค่ 127 ข้อ (ประชด 30) -- แบ่ง 80/20 จะเหลือ test 25 ข้อ ประชด 6 ข้อ
  -> ตัวเลขจะเป็น noise ล้วน และเทียบกับ baseline/multi-agent (ที่วัดบน 127 ข้อ) ไม่ได้
  5-fold CV: ทุกข้อถูกทำนายโดยโมเดลที่ "ไม่เคยเห็นข้อนั้น" -> ได้ pred ครบ 127 ข้อ
  -> เทียบข้อต่อข้อกับอีกสองระบบได้ตรงๆ เข้า paired bootstrap ตัวเดิมได้เลย

จัดการ class imbalance (97:30) ด้วย pos_weight ใน loss
ไม่งั้นโมเดลจะเรียนรู้ทางลัด "ตอบ 0 ทุกข้อ" ซึ่งได้ accuracy 0.764 ฟรีๆ

รัน: python wangchanberta.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as hf_log

from baseline import metrics  # harness เดียวกับอีกสองระบบ -- พิสูจน์ว่าไม่ได้วัดคนละมาตรฐาน

sys.stdout.reconfigure(encoding="utf-8")
hf_log.set_verbosity_error()

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_CSV = os.path.join(HERE, "gold.csv")
PRED_CSV = os.path.join(HERE, "wangchanberta_preds.csv")

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
N_FOLDS = 5
SEEDS = [42, 7, 2024]   # เทรนหลาย seed -- ข้อมูลน้อยขนาดนี้ผลแกว่งมาก ต้องรายงานความแกว่งด้วย
EPOCHS = 4
BATCH = 8
LR = 2e-5
MAX_LEN = 256


class DS(Dataset):
    """เก็บ text ดิบไว้ -- ให้ collate ตัดสินความยาว pad ทีละ batch
    (ข้อความจริง p50 แค่ 87 tokens ถ้า pad ทุกข้อไป 256 = เผาเวลา CPU กับ padding เปล่าๆ ~3 เท่า)"""

    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.y = list(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.texts[i], self.y[i]


def make_collate(tok):
    def collate(batch):
        texts, ys = zip(*batch)
        enc = tok(list(texts), truncation=True, padding=True,
                  max_length=MAX_LEN, return_tensors="pt")
        enc["labels"] = torch.tensor(ys, dtype=torch.long)
        return enc
    return collate


def train_one_fold(tr_texts, tr_y, te_texts, tok, pos_weight, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.train()

    dl = DataLoader(DS(tr_texts, tr_y), batch_size=BATCH, shuffle=True,
                    collate_fn=make_collate(tok))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    # ถ่วงน้ำหนักคลาสประชด -- กันโมเดลเลือกทางลัด "ตอบ 0 ทุกข้อ" (ซึ่งได้ acc 0.764 ฟรีๆ)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], dtype=torch.float))

    for _ in range(EPOCHS):
        for b in dl:
            opt.zero_grad()
            y = b.pop("labels")
            loss = lossf(model(**b).logits, y)
            loss.backward()
            opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te_texts), BATCH):
            enc = tok(list(te_texts[i:i + BATCH]), truncation=True, padding=True,
                      max_length=MAX_LEN, return_tensors="pt")
            preds += model(**enc).logits.argmax(-1).tolist()
    return preds


def main():
    df = pd.read_csv(GOLD_CSV, dtype=str).fillna("")
    df["label"] = df["label"].str.strip()
    df = df[df["label"].isin(["0", "1"])].reset_index(drop=True)
    y = df["label"].astype(int).values
    texts = df["text"].values

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    pos_weight = n_neg / n_pos
    print(f"gold {len(df)} ข้อ | ประชด {n_pos} / ไม่ประชด {n_neg} | pos_weight {pos_weight:.2f}")
    print(f"โมเดล {MODEL_NAME}")
    print(f"{N_FOLDS}-fold CV × {len(SEEDS)} seeds × {EPOCHS} epochs (CPU -- ใช้เวลาสักพัก)\n")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    t0 = time.time()
    per_seed = []
    oof_all = {}

    for seed in SEEDS:
        oof = np.zeros(len(df), dtype=int)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(texts, y), 1):
            ts = time.time()
            oof[te] = train_one_fold(texts[tr], y[tr], texts[te], tok, pos_weight, seed)
            print(f"  seed {seed} fold {k}/{N_FOLDS}  ({time.time()-ts:.0f}s)")

        pred = [str(p) for p in oof]
        acc, prec, rec, f1, (tn, fp, fn, tp) = metrics(df["label"].tolist(), pred)
        per_seed.append((seed, f1, prec, rec, tp, fp, fn))
        oof_all[seed] = pred
        print(f"  -> seed {seed}: F1 {f1:.3f} | prec {prec:.3f} | recall {rec:.3f} "
              f"| TP {tp} FP {fp} FN {fn}\n")

    total = time.time() - t0

    # เลือก seed ที่ให้ F1 กลาง (median) เป็นตัวแทน -- ไม่ใช่ seed ที่ดีที่สุด (นั่นคือการ cherry-pick)
    f1s = [r[1] for r in per_seed]
    med_i = int(np.argsort(f1s)[len(f1s) // 2])
    rep_seed = per_seed[med_i][0]

    out = df[["text", "label", "source"]].copy() if "source" in df.columns else df[["text", "label"]].copy()
    out["pred"] = oof_all[rep_seed]
    for s in SEEDS:
        out[f"pred_seed{s}"] = oof_all[s]
    out.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")

    print("=" * 62)
    print(f"F1 ต่อ seed : {', '.join(f'{f:.3f}' for f in f1s)}")
    print(f"F1 เฉลี่ย    : {np.mean(f1s):.3f}  (SD {np.std(f1s):.3f})")
    print(f"seed ตัวแทน  : {rep_seed} (median -- ไม่เลือก seed ที่ดีที่สุด นั่นคือ cherry-pick)")
    print(f"เวลาเทรนรวม  : {total/60:.1f} นาที (CPU)")
    print(f"ค่า API      : $0.00  |  LLM calls: 0")
    print(f"บันทึก -> {PRED_CSV}")
    print("=" * 62)
    print("\nเทียบกับ LLM (วัด gold ชุดเดียวกัน 127 ข้อ):")
    print(f"  baseline (เดี่ยว)   F1 0.690 | $0.094")
    print(f"  multi-agent v2      F1 0.744 | $0.169")
    print(f"  WangchanBERTa       F1 {np.mean(f1s):.3f} | $0.00")
    print("\nรัน compare_systems.py เพื่อทดสอบนัยสำคัญแบบ paired")


if __name__ == "__main__":
    main()
