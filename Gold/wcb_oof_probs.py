# -*- coding: utf-8 -*-
"""ผลิต "ความน่าจะเป็น" out-of-fold ของ WangchanBERTa (ไม่ใช่แค่ 0/1)

ทำไมต้องมีไฟล์นี้:
  wangchanberta.py เก็บแค่ argmax -> ได้ป้าย 0/1 ที่ threshold 0.5 ตายตัว
  แต่ cascade ต้องการ "คะแนน" เพื่อจะรูดหา threshold ที่ recall สูงพอจะเป็นด่านคัดกรองได้
  (ด่านคัดกรองไม่ต้องแม่น -- ต้องไม่ปล่อยประชดหลุด ส่วนที่เกินให้ verifier ตัดทิ้งทีหลัง)

โปรโตคอลเหมือน wangchanberta.py เป๊ะ (5-fold × 3 seeds × 4 epochs, pos_weight เดิม)
ต่างแค่เก็บ prob ของคลาส "ประชด" แทน argmax + เก็บว่าข้อไหนอยู่ fold ไหน
  -> fold ใช้ตอนเลือก threshold แบบ leave-fold-out ใน cascade.py (กันเลือก threshold จากข้อที่กำลังจะทำนาย)

ทุกข้อถูกให้คะแนนโดยโมเดลที่ไม่เคยเห็นข้อนั้น -- เอาไปเทียบข้อต่อข้อกับระบบอื่นได้ตรงๆ

รัน: python wcb_oof_probs.py       (ฟรี ไม่แตะ API -- เทรน 15 โมเดล ใช้เวลาสักพัก)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as hf_log

from baseline import metrics
from wangchanberta import BATCH, DS, EPOCHS, GOLD_CSV, LR, MAX_LEN, MODEL_NAME, N_FOLDS, SEEDS, make_collate

sys.stdout.reconfigure(encoding="utf-8")
hf_log.set_verbosity_error()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "wcb_oof_probs.csv")


def device():
    """CPU เป็นค่าเริ่มต้น -- ลอง MPS แล้ว "ช้ากว่า" CPU ชัดเจนกับโมเดล/แบตช์ขนาดนี้ (batch 8, ข้อความสั้น)
    งานเล็กเกินกว่าจะคุ้มค่า overhead ของการโยนข้อมูลไป GPU
    แถม CPU ยังตรงกับที่ wangchanberta.py รายงาน 0.620 ไว้ -> เทียบกันได้ตรงๆ
    อยากลอง MPS: WCB_DEVICE=mps python wcb_oof_probs.py"""
    want = os.environ.get("WCB_DEVICE", "cpu")
    if want == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_fold_probs(tr_texts, tr_y, te_texts, tok, pos_weight, seed, dev):
    """เทรนบน train fold แล้วคืน P(ประชด) ของ test fold -- ก๊อป train_one_fold มาแก้ให้คืน prob"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(dev)
    model.train()

    dl = DataLoader(DS(tr_texts, tr_y), batch_size=BATCH, shuffle=True, collate_fn=make_collate(tok))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], dtype=torch.float).to(dev))

    for _ in range(EPOCHS):
        for b in dl:
            opt.zero_grad()
            b = {k: v.to(dev) for k, v in b.items()}
            y = b.pop("labels")
            loss = lossf(model(**b).logits, y)
            loss.backward()
            opt.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(te_texts), BATCH):
            enc = tok(list(te_texts[i:i + BATCH]), truncation=True, padding=True,
                      max_length=MAX_LEN, return_tensors="pt").to(dev)
            p = torch.softmax(model(**enc).logits, -1)[:, 1]     # P(ประชด)
            probs += p.float().cpu().tolist()
    del model
    return probs


def main():
    df = pd.read_csv(GOLD_CSV, dtype=str).fillna("")
    df["label"] = df["label"].str.strip()
    df = df[df["label"].isin(["0", "1"])].reset_index(drop=True)
    y = df["label"].astype(int).values
    texts = df["text"].values

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    pos_weight = n_neg / n_pos
    dev = device()
    print(f"gold {len(df)} ข้อ | ประชด {n_pos} / ไม่ประชด {n_neg} | pos_weight {pos_weight:.2f}")
    print(f"{N_FOLDS}-fold × {len(SEEDS)} seeds = {N_FOLDS*len(SEEDS)} โมเดล | device {dev}\n")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    out = df[["text", "label"]].copy()
    t0 = time.time()

    for seed in SEEDS:
        probs = np.zeros(len(df), dtype=float)
        folds = np.zeros(len(df), dtype=int)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(texts, y), 1):
            ts = time.time()
            probs[te] = train_fold_probs(texts[tr], y[tr], texts[te], tok, pos_weight, seed, dev)
            folds[te] = k
            print(f"  seed {seed} fold {k}/{N_FOLDS}  ({time.time()-ts:.0f}s)", flush=True)

        out[f"prob_seed{seed}"] = probs
        out[f"fold_seed{seed}"] = folds

        # sanity check: argmax(prob>0.5) ต้องได้ F1 ใกล้ๆ 0.620 ที่รายงานไว้ ไม่งั้นแปลว่าเพี้ยน
        pred = [("1" if p >= 0.5 else "0") for p in probs]
        _, prec, rec, f1, (tn, fp, fn, tp) = metrics(df["label"].tolist(), pred)
        print(f"  -> seed {seed} @0.5: F1 {f1:.3f} | prec {prec:.3f} | recall {rec:.3f} "
              f"| TP {tp} FP {fp} FN {fn}\n", flush=True)

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("=" * 62)
    print(f"เวลารวม {(time.time()-t0)/60:.1f} นาที | ค่า API $0.00")
    print(f"บันทึก -> {OUT_CSV}")
    print("ต่อไป: python cascade.py --dry-run   (ดูว่า threshold ไหนคุ้ม ก่อนจ่ายเงินจริง)")
    print("=" * 62)


if __name__ == "__main__":
    main()
