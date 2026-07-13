# -*- coding: utf-8 -*-
"""เทรน WangchanBERTa ตัวสุดท้ายบน gold ทั้ง 127 ข้อ แล้วเซฟไว้ให้เว็บเรียกใช้

ต่างจาก wangchanberta.py:
  wangchanberta.py = 5-fold CV เพื่อ "วัดผล" (เทรน 15 โมเดลแล้วทิ้ง -- ไม่มีโมเดลเหลือ)
  ไฟล์นี้        = เทรนตัวเดียวบนข้อมูลทั้งหมดเพื่อ "เอาไปใช้จริง"

*** เตือน ***
โมเดลตัวนี้เห็น gold ครบทั้ง 127 ข้อตอนเทรน
=> ห้ามเอาไปวัดผลบน gold เด็ดขาด มันจะได้คะแนนสูงปลอมๆ (เพราะจำคำตอบได้)
   ตัวเลขที่ใช้รายงานต้องมาจาก wangchanberta.py (out-of-fold) เท่านั้น
   ตัวนี้มีไว้ให้เว็บลองข้อความ "ใหม่" ที่ไม่อยู่ใน gold เท่านั้น

รัน: C:/Users/thana/pt/Scripts/python.exe train_final_wcb.py
"""
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as hf_log

from wangchanberta import BATCH, DS, EPOCHS, GOLD_CSV, LR, MODEL_NAME, make_collate

sys.stdout.reconfigure(encoding="utf-8")
hf_log.set_verbosity_error()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "wcb_model")
SEED = 42


def main():
    df = pd.read_csv(GOLD_CSV, dtype=str).fillna("")
    df["label"] = df["label"].str.strip()
    df = df[df["label"].isin(["0", "1"])].reset_index(drop=True)
    y = df["label"].astype(int).tolist()
    texts = df["text"].tolist()

    n_pos, n_neg = sum(y), len(y) - sum(y)
    pos_weight = n_neg / n_pos
    print(f"เทรนบน gold ทั้งหมด {len(df)} ข้อ (ประชด {n_pos}) | pos_weight {pos_weight:.2f}")

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.train()

    dl = DataLoader(DS(texts, y), batch_size=BATCH, shuffle=True, collate_fn=make_collate(tok))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], dtype=torch.float))

    t0 = time.time()
    for ep in range(EPOCHS):
        tot = 0.0
        for b in dl:
            opt.zero_grad()
            yy = b.pop("labels")
            loss = lossf(model(**b).logits, yy)
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"  epoch {ep+1}/{EPOCHS}  loss {tot/len(dl):.4f}  ({time.time()-t0:.0f}s)")

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"\nเซฟแล้ว -> {OUT_DIR}  ({time.time()-t0:.0f}s)")
    print("เตือน: โมเดลนี้เห็น gold ครบแล้ว -- ห้ามใช้วัดผลบน gold (ใช้เลขจาก wangchanberta.py)")


if __name__ == "__main__":
    main()
