# -*- coding: utf-8 -*-
"""Produce out-of-fold "probabilities" for WangchanBERTa (not just 0/1)

Why this file exists:
  wangchanberta.py stores only argmax -> 0/1 labels at a fixed 0.5 threshold
  but cascade needs "scores" to slide the threshold to a recall high enough to be a screener
  (a screener needn't be accurate -- it must not let sarcasm slip; the verifier trims the excess later)

Same protocol as wangchanberta.py exactly (5-fold × 3 seeds × 4 epochs, same pos_weight)
differing only in storing the "sarcastic" class prob instead of argmax + recording each item's fold
  -> the fold is used to choose the threshold leave-fold-out in cascade.py (avoids choosing it from the item being predicted)

Every item is scored by a model that never saw it -- compares item-by-item with the other systems directly.

Run: python wcb_oof_probs.py       (free, no API -- trains 15 models, takes a while)
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
    """CPU by default -- tried MPS and it's clearly "slower" than CPU at this model/batch size (batch 8, short text)
    the task is too small to be worth the overhead of shipping data to the GPU
    plus CPU matches the 0.620 that wangchanberta.py reported -> directly comparable
    to try MPS: WCB_DEVICE=mps python wcb_oof_probs.py"""
    want = os.environ.get("WCB_DEVICE", "cpu")
    if want == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_fold_probs(tr_texts, tr_y, te_texts, tok, pos_weight, seed, dev):
    """train on the train fold and return the test fold's P(sarcastic) -- copy of train_one_fold modified to return prob"""
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
            p = torch.softmax(model(**enc).logits, -1)[:, 1]     # P(sarcastic)
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
    print(f"gold {len(df)} items | sarcastic {n_pos} / not {n_neg} | pos_weight {pos_weight:.2f}")
    print(f"{N_FOLDS}-fold × {len(SEEDS)} seeds = {N_FOLDS*len(SEEDS)} models | device {dev}\n")

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

        # sanity check: argmax(prob>0.5) should give F1 near the reported 0.620, else something is wrong
        pred = [("1" if p >= 0.5 else "0") for p in probs]
        _, prec, rec, f1, (tn, fp, fn, tp) = metrics(df["label"].tolist(), pred)
        print(f"  -> seed {seed} @0.5: F1 {f1:.3f} | prec {prec:.3f} | recall {rec:.3f} "
              f"| TP {tp} FP {fp} FN {fn}\n", flush=True)

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("=" * 62)
    print(f"total time {(time.time()-t0)/60:.1f} min | API cost $0.00")
    print(f"saved -> {OUT_CSV}")
    print("next: python cascade.py --dry-run   (see which threshold is worth it before paying)")
    print("=" * 62)


if __name__ == "__main__":
    main()
