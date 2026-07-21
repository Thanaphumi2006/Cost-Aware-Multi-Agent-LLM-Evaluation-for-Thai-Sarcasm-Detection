# -*- coding: utf-8 -*-
"""System ③ — WangchanBERTa (a self-trained small model) vs. the LLM

The question this system answers: "do you really need the LLM, or is a free small model enough?"

Why 5-fold CV, not a train/test split:
  gold has only 127 items (30 sarcastic) -- an 80/20 split leaves a test set of 25, with 6 sarcastic
  -> the numbers would be pure noise, and not comparable to baseline/multi-agent (measured on 127)
  5-fold CV: every item is predicted by a model that "never saw that item" -> full 127-item predictions
  -> compares item-by-item with the other two systems directly, straight into the same paired bootstrap

Handle the class imbalance (97:30) with pos_weight in the loss,
or the model learns the shortcut "answer 0 for everything," which gets accuracy 0.764 for free.

Run: python wangchanberta.py
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

from baseline import metrics  # same harness as the other two systems -- proving no double standard

sys.stdout.reconfigure(encoding="utf-8")
hf_log.set_verbosity_error()

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_CSV = os.path.join(HERE, "gold.csv")
PRED_CSV = os.path.join(HERE, "wangchanberta_preds.csv")

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
N_FOLDS = 5
SEEDS = [42, 7, 2024]   # multiple seeds -- with this little data results swing a lot, so report the variance
EPOCHS = 4
BATCH = 8
LR = 2e-5
MAX_LEN = 256


class DS(Dataset):
    """Keep raw text -- let collate decide the length and pad per batch
    (real text is p50 only 87 tokens; padding everything to 256 wastes ~3x the CPU time on empty padding)"""

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
    # weight the sarcastic class -- prevent the shortcut "answer 0 for everything" (which gets acc 0.764 for free)
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
    print(f"gold {len(df)} items | sarcastic {n_pos} / not {n_neg} | pos_weight {pos_weight:.2f}")
    print(f"model {MODEL_NAME}")
    print(f"{N_FOLDS}-fold CV × {len(SEEDS)} seeds × {EPOCHS} epochs (CPU -- takes a while)\n")

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

    # pick the median-F1 seed as representative -- not the best seed (that would be cherry-picking)
    f1s = [r[1] for r in per_seed]
    med_i = int(np.argsort(f1s)[len(f1s) // 2])
    rep_seed = per_seed[med_i][0]

    out = df[["text", "label", "source"]].copy() if "source" in df.columns else df[["text", "label"]].copy()
    out["pred"] = oof_all[rep_seed]
    for s in SEEDS:
        out[f"pred_seed{s}"] = oof_all[s]
    out.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")

    print("=" * 62)
    print(f"F1 per seed : {', '.join(f'{f:.3f}' for f in f1s)}")
    print(f"F1 mean     : {np.mean(f1s):.3f}  (SD {np.std(f1s):.3f})")
    print(f"repr. seed  : {rep_seed} (median -- not the best seed, that would be cherry-picking)")
    print(f"total train : {total/60:.1f} min (CPU)")
    print(f"API cost    : $0.00  |  LLM calls: 0")
    print(f"saved -> {PRED_CSV}")
    print("=" * 62)
    print("\nvs the LLM (measured on the same 127-item gold):")
    print(f"  baseline (single)   F1 0.690 | $0.094")
    print(f"  multi-agent v2      F1 0.744 | $0.169")
    print(f"  WangchanBERTa       F1 {np.mean(f1s):.3f} | $0.00")
    print("\nrun compare_systems.py for the paired significance test")


if __name__ == "__main__":
    main()
