# -*- coding: utf-8 -*-
"""Distillation step 2 -- train WangchanBERTa (student) on gold + silver and measure it "without cheating"

Question answered: "can a multi-agent/GPT teacher make a small student better" (the paper second finding)

*** the evaluation protocol must not leak (most important) ***
We evaluate only on gold (real human labels), but silver may be used to "train"
  -> use the same 5-fold OOF as wangchanberta.py: the test fold is for measuring, silver + gold-train-folds are for training
  -> every gold item is predicted by a model that never saw it -> directly comparable to baseline WCB (F1~0.62)
silver goes into "every training fold" (it is auxiliary data, not an evaluation set)

reuse train_one_fold from wangchanberta.py -> training behavior identical to baseline, only the "training data" differs

Run (takes time -- CPU, 5 folds × 3 seeds):
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

# reuse the same training pieces/config as baseline WCB (must not diverge or it is not comparable)
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
    ap.add_argument("--silver", required=True, help="silver.csv from distill_label.py (text, silver_label)")
    ap.add_argument("--out", default="wcb_distill_oof.csv", help="write OOF predictions (for comparison/bootstrap)")
    a = ap.parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from transformers import AutoTokenizer

    # gold = evaluation set (human labels)
    g = pd.read_csv(GOLD_CSV, dtype=str).fillna("")
    g["label"] = g["label"].str.strip()
    g = g[g["label"].isin(["0", "1"])].reset_index(drop=True)
    gy = np.array(g["label"].astype(int)); gt = g["text"].tolist()

    # silver = auxiliary data (teacher labels)
    s = pd.read_csv(a.silver, dtype=str).fillna("")
    s["silver_label"] = s["silver_label"].astype(str).str.strip()
    s = s[s["silver_label"].isin(["0", "1"])].reset_index(drop=True)
    st = s["text"].astype(str).tolist(); sy = s["silver_label"].astype(int).tolist()
    print(f"gold {len(gt)} (sarcasm {int(gy.sum())}) + silver {len(st)} (sarcasm {sum(sy)}) "
          f"| eval on gold with {N_FOLDS}-fold OOF × {len(SEEDS)} seeds (CPU -- takes time)\n")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    f1s, oof_all = [], {}
    for seed in SEEDS:
        oof = np.zeros(len(gt), dtype=int)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(gt, gy), 1):
            tr_texts = [gt[i] for i in tr] + st            # gold-train + all silver
            tr_y = [int(gy[i]) for i in tr] + list(sy)
            n_pos = sum(tr_y); n_neg = len(tr_y) - n_pos
            pos_weight = n_neg / max(n_pos, 1)             # imbalance of the combined training set incl. silver
            ts = time.time()
            oof[te] = train_one_fold(tr_texts, tr_y, [gt[i] for i in te], tok, pos_weight, seed)
            print(f"  seed {seed} fold {k}/{N_FOLDS}  ({time.time() - ts:.0f}s)")
        P, R, F1, *_ = prf(list(gy), list(oof))
        f1s.append(F1); oof_all[seed] = oof.copy()
        print(f"  -> seed {seed}: F1 {F1:.3f} | prec {P:.3f} | recall {R:.3f}\n")

    # representative seed = median (no cherry-picking the best) -- same protocol as wangchanberta.py
    order = sorted(range(len(f1s)), key=lambda i: f1s[i])
    rep = SEEDS[order[len(f1s) // 2]]
    g["distill_pred"] = oof_all[rep]
    for sd in SEEDS:
        g[f"distill_pred_seed{sd}"] = oof_all[sd]
    g.to_csv(a.out, index=False, encoding="utf-8-sig")

    P, R, F1, TP, FP, FN = prf(list(gy), list(oof_all[rep]))
    print("=" * 48)
    print(f"F1 per seed : {', '.join(f'{f:.3f}' for f in f1s)}  (representative seed = median {rep})")
    print(f"distilled WCB (median): F1 {F1:.3f} | prec {P:.3f} | recall {R:.3f}  (TP{TP} FP{FP} FN{FN})")
    print(f"baseline WCB (no silver): F1 ~0.62   |  teacher pipeline: F1 ~0.74")
    print(f"wrote OOF -> {a.out}")
    print("reading the result: how much of the gap closed? if F1 moves toward 0.74 = the teacher really taught the student")
    print("(if precision drops = silver positives are contaminated as warned -> raise --pos-conf in distill_label)")


if __name__ == "__main__":
    main()
