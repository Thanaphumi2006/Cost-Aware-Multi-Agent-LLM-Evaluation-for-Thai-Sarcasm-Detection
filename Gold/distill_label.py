# -*- coding: utf-8 -*-
"""Distillation step 1 -- have the "teacher" put silver labels on unlabeled text

Idea: use the GPT system (teacher) to label lots of raw text, getting "silver data" (labels that may be partly wrong)
then train WangchanBERTa (student) on it -> hoping to close the F1 gap 0.62 -> 0.74 at per-item cost = 0

*** the key that keeps silver from breaking despite an imprecise teacher (teacher precision ~0.68) ***
the teacher has "confidence" (P(sarcasm) from logprob) -> keep only high-confidence items on both sides, drop the middle
  sarcasm     : prob >= --pos-conf (e.g. 0.90)
  not sarcasm : prob <= --neg-conf (e.g. 0.05)
-> reduces label noise (the borderline range is where the teacher errs most)

Why the teacher is a "single agent" (predict.py/batch_eval.py), not pipeline v2:
  - the pipeline gives only hard labels with no confidence -> cannot filter noise
  - the single agent has logprob (can select by confidence) + higher precision (0.68 vs 0.60) = a better teacher for distillation

Steps:
  1) have the teacher (cheapest) label the raw text first -- batch API recommended, halves the price:
       python batch_eval.py --csv harvest_to_review.csv --out harvest_pred.csv
  2) filter into silver:
       python distill_label.py --pred harvest_pred.csv --out silver.csv --pos-conf 0.9 --neg-conf 0.05 --balance
"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="CSV from batch_eval.py/predict.py (must have text + pred_prob)")
    ap.add_argument("--out", default="silver.csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--pos-conf", type=float, default=0.90, help="keep as sarcasm when prob >= this")
    ap.add_argument("--neg-conf", type=float, default=0.05, help="keep as not-sarcasm when prob <= this")
    ap.add_argument("--max-pos", type=int, default=0, help="cap the number of silver sarcasm (0=no cap)")
    ap.add_argument("--max-neg", type=int, default=0, help="cap the number of silver not-sarcasm (0=no cap)")
    ap.add_argument("--balance", action="store_true", help="trim negatives to match positives (prevent silver skewing negative)")
    a = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(a.pred, dtype=str).fillna("")
    if a.text_col not in df.columns or "pred_prob" not in df.columns:
        sys.exit(f"columns '{a.text_col}' and 'pred_prob' required (run batch_eval.py/predict.py --csv first)")
    df = df[df["pred_prob"] != ""].copy()
    df["p"] = df["pred_prob"].astype(float)

    pos = df[df["p"] >= a.pos_conf].copy(); pos["silver_label"] = 1
    neg = df[df["p"] <= a.neg_conf].copy(); neg["silver_label"] = 0
    pos = pos.sort_values("p", ascending=False)   # most confident first (in case of cap)
    neg = neg.sort_values("p", ascending=True)
    if a.max_pos:
        pos = pos.head(a.max_pos)
    if a.max_neg:
        neg = neg.head(a.max_neg)
    if a.balance:
        k = min(len(pos), len(neg))
        pos, neg = pos.head(k), neg.head(k)

    out = pd.concat([pos, neg], ignore_index=True).rename(columns={a.text_col: "text"})
    out = out[["text", "silver_label", "p"]].rename(columns={"p": "teacher_prob"})
    out["teacher_prob"] = out["teacher_prob"].round(3)
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"] != ""].drop_duplicates(subset="text").reset_index(drop=True)
    out.to_csv(a.out, index=False, encoding="utf-8-sig")

    npos = int((out["silver_label"] == 1).sum()); nneg = int((out["silver_label"] == 0).sum())
    print(f"wrote {a.out} · silver {len(out)} items (sarcasm {npos} / not {nneg}) · "
          f"dropped the uncertain middle {len(df) - len(out)} items (pos>={a.pos_conf}, neg<={a.neg_conf})")
    print("warning: 'silver' = the teacher may be wrong -- teacher precision is limited, sarcasm (positives) may be contaminated "
          "train on it then measure only with distill_train_eval.py (OOF, no leak)")


if __name__ == "__main__":
    main()
