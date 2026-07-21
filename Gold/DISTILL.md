# Distillation: can the teacher (GPT) teach the student (WangchanBERTa)?

**Status: finding-in-progress.** The pipeline is ready and the dry-check passes (wiring OK, no API used); not yet run for real on a real pool.

Research question (the paper's second finding): *"Can an expensive multi-agent/GPT system teach a free small model to be better?"*

## Motivation (the gap to close)
| System | F1 | precision | recall | cost/item |
|---|---|---|---|---|
| WangchanBERTa (baseline, no silver) | ~0.62 | 0.55 | 0.70 | **$0** |
| deployed single-agent (gpt-4.1-mini) | ~0.72 | 0.68 | 0.83 | ~$0.0001 |
| teacher pipeline (gpt-4o detector→verifier) | ~0.74 | 0.60 | 0.97 | ~$0.0006 |

If distillation closes even half of the 0.62→0.74 gap = quality near the pipeline at a per-item cost of 0.

## Pipeline (4 stages, every script is data-source-agnostic)
```
fetch_to_csv.py  <urls>          -> pool.csv        fetch Thai comments (any platform fetch_social supports)
batch_eval.py    --csv pool.csv  -> pool_pred.csv   teacher labels (Batch API halves the price)
distill_label.py --pred ...      -> silver.csv      keep only items the teacher is confident on (drop the middle)
distill_train_eval.py --silver   -> OOF F1 vs 0.62  train WCB + measure leak-free
```

## Honest design (why it's built this way)
1. **Confidence filtering** (`--pos-conf`/`--neg-conf`): the teacher's precision is limited (~0.68) → silver labels carry noise,
   especially in the borderline range. Keep only the two tails (high confidence), drop the middle → less noise entering the student's weights.
2. **teacher = single-agent, not pipeline v2**: the pipeline gives hard labels with no confidence (can't filter noise),
   while the single-agent has a logprob (can filter by confidence) + higher precision (0.68 vs 0.60) = a better teacher for distillation.
3. **Leak-free evaluation (most important)**: same 5-fold OOF as `wangchanberta.py`. Silver goes into every training fold,
   evaluate on the gold fold the model never saw → the F1 compares directly to the 0.62 baseline, not a fake score from memorization.
   (reuse `train_one_fold` → identical training behavior to the baseline, differing only in "training data")
4. **Mine from the target domain**: the real failure is cross-domain (precision 0.68→0.40 on Pantip).
   Silver from the domain you'll deploy to (web boards) = teaching the student that domain = exactly where it breaks.
   Silver from the original Wongnai/Wisesight = reinforcing what the model already knows; no help cross-domain.

## How to read the result
- F1 (OOF) moves toward 0.74 → **the teacher genuinely teaches the student** (a positive finding).
- F1 flat/down, precision down → silver positives are contaminated (teacher over-flags) → raise `--pos-conf` and retry.
- recall up but precision down → silver adds catches but brings noise → adjust balance/threshold.

## Known limitations/risks up front (honest)
- The teacher itself is capped at F1 ~0.74, precision ~0.60 → the student is unlikely to exceed the teacher.
- Sarcasm is rare → the raw pool is mostly negative; most of the teacher's positives may be false positives
  → the sarcastic side of silver is the most at risk of contamination (high pos-conf filtering helps but doesn't eliminate it).
- gold already has self-selection bias (see PROVENANCE.md) → distillation inherits that bias.

## Dry-check already passed (no API)
- `distill_label.py` on `gold_pred.csv` → silver 66 items (33/33), dropped the middle 61, columns correct.
- `distill_train_eval.py` data prep: fold0 = gold-train 101 + silver 66 = 167 train / 26 eval, labels/shape correct.
- (uses gold_pred.csv only to test the plumbing; a real run needs a fresh pool from fetch_to_csv.py, not labels on gold)

## Files
- `fetch_to_csv.py` — fetch comments from multiple sources → CSV (the data-expansion entry point)
- `batch_eval.py` — teacher labeling (Batch API, half price)
- `distill_label.py` — confidence filtering → silver
- `distill_train_eval.py` — train + OOF evaluation (comparable to the 0.62 baseline)
- generated artifacts (`pool*.csv`, `silver*.csv`, `wcb_distill_oof.csv`) are gitignored (regenerable + other people's data)
