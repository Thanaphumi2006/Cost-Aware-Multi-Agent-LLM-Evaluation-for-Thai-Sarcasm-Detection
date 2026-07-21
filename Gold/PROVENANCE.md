# Provenance of gold.csv (must be cited in the report)

Summary: **the sarcastic side (label=1) does not come from random sampling** but from LLM-assisted two-stage mining.
A human decides every final label, but *which items the human saw* were chosen by an LLM.

## Current state

| | count |
|---|---|
| gold.csv | 127 items |
| sarcastic (1) | 30 |
| not sarcastic (0) | 97 |
| source | wongnai 85 / wisesight 42 |

## The data's path

1. **First round (random/keyword)** `Keyword Filter for NLP Project.py` scores suspicion
   → `to_label.csv` 400 items (high-suspicion 60% / normal 40%) → `human_review.py` (blind mode)
   → first gold version, 102 items (14 sarcastic / 88 not).
   The sarcastic side this round is **not biased toward an LLM** but biased toward keywords.

2. **Mining round (LLM-assisted)** 14 sarcastic items aren't enough to measure F1.
   - `Harvest.py` uses **GPT-4o** to scan 800 items of `scored_texts.csv` (wisesight only, ≤150 chars)
     narrowing to 495 that GPT-4o thinks "might be sarcastic" → `harvest_to_review.csv`
   - **Claude** reads the top 130 of that pool and filters per `labeling_rubric.md` down to 37,
     written to `shortlist.csv`, used to order `Quick_Review.py` to show these first.
   - A human reviews (blind to `llm_conf`) the top 25 → 16 sarcastic / 9 not (hit rate 64%).
   - Merged into gold → 127 items / 30 sarcastic.

## Limitations to state in the report

- The 16 sarcastic items added are **sarcasm that GPT-4o found suspicious and Claude judged likely.**
  Sarcasm that both LLMs overlooked never has a chance to enter gold.
- Consequence: an LLM baseline (especially GPT-4o) gets an **inflated recall on the sarcastic side**,
  because it is measured on examples it helped select.
- Mitigation (still possible): `harvest_to_review.csv` has 470 unreviewed items left.
  Reviewing the group with **no cues** (no `555` / no elongation, at the tail of `Quick_Review.py`'s queue)
  and keeping the sarcasm found there would add sarcasm the LLM can't detect, balancing the gold set.

## A note on the accuracy of this record

`shortlist.csv` has been deleted and cannot be recreated exactly
(the row order in `harvest_to_review.csv` was rewritten, so the index used for filtering is always off at the score point).
So it is **not possible to state per-item** which sarcastic item came from the shortlist vs. from pure `llm_conf` ordering.
What can be confirmed: all 25 human-reviewed items have `llm_conf` between 0.80-0.95
and were all surfaced by the LLM; none was random.
