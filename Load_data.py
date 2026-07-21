# -*- coding: utf-8 -*-
"""
Step 1: load 2 raw datasets -> keep only the text -> drop original labels -> merge into one file

Key points:
- both are "loading-script" datasets, which datasets>=4.0 no longer supports
  so pin datasets<4.0 and pass trust_remote_code=True
- text columns: wisesight = 'texts' , wongnai = 'review_body'

Install (run once):
    pip install "datasets<4.0" pandas
"""

from datasets import load_dataset
import pandas as pd


def get_text_column(ds_split, candidates):
    """flexibly pick the text column name, in case the column name changes"""
    for c in candidates:
        if c in ds_split.column_names:
            return c
    raise KeyError(f"text column not found, check: {ds_split.column_names}")


# ── 1) load wisesight (Thai social text) ──────────────────────────────
ws = load_dataset("pythainlp/wisesight_sentiment", trust_remote_code=True)
ws_texts = []
for split in ws:                                   # combine all splits (train/validation/test)
    col = get_text_column(ws[split], ["texts", "text"])
    ws_texts += list(ws[split][col])               # keep only text, do not touch 'category'

# ── 2) load wongnai (restaurant reviews) ───────────────────────────────────
wg = load_dataset("Wongnai/wongnai_reviews", trust_remote_code=True)
wg_texts = []
for split in wg:
    col = get_text_column(wg[split], ["review_body", "text"])
    wg_texts += list(wg[split][col])               # keep only text, do not touch 'star_rating'

# ── 3) combine into one DataFrame, keep only text + source (drop all original labels) ──
df = pd.DataFrame(
    [{"text": t, "source": "wisesight"} for t in ws_texts]
    + [{"text": t, "source": "wongnai"} for t in wg_texts]
)

# ── 4) basic cleaning: trim whitespace, drop empties, drop duplicates ──────────────────
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0]
df = df.drop_duplicates(subset="text").reset_index(drop=True)

# ── 5) inspect + save for later use ─────────────────────────────────────────────
print("total texts:", len(df))
print(df["source"].value_counts())
print("\nsample of 5 texts:")
print(df.sample(5, random_state=0).to_string(index=False))

df.to_csv("raw_texts.csv", index=False, encoding="utf-8-sig")   # utf-8-sig so Thai is not garbled when opened in Excel
print("\nsaved to: raw_texts.csv")

# note: if you still get script errors (e.g. the machine forces datasets 4.x)
# fallback: load from the parquet files HF auto-converts, e.g.
#   df_ws = pd.read_parquet("hf://datasets/pythainlp/wisesight_sentiment/data/train-00000-of-00001.parquet")
# then pull the 'texts' column instead (install: pip install huggingface_hub pyarrow)