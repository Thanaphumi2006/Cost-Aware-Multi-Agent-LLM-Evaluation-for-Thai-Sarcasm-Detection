# -*- coding: utf-8 -*-
"""
Step 3 (round 1): flag "sarcasm suspects" using keywords/patterns

Input : raw_texts.csv  (from step 1 -- has columns text, source)
Output:
  - scored_texts.csv : every text + a suspicion score (for inspection/tuning)
  - to_label.csv     : the pile to actually label (mix of suspect + normal, with a blank label column)

Principle: the more sarcasm signals, the higher the score
Strongest signal = both "praise" and "negative context" in one text
Run:  pip install pandas   then   python round1_keyword_filter.py
"""

import re
import pandas as pd

# ================== tunable here ==================
INPUT_CSV = "raw_texts.csv"
OUTPUT_ALL = "scored_texts.csv"
OUTPUT_TO_LABEL = "to_label.csv"
HIGH_SUSPECT_THRESHOLD = 3     # score >= this = high suspect
N_TO_LABEL = 400               # how many to pull for labeling (to yield ~200-300 gold after filtering)
SUSPECT_RATIO = 0.6            # fraction of suspects in the labeling pile (the rest are normal)
MIN_LEN = 15                   # drop texts shorter than this (usually undecidable)
RANDOM_SEED = 42
# =====================================================

# ---- signals (based on the labeling guide) ----
PRAISE = ["ดี", "เยี่ยม", "สุดยอด", "เลิศ", "ปัง", "เทพ", "ฟิน", "ประทับใจ",
          "เก่ง", "ดีงาม", "คุณภาพ", "สุดๆ", "เว่อร์", "ที่สุด"]
NEG_CONTEXT = ["รอ", "นาน", "ช้า", "พัง", "เสีย", "แย่", "ผิด", "ยกเลิก", "ไม่มา",
               "หาย", "เจ๊ง", "ห่วย", "งง", "ผิดหวัง", "ไม่คุ้ม", "โกง", "ปัญหา", "ไม่ได้"]
THANKS = ["ขอบคุณ", "ขอบใจ"]
SARCASTIC_EMOJI = ["🙄", "🙃", "👏", "😑", "😏", "🤡", "✨"]


def score_text(t):
    """return (score, signals found) -- signals show the labeler why an item was pulled"""
    s, hits = 0, []
    praise = [w for w in PRAISE if w in t]
    if praise:
        s += 1; hits.append("praise:" + ",".join(praise[:3]))
    emoji = [e for e in SARCASTIC_EMOJI if e in t]
    if emoji:
        s += 2; hits.append("emoji:" + "".join(emoji))
    if re.search(r"5{3,}", t):                 # laughter 555
        s += 1; hits.append("555")
    if re.search(r"(.)\1\1", t):               # elongated chars, e.g. มากกก / เยี่ยมมม
        s += 1; hits.append("elongation")
    if any(w in t for w in THANKS):
        s += 1; hits.append("thanks")
    neg = [w for w in NEG_CONTEXT if w in t]
    if praise and neg:                          # ** strongest signal: praise + negative context **
        s += 3; hits.append("praise+neg:" + ",".join(neg[:3]))
    return s, "; ".join(hits)


# ---- 1) load + basic cleaning ----
df = pd.read_csv(INPUT_CSV)
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() >= MIN_LEN].drop_duplicates("text").reset_index(drop=True)

# ---- 2) score ----
scored = df["text"].apply(lambda t: pd.Series(score_text(t), index=["suspect_score", "signals"]))
df = pd.concat([df, scored], axis=1)
df["group"] = df["suspect_score"].apply(
    lambda x: "high_suspect" if x >= HIGH_SUSPECT_THRESHOLD else "normal"
)
df = df.sort_values("suspect_score", ascending=False).reset_index(drop=True)
df.to_csv(OUTPUT_ALL, index=False, encoding="utf-8-sig")

print("== scoring summary ==")
print(df["group"].value_counts())
print("score distribution:")
print(df["suspect_score"].value_counts().sort_index())

# ---- 3) build the labeling pile: mix high-suspect + normal (avoid skew) ----
n_suspect = int(N_TO_LABEL * SUSPECT_RATIO)
n_normal = N_TO_LABEL - n_suspect
suspect_pool = df[df["group"] == "high_suspect"]
normal_pool = df[df["group"] == "normal"]

take_suspect = suspect_pool.sample(min(n_suspect, len(suspect_pool)), random_state=RANDOM_SEED)
take_normal = normal_pool.sample(min(n_normal, len(normal_pool)), random_state=RANDOM_SEED)

to_label = (
    pd.concat([take_suspect, take_normal])
    .sample(frac=1, random_state=RANDOM_SEED)   # shuffle so labeling can't guess the pile
    .reset_index(drop=True)
)
to_label["label"] = ""     # label column: 1=sarcasm, 0=not sarcasm, X=exclude
to_label["note"] = ""      # reason for hard cases
to_label[["text", "source", "suspect_score", "signals", "label", "note"]].to_csv(
    OUTPUT_TO_LABEL, index=False, encoding="utf-8-sig"
)

print(f"\nlabeling pile {len(to_label)} items (high-suspect {len(take_suspect)} / normal {len(take_normal)})")
print("saved to:", OUTPUT_TO_LABEL)

# notes:
# - keywords catch "suspects", not "definite sarcasm"; a human labeler still decides the truth
# - if the high-suspect pile is too small, lower HIGH_SUSPECT_THRESHOLD to 2
# - for better quality, re-filter to_label with an LLM (round 2)
