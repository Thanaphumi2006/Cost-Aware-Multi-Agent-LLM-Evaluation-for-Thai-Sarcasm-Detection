# -*- coding: utf-8 -*-
"""
Mine more "sarcasm suspects" with an LLM (more accurate than keywords) to grow the sarcastic side of gold

Why it exists: keywords find few real sarcasm cases (long Wongnai reviews are noisy)
Approach: focus on Wisesight + drop long texts + let the LLM pick only likely-sarcastic ones
Output: harvest_to_review.csv = the pile the LLM thinks is likely sarcastic, sorted most-confident first
         -> you hand-check (blind) only this pile, finding sarcasm much faster

Input : to_label.csv (source) + gold.csv (already-checked items, to avoid duplicates)
Install:  pip install openai pandas
Set key:  export OPENAI_API_KEY="sk-..."
Run:       python harvest_positives_llm.py
"""

import os, json, time
import pandas as pd
from openai import OpenAI

# robust paths: reference the script file location (run from anywhere)
HERE = os.path.dirname(os.path.abspath(__file__))       # the Gold/ folder
BASE = os.path.dirname(HERE)                            # the project folder

# ============ tunable ============
SRC_CSV = os.path.join(BASE, "scored_texts.csv")  # mine from the big 68k pile (not just 400)
GOLD_CSV = os.path.join(HERE, "gold.csv")         # used to exclude already-checked items
OUT_CSV = os.path.join(HERE, "harvest_to_review.csv")
MODEL = "gpt-4o"
ONLY_WISESIGHT = True                 # focus on the source dense with sarcasm
MAX_LEN = 150                         # drop long texts (sarcasm is usually short and terse)
MAX_SCAN = 800                        # how many items to scan (controls API cost)
SLEEP = 0.3
# =================================

client = OpenAI()

SYSTEM = """ดูข้อความไทยแล้วตอบว่า "น่าจะประชด/เสียดสี" หรือไม่
ประชด = ผิวเผินชม/ขอบคุณ แต่เจตนาจริงคือเหน็บ/บ่น (ความหมายจริงต่างจากผิวเผิน)
ตำหนิตรงๆ หรือชมจริงใจ = ไม่ประชด
ตอบ JSON เท่านั้น: {"maybe_sarcasm": true/false, "conf": 0.0-1.0}"""

def judge(text):
    # retry against network drops / temporary limits (so one blip does not kill the whole job)
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL, max_tokens=40,
                response_format={"type": "json_object"},
                messages=[{"role":"system","content":SYSTEM},
                          {"role":"user","content":f"ข้อความ: {text}"}],
            )
            o = json.loads(r.choices[0].message.content)
            return bool(o.get("maybe_sarcasm", False)), float(o.get("conf", 0))
        except json.JSONDecodeError:
            return False, 0.0
        except Exception as e:
            if attempt == 3:
                print(f"    (skipped 1 item after retry: {type(e).__name__})")
                return False, 0.0
            time.sleep(2 * (attempt + 1))   # 2s, 4s, 6s

# ---- load + exclude already-checked items ----
df = pd.read_csv(SRC_CSV)
reviewed = set()
if os.path.exists(GOLD_CSV):
    reviewed = set(pd.read_csv(GOLD_CSV)["text"].astype(str))

df["text"] = df["text"].astype(str)
cand = df[~df["text"].isin(reviewed)]
if ONLY_WISESIGHT and "source" in cand.columns:
    cand = cand[cand["source"] == "wisesight"]
cand = cand[cand["text"].str.len() <= MAX_LEN]
# sort by suspect_score descending (more likely sarcastic first) then cap the scan count
if "suspect_score" in cand.columns:
    cand = cand.sort_values("suspect_score", ascending=False)
cand = cand.head(MAX_SCAN).reset_index(drop=True)
print(f"will scan {len(cand)} items with the LLM...")

rows = []
for i, t in enumerate(cand["text"], 1):
    maybe, conf = judge(t)
    if maybe:
        rows.append({"text": t, "llm_conf": conf})
    if i % 25 == 0:
        print(f"  ...{i}/{len(cand)}  suspects found {len(rows)}")
        if rows:  # save partial progress so work is not lost
            pd.DataFrame(rows).sort_values("llm_conf", ascending=False).to_csv(
                OUT_CSV, index=False, encoding="utf-8-sig")
    time.sleep(SLEEP)

out = pd.DataFrame(rows).sort_values("llm_conf", ascending=False)
out["label"] = ""   # a column for you to hand-check
out["note"] = ""
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nfound {len(out)} sarcasm suspects -> {OUT_CSV}")
print("next step: hand-check only this pile (blind), then merge the 1s into gold.csv")