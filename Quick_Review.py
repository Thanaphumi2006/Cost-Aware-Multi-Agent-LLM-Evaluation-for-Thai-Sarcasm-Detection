# -*- coding: utf-8 -*-
"""
Quick one-key review: sweep the harvest pile -> auto-collect sarcasm into gold.csv

Input : Gold/harvest_to_review.csv (the pile the LLM picked as likely sarcasm, sorted most-confident first)
         Gold/gold.csv              (existing, used to avoid duplicates)
Output: Gold/gold.csv             (appends reviewed 1/0 items)
         Gold/gold_backup.csv       (backup of the original before touching it)
         Gold/harvest_to_review.csv (remembers reviewed labels -> rerun resumes where you left off)

Keys:  y = sarcasm   n = not sarcasm   Enter = skip(unsure)   q = save and quit
Stops automatically once sarcasm in gold reaches TARGET_POS items

Run:  python Quick_Review.py
"""

import os
import re
import shutil
import sys

import pandas as pd

# ================== tunable ==================
TARGET_POS = 30      # stop when gold sarcasm reaches this
SAVE_EVERY = 5       # save every N items (avoid losing work)
SHOW_CONF = False    # show the LLM confidence? -- off, so the LLM answer does not bias you
RANK_BY_MARKERS = True   # surface items with sarcasm signals first -> find sarcasm faster (drops nothing)
# =============================================

# sarcasm signals: weights come from measuring on the wisesight side of gold (same source as harvest)
# 555      : appears in 3/6 sarcastic, 0/11 non-sarcastic  -> cleanest
# elongation : appears in 4/6 sarcastic, 1/11 non-sarcastic
# the rest : very thin evidence, low weight, just in case
# caution: the base is only 6 sarcastic items, numbers still swing a lot -- use only to "rank", not to "drop"
MARKERS = [
    (2, lambda t: bool(re.search(r"5{3,}", t))),                    # 555
    (2, lambda t: bool(re.search(r"(.)\1\1", t))),                  # elongated chars, e.g. มากกก เยี่ยมมม
    (1, lambda t: "?" in t or "ไหม" in t),                          # snide question
    (1, lambda t: "ขอบคุณ" in t),
    (1, lambda t: any(e in t for e in "🙄🙃👏😑😏🤡✨😂🤣")),
]


def marker_score(text):
    return sum(w for w, hit in MARKERS if hit(text))

# make Thai display correctly on the Windows console
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_DIR = HERE if os.path.exists(os.path.join(HERE, "gold.csv")) else os.path.join(HERE, "Gold")
GOLD_CSV = os.path.join(GOLD_DIR, "gold.csv")
BACKUP_CSV = os.path.join(GOLD_DIR, "gold_backup.csv")
HARVEST_CSV = os.path.join(GOLD_DIR, "harvest_to_review.csv")
# if this file exists, its items are reviewed first (delete it = back to normal ordering)
SHORTLIST_CSV = os.path.join(GOLD_DIR, "shortlist.csv")

GOLD_COLS = ["text", "label", "source", "suspect_score", "signals"]
DECIDED = {"1", "0"}


def load_csv(path, name):
    if not os.path.exists(path):
        sys.exit(f"หาไฟล์ไม่เจอ: {path}\n(วางสคริปต์ไว้ข้าง {name} หรือข้างโฟลเดอร์ Gold/)")
    # dtype=str: prevent blank labels being read as NaN and miscompared
    return pd.read_csv(path, dtype=str).fillna("")


def to_gold_rows(done):
    """convert a reviewed harvest row -> gold's column format
    harvest is all from wisesight (Harvest.py filters ONLY_WISESIGHT) so source is filled correctly
    suspect_score/signals are not in this pile, leave them blank"""
    rows = pd.DataFrame({
        "text": done["text"],
        "label": done["label"],
        "source": "wisesight",
        "suspect_score": "",
        "signals": "",
    })
    return rows[GOLD_COLS]


def save_all(gold, harvest):
    """write gold (existing + just-reviewed) and remember labels back into harvest"""
    done = harvest[harvest["label"].isin(DECIDED)]
    merged = pd.concat([gold, to_gold_rows(done)], ignore_index=True)
    merged = merged.drop_duplicates(subset="text", keep="first")
    merged.to_csv(GOLD_CSV, index=False, encoding="utf-8-sig")
    # drop the temporary column (_marker) so it does not leak into the file
    keep = [c for c in harvest.columns if not c.startswith("_")]
    harvest[keep].to_csv(HARVEST_CSV, index=False, encoding="utf-8-sig")
    return merged


def count_pos(gold, harvest, in_gold):
    """count all sarcasm: those in gold + just-reviewed in the harvest pile
    must exclude harvest rows already merged into gold (previous round) or it double-counts -> stops before the real target"""
    n = (gold["label"] == "1").sum()
    fresh = harvest[~harvest["text"].isin(in_gold)]
    n += (fresh["label"] == "1").sum()
    return int(n)


def ask():
    while True:
        a = input("   ประชดไหม? [y=ใช่ / n=ไม่ / Enter=ข้าม / q=ออก]: ").strip().lower()
        if a in ("y", "1"):
            return "1"
        if a in ("n", "0"):
            return "0"
        if a == "":
            return "skip"
        if a == "q":
            return "quit"
        print("   กดได้แค่: y / n / Enter / q")


def main():
    gold = load_csv(GOLD_CSV, "gold.csv")
    harvest = load_csv(HARVEST_CSV, "harvest_to_review.csv")

    for col in ("label", "note"):
        if col not in harvest.columns:
            harvest[col] = ""

    if not os.path.exists(BACKUP_CSV):
        shutil.copy2(GOLD_CSV, BACKUP_CSV)
        print(f"สำรอง gold เดิมไว้ที่ {os.path.basename(BACKUP_CSV)} แล้ว")

    # order: shortlist first -> sarcasm signals -> LLM confidence
    # (drops nothing, just moves them up; if mis-picked, you still sweep the rest)
    harvest["llm_conf"] = pd.to_numeric(harvest["llm_conf"], errors="coerce").fillna(0)
    sort_cols = ["llm_conf"]
    if RANK_BY_MARKERS:
        harvest["_marker"] = harvest["text"].map(marker_score)
        sort_cols = ["_marker", "llm_conf"]

    n_short = 0
    if os.path.exists(SHORTLIST_CSV):
        short = pd.read_csv(SHORTLIST_CSV, dtype=str).fillna("")
        order = {t: i for i, t in enumerate(short["text"])}
        # higher rank = higher value (descending) ; not in shortlist = 0
        harvest["_short"] = harvest["text"].map(lambda t: len(order) - order[t] if t in order else 0)
        sort_cols = ["_short"] + sort_cols
        n_short = int((harvest["_short"] > 0).sum())

    harvest = harvest.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    in_gold = set(gold["text"])
    todo = [i for i in harvest.index
            if harvest.at[i, "label"] not in DECIDED and harvest.at[i, "text"] not in in_gold]

    pos = count_pos(gold, harvest, in_gold)
    print(f"\nประชดใน gold ตอนนี้: {pos} ข้อ  (เป้า {TARGET_POS})")
    if pos >= TARGET_POS:
        print("ครบเป้าแล้ว ไม่ต้องตรวจเพิ่ม -> รัน Check_gold.py ต่อได้เลย")
        return
    if not todo:
        print("ไม่เหลือข้อให้ตรวจในกอง harvest แล้ว")
        return

    print(f"เหลือให้ตรวจ {len(todo)} ข้อ | ต้องเก็บประชดอีก ~{TARGET_POS - pos} ข้อ")
    if n_short:
        print(f"({n_short} ข้อแรกมาจาก shortlist -- คัดมาแล้วว่าน่าจะประชด ตรวจกองนี้ก่อน)")
    print("อ่านผ่านๆ ข้อละ 3-5 วิ ไม่ต้องคิดลึก\n")

    decided = 0
    for n, idx in enumerate(todo, 1):
        print("─" * 70)
        head = f"[{n}/{len(todo)}]  ประชดที่เก็บได้: {pos}/{TARGET_POS}"
        if SHOW_CONF:
            head += f"   (llm_conf={harvest.at[idx, 'llm_conf']:.2f})"
        print(head)
        print(f"\n{harvest.at[idx, 'text']}\n")

        ans = ask()
        if ans == "quit":
            break
        if ans == "skip":
            continue

        harvest.at[idx, "label"] = ans
        decided += 1
        if ans == "1":
            pos += 1

        if decided % SAVE_EVERY == 0:
            save_all(gold, harvest)

        if pos >= TARGET_POS:
            print(f"\nครบเป้าแล้ว! ประชด {pos} ข้อ")
            break

    merged = save_all(gold, harvest)
    n1 = int((merged["label"] == "1").sum())
    n0 = int((merged["label"] == "0").sum())

    print("\n" + "═" * 70)
    print(f"ตรวจรอบนี้ {decided} ข้อ")
    print(f"gold.csv = {len(merged)} ข้อ  (ประชด {n1} / ไม่ประชด {n0})")
    if n1 >= TARGET_POS:
        print("→ ครบเป้า รัน Check_gold.py ยืนยันอีกที แล้วไป baseline ได้")
    else:
        print(f"→ ยังขาดประชดอีก {TARGET_POS - n1} ข้อ รันสคริปต์นี้ซ้ำได้ ทำต่อจากเดิม")


if __name__ == "__main__":
    main()
