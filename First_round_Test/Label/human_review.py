# -*- coding: utf-8 -*-
"""
Step 2: a human reviews/fixes the LLM's draft labels item by item -> produces gold.csv

Input : to_label_prelabeled.csv   (has text, llm_label, llm_reason from step 1)
Output:
  - to_label_reviewed.csv : every item + human_label (the human's final label)
  - gold.csv              : only 1/0-labeled items (X removed), ready to train/evaluate

Two modes (set via the MODE variable below):
  - "blind": hide the LLM answer first -> you decide -> then reveal
             at the end it reports "% agreement with the LLM" (to gauge how much to trust it + reduce bias)
  - "fast" : show the LLM answer immediately -> Enter = agree, type 1/0/X = fix

Keys:  1=sarcasm  0=not sarcasm  X=undecidable  (fast: Enter=agree with the LLM)
                  s=skip for now   q=save and quit
Rerunnable: completed items are skipped, keep running to continue

Run:  python human_review.py
"""

import os
import pandas as pd

# ================== tunable ==================
IN_CSV = "to_label_prelabeled.csv"
REVIEW_CSV = "to_label_reviewed.csv"
GOLD_CSV = "gold.csv"
MODE = "blind"        # "blind" (recommended to start with) or "fast"
N_LIMIT = 50          # how many items to review this round (blind: ~50 recommended, then check the %)
SHOW_SIGNALS = True   # show signals from the keyword step (helpful hints, not the answer)
# =============================================

VALID = {"1", "0", "X"}


def load():
    """load: if a reviewed file already exists = resume"""
    if os.path.exists(REVIEW_CSV):
        df = pd.read_csv(REVIEW_CSV)
    else:
        df = pd.read_csv(IN_CSV)
    if "human_label" not in df.columns:
        df["human_label"] = ""
    df["human_label"] = df["human_label"].fillna("").astype(str)
    return df


def save(df):
    df.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")


def make_gold(df):
    """build gold.csv from human-decided 1/0 items (drop X and unfinished items)"""
    done = df[df["human_label"].isin(["1", "0"])].copy()
    # drop the old label/note columns (blank from labeling) to avoid clashing with human_label
    done = done.drop(columns=[c for c in ["label", "note"] if c in done.columns])
    done = done.rename(columns={"human_label": "label"})
    keep = [c for c in ["text", "label", "source", "suspect_score", "signals"] if c in done.columns]
    done[keep].to_csv(GOLD_CSV, index=False, encoding="utf-8-sig")
    return len(done)


def ask(prompt):
    """take human input, return 1/0/X or s(skip)/q(quit)"""
    while True:
        a = input(prompt).strip().upper()
        if a in VALID or a in ("S", "Q"):
            return a
        print("   พิมพ์ได้แค่: 1 / 0 / X / s(ข้าม) / q(ออก)")


def ask_fast(llm):
    """fast mode: Enter=agree with the LLM, type 1/0/X=fix, s/q"""
    while True:
        a = input(f"   [Enter=เห็นด้วย={llm}] แก้เป็น (1/0/X) / s / q: ").strip().upper()
        if a == "":
            return llm if llm in VALID else "X"
        if a in VALID or a in ("S", "Q"):
            return a
        print("   พิมพ์ได้แค่: Enter / 1 / 0 / X / s / q")


def main():
    df = load()
    todo = df.index[~df["human_label"].isin(VALID)].tolist()
    todo = todo[:N_LIMIT]
    if not todo:
        print("ตรวจครบแล้ว (หรือไม่มีข้อค้าง). สร้าง gold.csv ให้เลย")
        n = make_gold(df)
        print(f"gold.csv = {n} ข้อ")
        return

    print(f"== โหมด: {MODE} | จะตรวจ {len(todo)} ข้อรอบนี้ ==")
    print("พิมพ์: 1=ประชด  0=ไม่ประชด  X=ตัดสินไม่ได้  s=ข้าม  q=บันทึกแล้วออก\n")

    agree = 0        # how many agreed with the LLM (only counting 1/0/X decisions)
    decided = 0      # how many decided this round
    quit_now = False

    for n, idx in enumerate(todo, 1):
        text = str(df.at[idx, "text"])
        llm = str(df.at[idx, "llm_label"]).strip().upper() if "llm_label" in df.columns else "X"
        llm_reason = str(df.at[idx, "llm_reason"]) if "llm_reason" in df.columns else ""

        print("─" * 70)
        print(f"[{n}/{len(todo)}]  (แถว {idx})")
        if SHOW_SIGNALS and "signals" in df.columns:
            sig = str(df.at[idx, "signals"])
            if sig and sig != "nan":
                print(f"สัญญาณคีย์เวิร์ด: {sig}")
        print(f"\n{text}\n")

        if MODE == "blind":
            ans = ask("คุณคิดว่า? (1/0/X | s/q): ")
            if ans == "Q":
                quit_now = True
                break
            if ans == "S":
                continue
            # reveal
            same = (ans == llm)
            mark = "ตรงกับ LLM" if same else "ต่างจาก LLM"
            print(f"   -> LLM ว่า: [{llm}]  {llm_reason}")
            print(f"   -> {mark}")
            df.at[idx, "human_label"] = ans
            agree += 1 if same else 0
            decided += 1
        else:  # fast
            print(f"LLM ว่า: [{llm}]  {llm_reason}")
            ans = ask_fast(llm)
            if ans == "Q":
                quit_now = True
                break
            if ans == "S":
                continue
            df.at[idx, "human_label"] = ans
            agree += 1 if ans == llm else 0
            decided += 1

        if decided % 5 == 0:
            save(df)  # save every 5 items to avoid losing work

    save(df)
    print("\n" + "═" * 70)
    if decided:
        pct = 100.0 * agree / decided
        print(f"ตัดสินรอบนี้ {decided} ข้อ | เห็นตรงกับ LLM {agree} ข้อ = {pct:.1f}%")
        if MODE == "blind":
            if pct >= 85:
                print("→ % สูง: เชื่อ LLM ได้พอควร เปลี่ยนเป็น MODE='fast' ไล่ที่เหลือเร็วๆ ได้")
            else:
                print("→ % ต่ำ: อย่าเพิ่งเชื่อ LLM ตรวจให้ครบทุกข้อแบบ blind ต่อไป")
    else:
        print("ยังไม่ได้ตัดสินข้อไหนรอบนี้")

    n_gold = make_gold(df)
    n_left = int((~df["human_label"].isin(VALID)).sum())
    print(f"\nบันทึก: {REVIEW_CSV}")
    print(f"gold.csv (ป้าย 1/0 เท่านั้น, ตัด X แล้ว) = {n_gold} ข้อ")
    print(f"ยังเหลือให้ตรวจอีก {n_left} ข้อ" + ("  (ออกกลางคัน)" if quit_now else ""))


if __name__ == "__main__":
    main()
