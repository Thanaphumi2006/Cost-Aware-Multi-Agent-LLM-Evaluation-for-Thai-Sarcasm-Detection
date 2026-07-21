# -*- coding: utf-8 -*-
"""
Adjudicate only the items where "human vs LLM disagree"
Use alongside labeling_rubric.md (keep the guide open while adjudicating)

Input : ../to_label_reviewed.csv   (has human_label + llm_label)
Output:
  - ../to_label_reviewed.csv        (updates the final_label column)
  - ./gold.csv                      (rebuilt from final_label, keeping only 1/0)

Logic: final_label starts from human_label on every row, then re-decides only the conflicting items
Re-runnable (already-adjudicated items are skipped)  |  no API key needed

Run:  python adjudicate.py
"""

import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                       # the project folder
REVIEWED = os.path.join(BASE, "to_label_reviewed.csv")
GOLD = os.path.join(HERE, "gold.csv")

VALID = {"1", "0", "X"}


def make_gold(df):
    done = df[df["final_label"].isin(["1", "0"])].copy()
    done = done.drop(columns=[c for c in ["label", "note"] if c in done.columns])
    done = done.rename(columns={"final_label": "label"})
    keep = [c for c in ["text", "label", "source", "suspect_score", "signals"] if c in done.columns]
    done[keep].to_csv(GOLD, index=False, encoding="utf-8-sig")
    return len(done), int((done["label"] == "1").sum()), int((done["label"] == "0").sum())


def main():
    df = pd.read_csv(REVIEWED)
    df["human_label"] = df["human_label"].astype(str).str.strip()
    df["llm_label"] = df["llm_label"].astype(str).str.strip()

    # final_label: start from human_label; if it already exists (re-run), keep it
    if "final_label" not in df.columns:
        df["final_label"] = df["human_label"]
    df["final_label"] = df["final_label"].astype(str).str.strip()

    # must be items a human has decided (1/0/X) and that "conflict" with the LLM and are "not yet adjudicated"
    if "adjudicated" not in df.columns:
        df["adjudicated"] = ""
    df["adjudicated"] = df["adjudicated"].astype(str)

    mask = (
        df["human_label"].isin(VALID)
        & (df["human_label"] != df["llm_label"])
        & (df["adjudicated"] != "y")
    )
    todo = df.index[mask].tolist()

    print(f"ข้อที่คน vs LLM เห็นไม่ตรงกัน และยังไม่ชี้ขาด: {len(todo)} ข้อ")
    if not todo:
        n, n1, n0 = make_gold(df)
        print(f"ชี้ขาดครบแล้ว -> gold.csv = {n} ข้อ (ประชด {n1} / ไม่ประชด {n0})")
        return

    print("เปิด labeling_rubric.md อ่านคู่ไปด้วยนะครับ")
    print("พิมพ์:  1 / 0 / X  = คำตัดสินสุดท้าย")
    print("        Enter     = คงคำเดิมของคุณ (human_label)")
    print("        l         = เอาตาม LLM")
    print("        s = ข้าม,  q = บันทึกแล้วออก\n")

    changed = 0
    for n, idx in enumerate(todo, 1):
        text = " ".join(str(df.at[idx, "text"]).split())
        hum = df.at[idx, "human_label"]
        llm = df.at[idx, "llm_label"]
        reason = str(df.at[idx, "llm_reason"]) if "llm_reason" in df.columns else ""

        print("─" * 72)
        print(f"[{n}/{len(todo)}]")
        print(text[:400])
        print(f"\n   คุณเคยให้: [{hum}]   |   LLM ให้: [{llm}]  ({reason[:60]})")

        while True:
            a = input("   ชี้ขาด (1/0/X | Enter=คงเดิม | l=ตาม LLM | s/q): ").strip()
            au = a.upper()
            if a == "":
                final = hum
            elif au == "L":
                final = llm if llm in VALID else hum
            elif au in VALID:
                final = au
            elif au == "S":
                final = None
            elif au == "Q":
                df.to_csv(REVIEWED, index=False, encoding="utf-8-sig")
                n2, n1, n0 = make_gold(df)
                print(f"\nออกกลางคัน บันทึกแล้ว | gold.csv = {n2} (ประชด {n1}/ไม่ประชด {n0})")
                return
            else:
                print("      พิมพ์ได้: 1 / 0 / X / Enter / l / s / q")
                continue
            break

        if final is None:
            continue
        if final != df.at[idx, "final_label"]:
            changed += 1
        df.at[idx, "final_label"] = final
        df.at[idx, "adjudicated"] = "y"

        if n % 5 == 0:
            df.to_csv(REVIEWED, index=False, encoding="utf-8-sig")

    df.to_csv(REVIEWED, index=False, encoding="utf-8-sig")
    n2, n1, n0 = make_gold(df)
    print("\n" + "═" * 72)
    print(f"ชี้ขาดเสร็จ | แก้ป้าย {changed} ข้อ")
    print(f"gold.csv = {n2} ข้อ  (ประชด {n1} / ไม่ประชด {n0})")
    if n1 < 30:
        print(f"⚠ ประชดยังมีแค่ {n1} — แนะนำขุด positive เพิ่ม (harvest_positives.py)")


if __name__ == "__main__":
    main()
