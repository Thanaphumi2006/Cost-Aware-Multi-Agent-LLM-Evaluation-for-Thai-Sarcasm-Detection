# -*- coding: utf-8 -*-
"""
ขั้นที่ 2: คนตรวจ/แก้ป้ายร่างของ LLM ทีละข้อ -> ได้ gold.csv

อินพุต : to_label_prelabeled.csv   (มี text, llm_label, llm_reason จากขั้นที่ 1)
เอาต์พุต:
  - to_label_reviewed.csv : ทุกข้อ + human_label (ป้ายที่คนตัดสินสุดท้าย)
  - gold.csv              : เฉพาะข้อที่ป้าย 1/0 (ตัด X ออก) พร้อมใช้เทรน/วัดผล

สองโหมด (ปรับที่ตัวแปร MODE ด้านล่าง):
  - "blind": ปิดคำตอบ LLM ไว้ก่อน -> คุณตัดสินเอง -> ค่อยเฉลย
             จบรอบจะบอก "% เห็นตรงกับ LLM" (ไว้เช็คว่าเชื่อ LLM ได้แค่ไหน + ลด bias)
  - "fast" : โชว์คำตอบ LLM เลย -> Enter = เห็นด้วย, พิมพ์ 1/0/X = แก้

วิธีใช้ตอนพิมพ์:  1=ประชด  0=ไม่ประชด  X=ตัดสินไม่ได้  (fast: Enter=เห็นด้วยกับ LLM)
                  s=ข้ามไว้ก่อน   q=บันทึกแล้วออก
รันซ้ำได้: ข้อที่ทำแล้วจะถูกข้าม รันต่อได้เรื่อยๆ

รัน:  python human_review.py
"""

import os
import pandas as pd

# ================== ปรับได้ ==================
IN_CSV = "to_label_prelabeled.csv"
REVIEW_CSV = "to_label_reviewed.csv"
GOLD_CSV = "gold.csv"
MODE = "blind"        # "blind" (แนะนำเริ่มด้วยอันนี้) หรือ "fast"
N_LIMIT = 50          # จำนวนข้อที่จะตรวจในรอบนี้ (blind แนะนำ ~50 แล้วดู %)
SHOW_SIGNALS = True   # โชว์สัญญาณจากขั้นคีย์เวิร์ด (ช่วยดู แต่ไม่ใช่คำตอบ)
# =============================================

VALID = {"1", "0", "X"}


def load():
    """โหลด: ถ้ามีไฟล์ reviewed อยู่แล้ว = ทำต่อจากเดิม"""
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
    """สร้าง gold.csv จากข้อที่คนตัดสินเป็น 1/0 (ตัด X และข้อที่ยังไม่ทำออก)"""
    done = df[df["human_label"].isin(["1", "0"])].copy()
    # ทิ้งคอลัมน์ label/note เดิม (ช่องว่างจากตอนติดป้าย) กันชนกับ human_label
    done = done.drop(columns=[c for c in ["label", "note"] if c in done.columns])
    done = done.rename(columns={"human_label": "label"})
    keep = [c for c in ["text", "label", "source", "suspect_score", "signals"] if c in done.columns]
    done[keep].to_csv(GOLD_CSV, index=False, encoding="utf-8-sig")
    return len(done)


def ask(prompt):
    """รับอินพุตจากคน คืนค่า 1/0/X หรือ s(ข้าม)/q(ออก)"""
    while True:
        a = input(prompt).strip().upper()
        if a in VALID or a in ("S", "Q"):
            return a
        print("   พิมพ์ได้แค่: 1 / 0 / X / s(ข้าม) / q(ออก)")


def ask_fast(llm):
    """โหมด fast: Enter=เห็นด้วยกับ LLM, พิมพ์ 1/0/X=แก้, s/q"""
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

    agree = 0        # เห็นตรงกับ LLM กี่ข้อ (นับเฉพาะที่ตัดสิน 1/0/X)
    decided = 0      # ตัดสินไปกี่ข้อรอบนี้
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
            # เฉลย
            same = (ans == llm)
            mark = "✅ ตรงกับ LLM" if same else "❌ ต่างจาก LLM"
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
            save(df)  # เซฟทุก 5 ข้อ กันงานหาย

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
