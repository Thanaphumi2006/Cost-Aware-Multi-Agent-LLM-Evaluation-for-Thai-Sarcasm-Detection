# -*- coding: utf-8 -*-
"""
ตรวจเร็วกดปุ่มเดียว: ไล่กอง harvest -> เก็บประชดเข้า gold.csv ให้อัตโนมัติ

อินพุต : Gold/harvest_to_review.csv (กองที่ LLM คัดว่าน่าจะประชด เรียงจากมั่นใจสุด)
         Gold/gold.csv              (ของเดิม ใช้กันข้อซ้ำ)
เอาต์พุต: Gold/gold.csv             (เติมข้อที่ตรวจแล้ว 1/0 เข้าไป)
         Gold/gold_backup.csv       (สำรองของเดิมก่อนแตะ)
         Gold/harvest_to_review.csv (จำ label ที่ตรวจไปแล้ว -> รันซ้ำ ทำต่อจากเดิมได้)

กดปุ่ม:  y = ประชด   n = ไม่ประชด   Enter = ข้าม(งง)   q = บันทึกแล้วออก
หยุดเองเมื่อประชดใน gold ครบ TARGET_POS ข้อ

รัน:  python Quick_Review.py
"""

import os
import re
import shutil
import sys

import pandas as pd

# ================== ปรับได้ ==================
TARGET_POS = 30      # หยุดเมื่อประชดใน gold ครบเท่านี้
SAVE_EVERY = 5       # เซฟทุกกี่ข้อ (กันงานหาย)
SHOW_CONF = False    # โชว์ค่าความมั่นใจของ LLM ไหม -- ปิดไว้ เพื่อไม่ให้คำตอบ LLM ชี้นำคุณ
RANK_BY_MARKERS = True   # เอาข้อที่มีสัญญาณประชดขึ้นก่อน -> เจอประชดเร็วขึ้น (ไม่ทิ้งข้อไหน)
# =============================================

# สัญญาณประชด: น้ำหนักมาจากการวัดบน gold ฝั่ง wisesight (แหล่งเดียวกับ harvest)
# 555      : เจอในประชด 3/6, ไม่เจอในไม่ประชดเลย 0/11  -> สะอาดสุด
# ยืดอักษร : เจอในประชด 4/6, ไม่ประชด 1/11
# ที่เหลือ : หลักฐานบางมาก ให้น้ำหนักน้อย เผื่อไว้เฉยๆ
# ระวัง: ฐานคือประชดแค่ 6 ข้อ ตัวเลขยังแกว่งได้มาก -- ใช้แค่ "จัดลำดับ" ไม่ใช้ "ตัดทิ้ง"
MARKERS = [
    (2, lambda t: bool(re.search(r"5{3,}", t))),                    # 555
    (2, lambda t: bool(re.search(r"(.)\1\1", t))),                  # มากกก เยี่ยมมม
    (1, lambda t: "?" in t or "ไหม" in t),                          # คำถามเชิงเหน็บ
    (1, lambda t: "ขอบคุณ" in t),
    (1, lambda t: any(e in t for e in "🙄🙃👏😑😏🤡✨😂🤣")),
]


def marker_score(text):
    return sum(w for w, hit in MARKERS if hit(text))

# โผล่ภาษาไทยบน Windows console ได้ไม่เพี้ยน
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
# ถ้ามีไฟล์นี้ ข้อในนั้นจะถูกยกขึ้นมาตรวจก่อน (ลบไฟล์ทิ้ง = กลับไปเรียงแบบปกติ)
SHORTLIST_CSV = os.path.join(GOLD_DIR, "shortlist.csv")

GOLD_COLS = ["text", "label", "source", "suspect_score", "signals"]
DECIDED = {"1", "0"}


def load_csv(path, name):
    if not os.path.exists(path):
        sys.exit(f"หาไฟล์ไม่เจอ: {path}\n(วางสคริปต์ไว้ข้าง {name} หรือข้างโฟลเดอร์ Gold/)")
    # dtype=str: กัน label ช่องว่างถูกอ่านเป็น NaN แล้วเทียบพลาด
    return pd.read_csv(path, dtype=str).fillna("")


def to_gold_rows(done):
    """แปลงแถว harvest ที่ตรวจแล้ว -> รูปแบบคอลัมน์ของ gold
    harvest มาจาก wisesight ล้วน (Harvest.py กรอง ONLY_WISESIGHT) เลยเติม source ได้ตรง
    suspect_score/signals ไม่มีในกองนี้ ปล่อยว่าง"""
    rows = pd.DataFrame({
        "text": done["text"],
        "label": done["label"],
        "source": "wisesight",
        "suspect_score": "",
        "signals": "",
    })
    return rows[GOLD_COLS]


def save_all(gold, harvest):
    """เขียน gold (เดิม + ที่เพิ่งตรวจ) และจำ label ลง harvest"""
    done = harvest[harvest["label"].isin(DECIDED)]
    merged = pd.concat([gold, to_gold_rows(done)], ignore_index=True)
    merged = merged.drop_duplicates(subset="text", keep="first")
    merged.to_csv(GOLD_CSV, index=False, encoding="utf-8-sig")
    # ตัดคอลัมน์ชั่วคราว (_marker) ไม่ให้หลุดลงไฟล์
    keep = [c for c in harvest.columns if not c.startswith("_")]
    harvest[keep].to_csv(HARVEST_CSV, index=False, encoding="utf-8-sig")
    return merged


def count_pos(gold, harvest, in_gold):
    """นับประชดทั้งหมด: ของใน gold + ที่เพิ่งตรวจในกอง harvest
    ต้องกัน harvest แถวที่ถูกรวมเข้า gold ไปแล้ว (รอบก่อน) ไม่งั้นนับซ้ำ -> หยุดก่อนถึงเป้าจริง"""
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

    # เรียง: shortlist ก่อน -> สัญญาณประชด -> ความมั่นใจ LLM
    # (ไม่ตัดข้อไหนทิ้ง แค่เลื่อนขึ้นมาก่อน ถ้าคัดพลาด ยังไล่กองที่เหลือต่อได้)
    harvest["llm_conf"] = pd.to_numeric(harvest["llm_conf"], errors="coerce").fillna(0)
    sort_cols = ["llm_conf"]
    if RANK_BY_MARKERS:
        harvest["_marker"] = harvest["text"].map(marker_score)
        sort_cols = ["_marker", "llm_conf"]

    n_short = 0
    if os.path.exists(SHORTLIST_CSV):
        short = pd.read_csv(SHORTLIST_CSV, dtype=str).fillna("")
        order = {t: i for i, t in enumerate(short["text"])}
        # ยิ่งอันดับต้น ค่ายิ่งสูง (เรียง descending) ; ไม่อยู่ใน shortlist = 0
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
            print(f"\n🎉 ครบเป้าแล้ว! ประชด {pos} ข้อ")
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
