# -*- coding: utf-8 -*-
"""เครื่องมือ label ประชดในเทอร์มินัล -- เติม human_label ใน to_label_next.csv

ออกแบบให้ label เร็วและปลอดภัย:
  - โชว์ทีละข้อ เรียงตาม P(ประชด) มาก->น้อย (ผู้สมัคร positive ก่อน) เฉพาะข้อที่ยังไม่ label
  - กด 1/0 ตัดสิน, u=ไม่แน่ใจ(ข้าม), b=ย้อนกลับ, q=บันทึกแล้วออก
  - **บันทึกลงไฟล์ทุกครั้งที่ตอบ** -> ปิดกลางคันได้ เปิดใหม่ทำต่อจากเดิม
  - เตือน rubric: ประชดต้องมี "การเสแสร้ง" (แกล้งชม) -- รีวิวสมดุล/บ่นตรงๆ = 0

รัน:  python label_cli.py            (label เฉพาะ P>0.2 = ~37 ข้อที่คุ้มสุด)
      python label_cli.py --all      (label ทุกข้อรวม P<0.2)
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "to_label_next.csv")

RULE = """  ┌─ เกณฑ์ตัดสิน (ประชดต้องมี "การเสแสร้ง") ─────────────────────┐
  │  1 = ประชด : แกล้งชม/แกล้งขอบคุณ ทั้งที่จริงไม่พอใจ            │
  │  0 = ไม่   : รีวิวสมดุล (ชมจริง+ติจริง) · บ่นตรงๆ · เล่าเฉยๆ   │
  └──────────────────────────────────────────────────────────────┘"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="รวมข้อ P<0.2 ด้วย")
    a = ap.parse_args()

    if not os.path.exists(CSV):
        sys.exit(f"ไม่พบ {CSV}")
    df = pd.read_csv(CSV, dtype=str).fillna("")
    if "human_label" not in df.columns:
        df["human_label"] = ""

    def pnum(v):
        try:
            return float(v)
        except ValueError:
            return -1.0

    # คิวงาน: ยังไม่ label + (ถ้าไม่ใส่ --all) เอาเฉพาะ P>0.2 ; เรียง P มาก->น้อย
    order = df.assign(_p=df["P_sarcasm"].map(pnum)).sort_values("_p", ascending=False).index
    queue = [i for i in order
             if df.at[i, "human_label"].strip() == ""
             and (a.all or pnum(df.at[i, "P_sarcasm"]) > 0.2)]
    done = int((df["human_label"].str.strip() != "").sum())

    if not queue:
        print(f"ไม่มีข้อค้าง (label ไปแล้ว {done} ข้อ). ใช้ --all เพื่อดูข้อ P<0.2")
        return

    print(f"\n label ประชดภาษาไทย · เหลือ {len(queue)} ข้อ (label แล้ว {done})")
    print(RULE)
    print("  พิมพ์:  1=ประชด  0=ไม่ประชด  u=ไม่แน่ใจ(ข้าม)  b=ย้อนกลับ  q=บันทึก+ออก\n")

    hist = []          # ประวัติ index ที่เพิ่งตอบ -> ให้ b ย้อนได้
    pos = 0
    while pos < len(queue):
        i = queue[pos]
        r = df.loc[i]
        n_pos = int((df["human_label"].str.strip() == "1").sum())
        print("─" * 66)
        print(f" ข้อ {pos+1}/{len(queue)}  ·  โมเดลเดา P(ประชด)={r['P_sarcasm']}  ·  "
              f"แหล่ง {r['source']}  ·  (ได้ประชดแล้ว {n_pos})")
        print(f"\n  {r['text']}\n")
        if r.get("llm_reason", "").strip():
            print(f"  เหตุผลโมเดล: {r['llm_reason'][:110]}")
        ans = input("  คำตอบ [1/0/u/b/q]: ").strip().lower()

        if ans == "q":
            break
        if ans == "b":
            if hist:
                pos = hist.pop()
                df.at[queue[pos], "human_label"] = ""   # ล้างคำตอบเก่าให้ตอบใหม่
                df.to_csv(CSV, index=False, encoding="utf-8-sig")
            else:
                print("  (ย้อนไม่ได้ อยู่ข้อแรกแล้ว)")
            continue
        if ans in ("1", "0"):
            df.at[i, "human_label"] = ans
            df.to_csv(CSV, index=False, encoding="utf-8-sig")   # เซฟทุกครั้ง
            hist.append(pos); pos += 1
        elif ans == "u":
            df.at[i, "human_label"] = "X"                       # ไม่แน่ใจ -> ไม่เข้า gold
            df.to_csv(CSV, index=False, encoding="utf-8-sig")
            hist.append(pos); pos += 1
        else:
            print("  (พิมพ์ 1, 0, u, b หรือ q)")

    done = int((df["human_label"].str.strip().isin(["0", "1"])).sum())
    npos = int((df["human_label"].str.strip() == "1").sum())
    print("─" * 66)
    print(f"\nบันทึกแล้ว. label ทั้งหมด {done} ข้อ (ประชด {npos} · ไม่ประชด {done-npos})")
    print(f"ไฟล์: {CSV}")
    print("ทำต่อ: python label_cli.py   ·   เสร็จแล้วบอกให้ผม merge เข้า gold.csv + รันเทียบใหม่")


if __name__ == "__main__":
    main()
