# -*- coding: utf-8 -*-
"""ทดสอบข้ามโดเมน YouTube ครบจบในคำสั่งเดียว — fetch -> label -> eval อัตโนมัติ

แทนที่จะรัน 3 สคริปต์ทีละอัน สั่งครั้งเดียว:
  python validate_yt.py "https://youtube.com/watch?v=XXXX"

มันจะ:
  1) ดึงคอมเมนต์ไทย (ข้ามถ้าดึงไว้แล้ว)
  2) เปิดหน้า label ให้ (กด 1/0/u/b/q · บันทึกทุกครั้ง · ปิดแล้วรันซ้ำทำต่อ)
  3) พอ label ครบพอ (ประชด ≥ เป้า) รัน eval อัตโนมัติ แล้วสรุปว่า "ใช้บน YouTube ได้ไหม"

รันซ้ำคำสั่งเดิมได้เรื่อยๆ — มันจำว่าทำถึงไหนแล้ว
ต้องมี OPENAI_API_KEY ตอนขั้น eval (ขั้น label ไม่ต้อง)
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def vid(url):
    m = re.search(r"(?:v=|youtu\.be/|/shorts/)([\w-]{6,})", url)
    return m.group(1) if m else "yt"


def count_pos(csv):
    if not os.path.exists(csv):
        return 0, 0
    import pandas as pd
    d = pd.read_csv(csv, dtype=str).fillna("")
    lab = d["label"].str.strip()
    return int((lab == "1").sum()), int(lab.isin(["0", "1"]).sum())


def main():
    ap = argparse.ArgumentParser(description="ทดสอบโมเดลบนคอมเมนต์ YouTube ครบจบคำสั่งเดียว")
    ap.add_argument("url", help="ลิงก์คลิป YouTube")
    ap.add_argument("-n", type=int, default=200, help="จำนวนคอมเมนต์ที่ดึง (ค่าเริ่มต้น 200)")
    ap.add_argument("--target-pos", type=int, default=30, help="ประชดขั้นต่ำก่อนจะ eval (ค่าเริ่มต้น 30)")
    ap.add_argument("--op", default="balanced", choices=["balanced", "high_recall"])
    a = ap.parse_args()

    base = os.path.join(HERE, f"yt_{vid(a.url)}")
    raw, labeled = base + "_raw.txt", base + "_raw_labeled.csv"

    # 1) fetch (ข้ามถ้ามีแล้ว)
    if not os.path.exists(raw):
        print("① ดึงคอมเมนต์...\n")
        r = subprocess.run([PY, os.path.join(HERE, "fetch_yt_comments.py"), a.url, "-n", str(a.n), "-o", raw])
        if r.returncode or not os.path.exists(raw):
            sys.exit("ดึงคอมเมนต์ไม่สำเร็จ")
    else:
        print(f"① มีคอมเมนต์แล้ว ({raw}) — ข้าม\n")

    # 2) label (interactive)
    npos, ntot = count_pos(labeled)
    if npos < a.target_pos:
        print(f"② label (มีประชด {npos}/{a.target_pos} แล้ว) — เปิดหน้า label...\n")
        subprocess.run([PY, os.path.join(HERE, "label_any.py"), raw, "--out", labeled])
        npos, ntot = count_pos(labeled)
    else:
        print(f"② label ครบแล้ว (ประชด {npos}) — ข้าม\n")

    # 3) eval (ถ้าประชดพอ)
    if npos < a.target_pos:
        print(f"\nยัง label ประชดไม่ถึงเป้า ({npos}/{a.target_pos}) — รันคำสั่งเดิมซ้ำเพื่อ label ต่อ")
        print("(หรือ eval เลยทั้งที่ยังน้อยก็ได้ แต่ CI จะกว้างมาก)")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print(f"\n③ พร้อม eval แล้ว (ประชด {npos}) แต่ยังไม่มี OPENAI_API_KEY")
        print(f"   export OPENAI_API_KEY=sk-...  แล้วรัน: python eval_domain.py {os.path.basename(labeled)}")
        return
    print(f"③ eval บน YouTube (ประชด {npos} จาก {ntot})...\n")
    subprocess.run([PY, os.path.join(HERE, "eval_domain.py"), labeled, "--op", a.op])


if __name__ == "__main__":
    main()
