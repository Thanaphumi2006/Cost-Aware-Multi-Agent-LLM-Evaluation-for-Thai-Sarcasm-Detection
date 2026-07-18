# -*- coding: utf-8 -*-
"""ดึงคอมเมนต์ไทยจากหลายลิงก์/หลายแพลตฟอร์ม รวมเป็น CSV (คอลัมน์ text) — ป้อน distillation/eval

เป็น "ปากทาง" ของ pipeline ขยายข้อมูล: เอา URL อะไรก็ได้ที่ fetch_social รองรับ (YouTube/Pantip/Reddit)
มาเทข้อความดิบเป็น CSV แล้วให้ teacher ติดป้าย silver ต่อ

*** เคล็ดสำคัญ (แก้ปัญหา cross-domain ที่ precision ตก 0.68->0.40) ***
ดึงจาก "โดเมนที่จะเอาไปใช้จริง" (เช่นเว็บบอร์ด Pantip) ไม่ใช่แค่ Wongnai/Wisesight
-> silver จากโดเมนเป้าหมาย = สอน student ให้รู้จักโดเมนนั้น = ตรงจุดที่โมเดลเคยพัง

ใช้:
  python fetch_to_csv.py <url1> <url2> ... --out pool.csv --limit 100
เชนต่อ:
  python batch_eval.py --csv pool.csv --out pool_pred.csv          # teacher ติดป้าย (ถูกครึ่งราคา)
  python distill_label.py --pred pool_pred.csv --out silver.csv    # กรองความมั่นใจ -> silver
  python distill_train_eval.py --silver silver.csv                 # เทรน+วัดผล OOF
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_social import fetch_any, UnsupportedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+", help="ลิงก์ YouTube/Pantip/Reddit (ใส่ได้หลายอัน)")
    ap.add_argument("--out", default="pool.csv")
    ap.add_argument("--limit", type=int, default=80, help="จำนวนคอมเมนต์ต่อ URL")
    a = ap.parse_args()

    import pandas as pd
    rows, seen = [], set()
    for url in a.urls:
        try:
            texts, plat = fetch_any(url, a.limit)
        except UnsupportedError as e:
            print(f"ข้าม {url} — เข้าถึงฟรีไม่ได้ ({e})", file=sys.stderr)
            continue
        n = 0
        for t in texts:
            if t not in seen:
                seen.add(t); rows.append({"text": t, "source": plat, "url": url}); n += 1
        print(f"{plat}: +{n} ข้อความ  ({url})", file=sys.stderr)

    if not rows:
        sys.exit("ไม่ได้ข้อความเลย (ลิงก์เข้าไม่ได้/ไม่มีคอมเมนต์ไทย)")
    pd.DataFrame(rows).to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"เขียน {a.out} · {len(rows)} ข้อความ (unique) จาก {len(a.urls)} ลิงก์ · "
          f"ต่อด้วย batch_eval.py --csv {a.out}")


if __name__ == "__main__":
    main()
