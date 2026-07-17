# -*- coding: utf-8 -*-
"""ทดสอบข้ามโดเมนจากลิงก์ ครบจบในคำสั่งเดียว: ดึงคอมเมนต์ -> label -> วัด F1

รองรับ Pantip, YouTube, Reddit (ดู fetch_social.py). เหมาะกับการ "validate ข้ามโดเมน":
  python validate_link.py "https://pantip.com/topic/XXXXXXXX"

มันจะ:
  1) ดึงคอมเมนต์ไทย (ข้ามถ้าดึงไว้แล้ว)
  2) เปิดหน้า label ให้ (กด 1=ประชด 0=ไม่ u=ข้าม b=ย้อน q=บันทึกออก · บันทึกทุกครั้ง · รันซ้ำทำต่อได้)
  3) พอ label ครบพอ รัน eval อัตโนมัติ บอก F1 บนโดเมนนี้ เทียบกับ gold เดิม

เคล็ดลับเลือกกระทู้ Pantip ที่มีประชด: เอากระทู้ที่คนถกเถียง/บ่น/การเมือง/ดราม่า
(กระทู้ถาม-ตอบเฉยๆ มักไม่มีประชด จะ label ยากได้ positive น้อย)

ต้องมี OPENAI_API_KEY ตอนขั้น eval (ขั้น label ไม่ต้อง)
"""
import argparse
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def base_name(url, plat):
    h = hashlib.sha1(url.encode()).hexdigest()[:8]
    return os.path.join(HERE, f"domain_{plat}_{h}")


def counts(csv):
    if not os.path.exists(csv):
        return 0, 0
    import pandas as pd
    d = pd.read_csv(csv, dtype=str).fillna("")
    lab = d["label"].str.strip()
    return int(lab.isin(["0", "1"]).sum()), int((lab == "1").sum())


def main():
    ap = argparse.ArgumentParser(description="validate ข้ามโดเมนจากลิงก์ (Pantip/YouTube/Reddit)")
    ap.add_argument("url", help="ลิงก์กระทู้/คลิป")
    ap.add_argument("-n", type=int, default=120, help="จำนวนคอมเมนต์ที่ดึง (ค่าเริ่มต้น 120)")
    ap.add_argument("--min-label", type=int, default=30, help="label อย่างน้อยเท่านี้ก่อน eval (ค่าเริ่มต้น 30)")
    ap.add_argument("--op", default="balanced", choices=["balanced", "high_recall"])
    a = ap.parse_args()

    import fetch_social as fs
    plat = fs.platform_of(a.url)
    base = base_name(a.url, plat)
    raw, labeled = base + "_raw.txt", base + "_raw_labeled.csv"

    # 1) fetch
    if not os.path.exists(raw):
        print(f"[1] ดึงคอมเมนต์จาก {plat} ...")
        try:
            comments, plat = fs.fetch_any(a.url, a.n)
        except fs.UnsupportedError as e:
            sys.exit(f"ดึงจาก {plat} อัตโนมัติไม่ได้ ({e}). รองรับ: Pantip, YouTube, Reddit")
        except Exception as e:
            sys.exit(f"ดึงไม่สำเร็จ: {type(e).__name__}: {e}")
        if not comments:
            sys.exit("ไม่พบคอมเมนต์ภาษาไทย (ลองกระทู้อื่น)")
        with open(raw, "w", encoding="utf-8") as f:
            f.write("\n".join(comments))
        print(f"    ได้ {len(comments)} คอมเมนต์ -> {os.path.basename(raw)}\n")
    else:
        print(f"[1] มีคอมเมนต์แล้ว ({os.path.basename(raw)}) ข้าม\n")

    # 2) label
    ntot, npos = counts(labeled)
    if ntot < a.min_label:
        print(f"[2] label (ทำไปแล้ว {ntot} ข้อ ประชด {npos}) เปิดหน้า label ...\n")
        subprocess.run([PY, os.path.join(HERE, "label_any.py"), raw, "--out", labeled])
        ntot, npos = counts(labeled)
    else:
        print(f"[2] label ครบแล้ว ({ntot} ข้อ ประชด {npos}) ข้าม\n")

    # 3) eval
    if ntot < 10:
        print(f"\nlabel ยังน้อย ({ntot} ข้อ) รันคำสั่งเดิมซ้ำเพื่อ label ต่อ แล้วค่อย eval")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print(f"\n[3] พร้อม eval แล้ว ({ntot} ข้อ) แต่ยังไม่มี OPENAI_API_KEY")
        print(f"    export OPENAI_API_KEY=sk-...  แล้วรัน: python eval_domain.py {os.path.basename(labeled)}")
        return
    if npos < 10:
        print(f"[!] ประชดมีแค่ {npos} ข้อ F1/CI จะหยาบ (แต่ยังพอเห็นทิศทางได้)")
    print(f"[3] วัด F1 บนโดเมน {plat} ({ntot} ข้อ ประชด {npos}) ...\n")
    subprocess.run([PY, os.path.join(HERE, "eval_domain.py"), labeled, "--op", a.op])


if __name__ == "__main__":
    main()
