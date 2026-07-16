# -*- coding: utf-8 -*-
"""ดึงคอมเมนต์ YouTube (ภาษาไทย) เป็นไฟล์ .txt พร้อมป้อน label_any.py — ปิด step 3 โดเมน YouTube

ดึงคอมเมนต์สาธารณะจากคลิปที่ระบุ ด้วย yt-dlp -> กรองเอาเฉพาะที่มีตัวอักษรไทย
-> ตัดอันสั้น/ยาวเกิน/ซ้ำ -> เขียนทีละบรรทัดใน .txt

ใช้เพื่อ "หาข้อความดิบ" สำหรับทดสอบข้ามโดเมนเท่านั้น (คน label เอง ด้วยเกณฑ์เดิม)
เป็นข้อมูลสาธารณะ · ดึงมาเพื่อ validate โมเดลตัวเอง

ใช้:
  python fetch_yt_comments.py "https://youtube.com/watch?v=XXXX" -n 200
  python fetch_yt_comments.py URL1 URL2 URL3 -o yt_raw.txt      # หลายคลิปรวมกัน
  แล้ว: python label_any.py yt_raw.txt   ->   python eval_domain.py yt_raw_labeled.csv
"""
import argparse
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
THAI = re.compile(r"[฀-๿]")


def is_thai(s, min_thai=3):
    return len(THAI.findall(s)) >= min_thai


def clean(s):
    s = re.sub(r"\s+", " ", s).strip()      # ยุบ newline/ช่องว่าง -> คอมเมนต์ 1 = บรรทัด 1
    return s


def fetch(url, limit):
    import yt_dlp
    opts = {
        "getcomments": True, "skip_download": True, "quiet": True, "no_warnings": True,
        "extractor_args": {"youtube": {"comment_sort": ["top"], "max_comments": [str(limit * 3)]}},
    }
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=False)
    return [c.get("text", "") for c in (info.get("comments") or [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+", help="ลิงก์คลิป YouTube (ใส่ได้หลายอัน)")
    ap.add_argument("-n", type=int, default=200, help="เป้าจำนวนคอมเมนต์ไทย (รวมทุกคลิป)")
    ap.add_argument("-o", "--out", default="yt_raw.txt")
    ap.add_argument("--min-len", type=int, default=8, help="สั้นกว่านี้ทิ้ง (ตัวอักษร)")
    ap.add_argument("--max-len", type=int, default=300, help="ยาวกว่านี้ทิ้ง")
    a = ap.parse_args()

    seen, kept = set(), []
    for url in a.urls:
        if len(kept) >= a.n:
            break
        print(f"ดึง {url} ...", file=sys.stderr, flush=True)
        try:
            raw = fetch(url, a.n)
        except Exception as e:
            print(f"  พลาด: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        n_thai = 0
        for c in raw:
            c = clean(c)
            if not is_thai(c) or not (a.min_len <= len(c) <= a.max_len):
                continue
            if c in seen:
                continue
            seen.add(c); kept.append(c); n_thai += 1
            if len(kept) >= a.n:
                break
        print(f"  ได้คอมเมนต์ไทย {n_thai} (รวม {len(kept)})", file=sys.stderr)

    if not kept:
        sys.exit("ไม่ได้คอมเมนต์ไทยเลย — คลิปอาจปิดคอมเมนต์ หรือคอมเมนต์ไม่ใช่ไทย")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(kept))
    print(f"\nเขียน {len(kept)} คอมเมนต์ -> {a.out}")
    print(f"ต่อไป: python label_any.py {a.out}   (label ~150 ข้อ ให้ได้ประชด ≥30)")
    print(f"       python eval_domain.py {a.out.replace('.txt','_labeled.csv')}")


if __name__ == "__main__":
    main()
