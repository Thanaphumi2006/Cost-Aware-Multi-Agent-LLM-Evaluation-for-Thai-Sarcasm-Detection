# -*- coding: utf-8 -*-
"""Label text from any domain, ready to feed eval_domain.py -- the tool that closes step 3 (cross-domain)

Accepts any input containing Thai text:
  - .txt : one text per line
  - .csv : has a text column (or specify --text-col)
then label one at a time (same criteria as gold: pretense), writing out a text,label CSV
saves on every answer -> quit midway and rerun to resume (already-labeled items are skipped)

Usage:
  python label_any.py news_raw.txt              -> news_raw_labeled.csv
  python label_any.py comments.csv --out yt.csv --text-col body
  python label_any.py news_raw.txt              (rerun = resume from where you left off)
  then: python eval_domain.py news_raw_labeled.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RULE = """  ┌─ เกณฑ์ (เหมือน gold — ประชดต้องมี "การเสแสร้ง") ─────────────┐
  │  1 = ประชด : แกล้งชม/แกล้งขอบคุณ ทั้งที่จริงไม่พอใจ            │
  │  0 = ไม่   : บ่นตรงๆ · ชมจริง · รีวิวสมดุล · เล่าเฉยๆ          │
  └──────────────────────────────────────────────────────────────┘"""


def load_texts(path, text_col):
    if path.lower().endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    df = pd.read_csv(path, dtype=str).fillna("")
    if text_col not in df.columns:
        sys.exit(f"ไม่มีคอลัมน์ '{text_col}' (มี: {list(df.columns)})")
    return [t.strip() for t in df[text_col] if t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="ไฟล์ข้อความ (.txt ต่อบรรทัด หรือ .csv)")
    ap.add_argument("--out", help="ไฟล์ผล (ค่าเริ่มต้น: <input>_labeled.csv)")
    ap.add_argument("--text-col", default="text")
    a = ap.parse_args()

    out = a.out or (os.path.splitext(a.input)[0] + "_labeled.csv")
    texts = load_texts(a.input, a.text_col)
    # dedupe, preserving order
    seen, uniq = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t); uniq.append(t)

    done = {}
    if os.path.exists(out):
        prev = pd.read_csv(out, dtype=str).fillna("")
        done = dict(zip(prev["text"], prev["label"]))

    queue = [t for t in uniq if t not in done]
    if not queue:
        npos = sum(1 for v in done.values() if v == "1")
        print(f"label ครบแล้ว: {len(done)} ข้อ (ประชด {npos}) -> {out}")
        print(f"ต่อไป: python eval_domain.py {out}")
        return

    print(f"\n label โดเมนใหม่ · เหลือ {len(queue)} (label แล้ว {len(done)}) -> {out}")
    print(RULE)
    print("  1=ประชด  0=ไม่  u=ข้าม  b=ย้อน  q=บันทึก+ออก\n")

    rows = list(done.items())
    hist, pos = [], 0
    while pos < len(queue):
        t = queue[pos]
        npos = sum(1 for _, v in rows if v == "1")
        print("─" * 66)
        print(f" {pos+1}/{len(queue)}  ·  (ประชดแล้ว {npos})\n\n  {t}\n")
        ans = input("  [1/0/u/b/q]: ").strip().lower()
        if ans == "q":
            break
        if ans == "b":
            if hist:
                pos = hist.pop()
                if rows and rows[-1][0] == queue[pos]:
                    rows.pop()
            else:
                print("  (ย้อนไม่ได้)")
            continue
        if ans in ("1", "0", "u"):
            if ans in ("1", "0"):
                rows.append((t, ans))
            pd.DataFrame(rows, columns=["text", "label"]).to_csv(out, index=False, encoding="utf-8-sig")
            hist.append(pos); pos += 1
        else:
            print("  (พิมพ์ 1, 0, u, b, q)")

    npos = sum(1 for _, v in rows if v == "1")
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(out, index=False, encoding="utf-8-sig")
    print("─" * 66)
    print(f"\nบันทึก {len(rows)} ข้อ (ประชด {npos}) -> {out}")
    if npos < 30:
        print(f"⚠ ประชดยัง {npos} ข้อ — เป้าหมาย ≥30 ถึงจะมี CI ที่มีความหมาย label เพิ่มได้")
    print(f"ต่อไป: python eval_domain.py {out}")


if __name__ == "__main__":
    main()
