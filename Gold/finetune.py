# -*- coding: utf-8 -*-
"""เทรนโมเดล "ของเราเอง" ที่จำ correction ไว้ในน้ำหนักถาวร (fine-tuning) — ต่างจาก few-shot

few-shot (ในเว็บ): เอาตัวอย่างยัดใส่โปรมป์ทุกครั้ง — ถาวรข้ามเซสชัน แต่จำกัดจำนวน + ไม่ได้ฝังในโมเดล
fine-tuning (ไฟล์นี้): เอา gold + correction ทั้งหมดไปเทรน -> ได้ "โมเดลใหม่" ที่ฝังความรู้ในน้ำหนัก
  -> ถาวรจริง ไม่ต้องยัดตัวอย่างในโปรมป์อีก ทำนายเร็ว/ถูกลง และจำได้ "ทุกอัน" ไม่จำกัด

ขั้นตอน:
  1) python finetune.py --export            -> ft_train.jsonl (gold + corrections เป็นรูปแบบเทรน)
  2) python finetune.py --train ft_train.jsonl   -> อัปโหลด + สั่งเทรน (เสียเงิน) พิมพ์ job id
  3) python finetune.py --status <job_id>   -> เช็คสถานะ พอเสร็จได้ชื่อโมเดล ft:...
  4) เอาชื่อโมเดลไปใส่ใน predict.py (OPERATING[...]['model']) หรือ SarcasmDetector(model="ft:...")

ข้อควรรู้ก่อนทำ (ซื่อสัตย์):
  - ต้องมีตัวอย่าง >= 10 (OpenAI ขั้นต่ำ) แนะนำ >= 50 correction ถึงจะเห็นผลชัด — ตอนนี้ correction ยังน้อย
  - เสียเงิน 2 ทอด: ค่าเทรน (ตามจำนวนโทเคน) + โมเดล fine-tuned คิดเงิน/ครั้งแพงกว่าตัวธรรมดา
  - gold ฝั่งประชดมี self-selection bias -> โมเดลจะรับ bias นั้นมาด้วย (เหมือนทุกระบบในโปรเจกต์)
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold.csv")
OUT_JSONL = os.path.join(HERE, "ft_train.jsonl")
BASE_DEFAULT = "gpt-4o-mini-2024-07-18"   # โมเดลฐานที่ fine-tune ได้ (เปลี่ยนได้ด้วย --base)


def _row(text, label):
    import predict
    return {"messages": [
        {"role": "system", "content": predict.DETECT_SYS},
        {"role": "user", "content": f"ข้อความ: {text}"},
        {"role": "assistant", "content": json.dumps({"label": label}, ensure_ascii=False)},
    ]}


def do_export(out):
    import pandas as pd
    import predict
    rows, seen = [], set()
    # 1) correction ของผู้ใช้ (โดเมนจริงที่อยากให้จำ) มาก่อน
    for c in predict.load_corrections():
        t = c["text"].strip()
        if t and t not in seen and c["label"] in ("0", "1"):
            rows.append(_row(t, c["label"])); seen.add(t)
    n_corr = len(rows)
    # 2) gold (คุณภาพสูง) เสริมพื้นฐาน
    g = pd.read_csv(GOLD, dtype=str).fillna("")
    g["label"] = g["label"].str.strip()
    for _, r in g[g["label"].isin(["0", "1"])].iterrows():
        t = r["text"].strip()
        if t and t not in seen:
            rows.append(_row(t, r["label"])); seen.add(t)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"เขียน {len(rows)} ตัวอย่าง -> {out}  (correction {n_corr} + gold {len(rows)-n_corr})")
    if len(rows) < 10:
        print("⚠ น้อยกว่า 10 ตัวอย่าง — OpenAI เทรนไม่ได้ ต้อง label/correct เพิ่ม")
    else:
        print(f"ต่อไป: python finetune.py --train {os.path.basename(out)}")


def do_train(path, base):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ต้องมี OPENAI_API_KEY")
    from openai import OpenAI
    c = OpenAI()
    print(f"อัปโหลด {path} ...")
    f = c.files.create(file=open(path, "rb"), purpose="fine-tune")
    print(f"สร้าง job (base {base}) ...")
    job = c.fine_tuning.jobs.create(training_file=f.id, model=base)
    print(f"job id: {job.id}  ·  สถานะ: {job.status}")
    print(f"เช็ค: python finetune.py --status {job.id}")


def do_status(job_id):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ต้องมี OPENAI_API_KEY")
    from openai import OpenAI
    j = OpenAI().fine_tuning.jobs.retrieve(job_id)
    print(f"สถานะ: {j.status}")
    if j.fine_tuned_model:
        print(f"โมเดลที่ได้: {j.fine_tuned_model}")
        print(f"ใช้: SarcasmDetector(model='{j.fine_tuned_model}')  หรือแก้ OPERATING ใน predict.py")
    elif j.status in ("running", "validating_files", "queued"):
        print("ยังเทรนอยู่ — รอสักครู่แล้วเช็คใหม่")
    elif j.error:
        print(f"error: {j.error}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", nargs="?", const=OUT_JSONL, help="สร้างไฟล์เทรนจาก gold + corrections")
    ap.add_argument("--train", help="อัปโหลด+สั่งเทรนจากไฟล์ jsonl")
    ap.add_argument("--status", help="เช็คสถานะ job")
    ap.add_argument("--base", default=BASE_DEFAULT, help="โมเดลฐาน")
    a = ap.parse_args()
    if a.export is not None:
        do_export(a.export)
    elif a.train:
        do_train(a.train, a.base)
    elif a.status:
        do_status(a.status)
    else:
        ap.print_help()
