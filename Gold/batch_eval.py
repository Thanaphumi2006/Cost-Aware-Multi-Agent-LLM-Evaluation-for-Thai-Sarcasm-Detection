# -*- coding: utf-8 -*-
"""ประเมินแบบ batch (offline) ผ่าน OpenAI Batch API — ถูกลง 50% แลกกับ "ไม่เรียลไทม์"

ใช้กับงาน "ยิงทั้งไฟล์ทีเดียว" (frontier/baseline/re-run บน gold ที่ขยายแล้ว) เท่านั้น
เว็บสด (app.py/predict.py) ยังต้องใช้ call ปกติ เพราะ batch รอได้ถึง 24 ชม.

ตรรกะเหมือน predict.py ทุกอย่าง: prompt เดียวกัน (DETECT_SYS), อ่าน logprob -> P(ประชด),
เทียบ threshold จาก OPERATING -> label. **ไม่ใส่ corrections** (ประเมิน "ระบบฐาน" ให้สะอาด ไม่ leak)

ขั้นตอน (async — batch ไม่เสร็จทันที):
  1) python batch_eval.py --csv in.csv --dry-run          เขียน .batch.jsonl + นับจำนวน (ฟรี ไม่ยิง)
  2) python batch_eval.py --csv in.csv --out out.csv       ยิง + รอจนเสร็จ + เขียนผล (blocking)
     หรือแยกเป็นสองจังหวะถ้าไม่อยากรอค้าง:
  2a) python batch_eval.py --csv in.csv --no-wait          ยิงแล้วพิมพ์ batch id ออกมา แล้วออก
  2b) python batch_eval.py --csv in.csv --out out.csv --fetch <batch_id>   ดึงผลของ batch ที่เสร็จแล้ว

หมายเหตุ: --fetch ต้องใช้ --csv "ไฟล์เดิม ลำดับเดิม" เพราะ map ผลกลับด้วย custom_id = row-<index>
"""
import argparse
import json
import math
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from predict import DETECT_SYS, OPERATING   # ใช้ prompt + จุดทำงาน (model/threshold) ชุดเดียวกับของจริง


def build_request(i, text, model):
    """หนึ่งบรรทัดใน JSONL = หนึ่ง request (body เดียวกับที่ predict._call ยิง แต่ไม่ใส่ corrections)"""
    return {
        "custom_id": f"row-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "max_tokens": 20,
            "response_format": {"type": "json_object"},
            "logprobs": True,
            "top_logprobs": 20,
            "messages": [
                {"role": "system", "content": DETECT_SYS},
                {"role": "user", "content": f"ข้อความ: {text}"},
            ],
        },
    }


def prob_from_body(body):
    """แกะ P(ประชด) จาก response body (dict จาก batch output) — ตรรกะเดียวกับ predict._call"""
    try:
        ch = body["choices"][0]
    except (KeyError, IndexError):
        return float("nan")
    lp = ch.get("logprobs") or {}
    for tok in (lp.get("content") or []):
        if tok.get("token", "").strip().strip('"') not in ("0", "1"):
            continue
        p0 = p1 = 0.0
        for alt in tok.get("top_logprobs") or []:
            t = alt.get("token", "").strip().strip('"')
            if t == "1":
                p1 += math.exp(alt["logprob"])
            elif t == "0":
                p0 += math.exp(alt["logprob"])
        if p0 + p1 > 0:
            return p1 / (p0 + p1)
    # อ่าน logprob ไม่ได้ -> ใช้คำตอบ hard จาก JSON
    try:
        return 1.0 if str(json.loads(ch["message"]["content"] or "{}").get("label", "")) == "1" else 0.0
    except (json.JSONDecodeError, KeyError, TypeError):
        return float("nan")


def load_texts(csv_path, text_col):
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if text_col not in df.columns:
        sys.exit(f"ไม่มีคอลัมน์ '{text_col}' (มี: {list(df.columns)})")
    return df


def write_jsonl(path, df, text_col, model):
    with open(path, "w", encoding="utf-8") as f:
        for i, text in enumerate(df[text_col]):
            f.write(json.dumps(build_request(i, text, model), ensure_ascii=False) + "\n")


def parse_output(raw_text):
    """map custom_id 'row-<i>' -> prob จากไฟล์ผลของ batch (JSONL)"""
    probs = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cid = rec.get("custom_id", "")
        resp = rec.get("response") or {}
        if resp.get("status_code") != 200:
            probs[cid] = float("nan")
            continue
        probs[cid] = prob_from_body(resp.get("body") or {})
    return probs


def finish(df, probs, t, out_path):
    """เอา probs (dict custom_id->prob) มาเทียบ threshold แล้วเขียน out.csv"""
    labels, prob_col, dec = [], [], []
    for i in range(len(df)):
        p = probs.get(f"row-{i}", float("nan"))
        if p != p:
            labels.append(None); prob_col.append(None); dec.append("error"); continue
        lab = 1 if p >= t else 0
        labels.append(lab); prob_col.append(round(p, 3))
        dec.append("sarcasm" if lab else "not_sarcasm")
    df = df.copy()
    df["pred_prob"] = prob_col
    df["pred_label"] = labels
    df["pred_decision"] = dec
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n = len(df); ns = sum(1 for d in dec if d == "sarcasm"); ne = sum(1 for d in dec if d == "error")
    print(f"เขียน {out_path} · {n} ข้อ · ประชด {ns}" + (f" · error {ne}" if ne else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="ไฟล์ input (ต้องมีคอลัมน์ข้อความ)")
    ap.add_argument("--out", help="ไฟล์ output (ต้องใส่เวลาจะเขียนผล)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--op", default="balanced", choices=list(OPERATING))
    ap.add_argument("--model", help="ทับ model ของ --op (เช่นกวาด frontier)")
    ap.add_argument("--dry-run", action="store_true", help="เขียน .jsonl + นับจำนวน ไม่ยิง API")
    ap.add_argument("--no-wait", action="store_true", help="ยิงแล้วพิมพ์ batch id ออก ไม่รอ")
    ap.add_argument("--fetch", help="ดึงผลของ batch id ที่เสร็จแล้ว (ใช้คู่ --csv เดิม + --out)")
    ap.add_argument("--poll", type=int, default=20, help="ช่วงเวลาถาม status (วินาที)")
    a = ap.parse_args()

    df = load_texts(a.csv, a.text_col)
    model = a.model or OPERATING[a.op]["model"]
    t = OPERATING[a.op]["t"]

    # --dry-run: เขียน jsonl อย่างเดียว
    jsonl_path = os.path.splitext(a.csv)[0] + ".batch.jsonl"
    if a.dry_run:
        write_jsonl(jsonl_path, df, a.text_col, model)
        print(f"[dry-run] เขียน {jsonl_path} · {len(df)} requests · model={model} · ไม่ได้ยิง (batch = ~50% ของราคาปกติ)")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ต้องมี OPENAI_API_KEY (export OPENAI_API_KEY=sk-...)")
    from openai import OpenAI
    client = OpenAI(timeout=60.0, max_retries=3)

    # --fetch: ข้ามการยิง ไปดึงผลของ batch เดิม
    if a.fetch:
        b = client.batches.retrieve(a.fetch)
        if b.status != "completed":
            sys.exit(f"batch {a.fetch} ยังไม่เสร็จ (status={b.status})")
        if not a.out:
            sys.exit("ใส่ --out ด้วยตอน --fetch")
        raw = client.files.content(b.output_file_id).text
        finish(df, parse_output(raw), t, a.out)
        return

    # ยิงใหม่: เขียน jsonl -> upload -> create batch
    write_jsonl(jsonl_path, df, a.text_col, model)
    up = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h")
    print(f"[batch] ยิงแล้ว · id={batch.id} · {len(df)} requests · model={model}", file=sys.stderr)

    if a.no_wait:
        print(batch.id)
        print(f"ดึงผลทีหลังด้วย:  python batch_eval.py --csv {a.csv} --out out.csv --fetch {batch.id}",
              file=sys.stderr)
        return

    if not a.out:
        sys.exit("ใส่ --out ด้วย (หรือใช้ --no-wait แล้วค่อย --fetch)")

    # รอจนเสร็จ (batch จริงมักเสร็จเร็วกว่า 24 ชม.มาก แต่เป็น async)
    while True:
        b = client.batches.retrieve(batch.id)
        rc = b.request_counts
        print(f"[batch] status={b.status} · {rc.completed}/{rc.total} เสร็จ · fail {rc.failed}", file=sys.stderr)
        if b.status == "completed":
            break
        if b.status in ("failed", "expired", "cancelled"):
            sys.exit(f"batch จบแบบไม่สำเร็จ: {b.status}")
        time.sleep(a.poll)

    raw = client.files.content(b.output_file_id).text
    probs = parse_output(raw)
    if b.error_file_id:   # ถ้ามีบางข้อ error เก็บไว้ให้รู้
        err = client.files.content(b.error_file_id).text
        print(f"[batch] มี error บางข้อ — ดูรายละเอียด {len(err.splitlines())} บรรทัด", file=sys.stderr)
    finish(df, probs, t, a.out)


if __name__ == "__main__":
    main()
