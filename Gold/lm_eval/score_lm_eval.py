# -*- coding: utf-8 -*-
"""แปลงผล lm-evaluation-harness (task thai_sarcasm) -> F1/precision/recall + CSV ฟอร์แมตโปรเจกต์

harness เก็บ per-item ไว้ใน samples_*.jsonl (ต้องรันด้วย --log_samples --output_path ...)
สคริปต์นี้อ่าน loglikelihood ของสองตัวเลือก [ไม่ใช่, ใช่] -> P(ประชด)=softmax(...)[1] -> label
แล้วเขียน CSV ที่มี pred_prob / pred_label / pred_decision เหมือน predict.py และ batch_eval.py
-> เอาไปเข้า compare_systems.py / bootstrap / McNemar ได้เลย (เทียบข้ามทุกระบบได้)

ใช้:
  python lm_eval/score_lm_eval.py --samples lm_eval_out --out typhoon_lmeval_pred.csv
  python lm_eval/score_lm_eval.py --samples lm_eval_out/.../samples_thai_sarcasm_xx.jsonl --out o.csv --threshold 0.5
"""
import argparse
import glob
import json
import math
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")


def find_samples(path):
    if os.path.isdir(path):
        hits = glob.glob(os.path.join(path, "**", "samples_thai_sarcasm_*.jsonl"), recursive=True)
        if not hits:
            sys.exit(f"ไม่พบ samples_thai_sarcasm_*.jsonl ใต้ {path} (รัน harness ด้วย --log_samples --output_path ยัง?)")
        return max(hits, key=os.path.getmtime)   # อันใหม่สุด
    return path


def choice_logliks(rec):
    """ดึง loglikelihood ต่อ choice จาก record (รองรับหลายเวอร์ชันของ harness)"""
    r = rec.get("filtered_resps") or rec.get("resps") or []
    out = []
    for c in r:
        x = c
        while isinstance(x, (list, tuple)) and x:
            x = x[0]
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            return []
    return out


def gold_of(rec):
    t = rec.get("target")
    if isinstance(t, int):
        return t
    doc = rec.get("doc") or {}
    if "gold" in doc:
        return int(doc["gold"])
    return 1 if str(t).strip() in ("1", "ใช่") else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="โฟลเดอร์ output_path หรือไฟล์ samples_*.jsonl")
    ap.add_argument("--out", required=True, help="ไฟล์ CSV ผลลัพธ์")
    ap.add_argument("--threshold", type=float, default=None,
                    help="ตัด P(ประชด) ที่ค่านี้ (ไม่ใส่ = argmax loglik ปกติ)")
    a = ap.parse_args()

    path = find_samples(a.samples)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        sys.exit(f"อ่านไม่เจอ record ใน {path}")

    import pandas as pd
    recs, yt, yp = [], [], []
    for rec in rows:
        lls = choice_logliks(rec)
        doc = rec.get("doc") or {}
        text = str(doc.get("text", "")).replace("\n", " ")
        g = gold_of(rec)
        if len(lls) < 2:
            recs.append({"text": text, "label": g, "pred_prob": None,
                         "pred_label": None, "pred_decision": "error"})
            continue
        m = max(lls[0], lls[1])
        p1 = math.exp(lls[1] - m) / (math.exp(lls[0] - m) + math.exp(lls[1] - m))  # P(ใช่/ประชด)
        pred = (1 if p1 >= a.threshold else 0) if a.threshold is not None else (1 if lls[1] > lls[0] else 0)
        recs.append({"text": text, "label": g, "pred_prob": round(p1, 3),
                     "pred_label": pred, "pred_decision": "sarcasm" if pred else "not_sarcasm"})
        yt.append(g); yp.append(pred)

    pd.DataFrame(recs).to_csv(a.out, index=False, encoding="utf-8-sig")

    TP = sum(t == 1 and p == 1 for t, p in zip(yt, yp))
    FP = sum(t == 0 and p == 1 for t, p in zip(yt, yp))
    FN = sum(t == 1 and p == 0 for t, p in zip(yt, yp))
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    print(f"อ่าน {path}")
    print(f"เขียน {a.out} · n={len(yt)} · positives(true)={sum(yt)} · predicted_pos={sum(yp)}")
    print(f"precision={P:.3f}  recall={R:.3f}  F1={F1:.3f}")


if __name__ == "__main__":
    main()
