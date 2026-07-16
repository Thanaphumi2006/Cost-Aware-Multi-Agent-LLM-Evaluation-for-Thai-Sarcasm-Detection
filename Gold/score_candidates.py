# -*- coding: utf-8 -*-
"""ให้คะแนน P(ประชด) กับ candidate ใน to_label_next.csv ด้วย logprob -> จัดอันดับใหม่ให้ล่าเป้า positive

เหตุผล: อันดับเดิมใช้ keyword signal ซึ่ง noisy (priority 2 = 142 ข้อ แยกไม่ค่อยออก)
logprob ของโมเดลจัดอันดับดีกว่ามาก -> เอา positive candidate (P สูง) ขึ้นบน + ทำแถบก้ำกึ่ง (0.2-0.8) ให้เห็นชัด
คอขวดคือ positive (n=30) -> อยาก label ข้อที่ "น่าจะได้ positive จริง" ก่อน

ใช้ gpt-4.1-mini (โมเดลที่ดีสุด+ถูกใน  finding 9) · DETECT_SYS เดิม · ~$0.03
รัน: python score_candidates.py
"""
import math
import os
import sys

import pandas as pd

import multiagent

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "to_label_next.csv")
MODEL = "gpt-4.1-mini"


def score_one(client, text):
    r = client.chat.completions.create(
        model=MODEL, max_tokens=20, response_format={"type": "json_object"},
        logprobs=True, top_logprobs=20,
        messages=[{"role": "system", "content": multiagent.DETECT_SYS},
                  {"role": "user", "content": f"ข้อความ: {text}"}],
    )
    it, ot = r.usage.prompt_tokens, r.usage.completion_tokens
    for tok in (r.choices[0].logprobs.content or []):
        if tok.token.strip().strip('"') not in ("0", "1"):
            continue
        p0 = p1 = 0.0
        for alt in tok.top_logprobs:
            t = alt.token.strip().strip('"')
            if t == "1":
                p1 += math.exp(alt.logprob)
            elif t == "0":
                p0 += math.exp(alt.logprob)
        if p0 + p1 > 0:
            return p1 / (p0 + p1), it, ot
    return float("nan"), it, ot


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ต้องมี OPENAI_API_KEY")
    df = pd.read_csv(CSV, dtype=str).fillna("")
    from openai import OpenAI
    client = OpenAI(timeout=30.0, max_retries=3)

    probs, ti, to = [], 0, 0
    for n, text in enumerate(df["text"], 1):
        try:
            p, i, o = score_one(client, text)
        except Exception as e:
            print(f"\n  ข้อ {n} พัง: {type(e).__name__}"); p, i, o = float("nan"), 0, 0
        probs.append(p); ti += i; to += o
        print(f"  {n}/{len(df)}", end="\r", flush=True)

    df["P_sarcasm"] = [round(p, 3) if p == p else "" for p in probs]
    # แถบก้ำกึ่ง 0.2-0.8 = ข้อที่โมเดลไม่มั่นใจ = แยกระบบได้ดี + คน label ต้องตัดสินจริง
    df["band"] = ["ก้ำกึ่ง (0.2-0.8)" if (p == p and 0.2 <= p <= 0.8)
                  else ("น่าจะประชด (>0.8)" if (p == p and p > 0.8)
                        else "น่าจะไม่ (<0.2)") for p in probs]
    # จัดอันดับใหม่: P สูงก่อน (ล่า positive) -- ข้อ P สูงคือผู้สมัคร positive ที่ดีสุด
    df["_s"] = [p if p == p else -1 for p in probs]
    df = df.sort_values("_s", ascending=False).drop(columns="_s")
    df.to_csv(CSV, index=False, encoding="utf-8-sig")

    c = ti/1e6*0.40 + to/1e6*1.60
    hi = sum(1 for p in probs if p == p and p > 0.8)
    mid = sum(1 for p in probs if p == p and 0.2 <= p <= 0.8)
    print(f"\nให้คะแนน {len(df)} ข้อ | ${c:.4f} | {ti} in / {to} out")
    print(f"  น่าจะประชด (P>0.8): {hi} ข้อ  <- ผู้สมัคร positive ที่ดีสุด label ก่อน")
    print(f"  ก้ำกึ่ง (0.2-0.8) : {mid} ข้อ  <- ต้องคนตัดสิน แยกระบบได้ดี")
    print(f"เขียนทับ {os.path.basename(CSV)} (เรียง P มาก->น้อย, เพิ่มคอลัมน์ P_sarcasm + band)")


if __name__ == "__main__":
    main()
