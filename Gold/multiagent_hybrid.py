# -*- coding: utf-8 -*-
"""ระบบ ⑤ -- HYBRID: เอา debate มาเป็น verifier ของ pipeline

ที่มา (สำคัญ -- นี่คือการทดลองแบบแยกตัวแปร):
  pipeline v2 (F1 0.744) ชนะ เพราะ verifier "ถูกโครงสร้างบังคับ" ให้พลิกได้ทางเดียว (1->0)
      -> หวง recall 1.000 ที่ detector ได้มาฟรีไว้ได้
  debate (F1 0.694) แพ้ เพราะ "ตัดสินใหม่ได้ทั้งสองทาง" ทุกข้อ
      -> ทนายแก้ต่างสำเร็จให้ประชดจริง 5 ข้อ recall ร่วงเหลือ 0.833

แต่ debate มี "การไตร่ตรองที่ลึกกว่า" (สองฝ่ายเถียงกันก่อนตัดสิน) ซึ่ง verifier เดี่ยวไม่มี
คำถาม: ความลึกนั้นมีค่าจริงไหม -- หรือที่ pipeline ชนะเป็นเพราะ "ข้อจำกัด" ล้วนๆ

HYBRID = ความลึกของ debate + ข้อจำกัดของ pipeline
  detector (เหมือน baseline เป๊ะ)
    -> ถ้าตอบ 0: จบเลย (verifier เพิ่มประชดใหม่ไม่ได้อยู่แล้ว -> ยิงไปก็เปลืองเปล่า)
    -> ถ้าตอบ 1: อัยการ vs ทนาย เถียงกัน -> ผู้พิพากษาตัดสิน
                 *** ผู้พิพากษาปัดตกได้อย่างเดียว (1->0) เพิ่มประชดใหม่ไม่ได้ ***

ตีความผล:
  ถ้า hybrid ชนะ v2  -> การไตร่ตรองมีค่า (debate แค่เอาไปใช้ผิดที่)
  ถ้า hybrid เสมอ/แพ้ -> "ข้อจำกัด" ต่างหากที่สำคัญ ไม่ใช่จำนวน agent หรือความลึก

รัน: python multiagent_hybrid.py
"""
import json
import os
import sys
import time

import pandas as pd

from baseline import MODELS, PRICE_PER_MTOK, metrics
from multiagent import DETECT_SCHEMA, DETECT_SYS, _ask
from multiagent_debate import ARG_TOKENS, DEFE_SYS, PROS_SYS, _argue

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_CSV = os.path.join(HERE, "gold.csv")

PROVIDER = "gpt"
VARIANT = "hybrid"
PRED_CSV = os.path.join(HERE, f"multiagent_preds_{PROVIDER}_{VARIANT}.csv")

LIMIT = None
SAVE_EVERY = 10
SLEEP_SEC = 0.2

COLS = ["pred", "detect", "pros", "defe", "judge", "in_tok", "out_tok", "calls", "latency_ms", "err"]

# ผู้พิพากษาแบบ "ถูกจำกัดอำนาจ" -- ต่างจาก judge ของ debate ตรงนี้จุดเดียว
# (กฎก้ำกึ่ง = คงไว้เป็นประชด เหมือน verifier ของ v2 เป๊ะ -> เทียบกันได้ยุติธรรม)
JUDGE_SYS = """คุณคือ "ผู้พิพากษา" -- แต่ **อำนาจของคุณถูกจำกัด**

มีคนตัดสินมาก่อนแล้วว่าข้อความไทยนี้ "ประชด" และคนนั้นจับประชดเก่งมาก (แทบไม่พลาด)
หน้าที่คุณคือ "ตรวจจับความผิดพลาดชัดๆ" เท่านั้น -- ไม่ใช่ตัดสินใหม่ตั้งแต่ต้น
คุณ **ปัดตกได้อย่างเดียว** (ประชด -> ไม่ประชด) จะเพิ่มประชดใหม่ไม่ได้

คุณจะได้: ข้อความต้นฉบับ + คำแถลงอัยการ (ว่าประชด) + คำแถลงทนาย (ว่าไม่ประชด)

พลิกเป็น "ไม่ประชด (0)" เฉพาะเมื่อ **ทนายพิสูจน์ได้ชัดเจน** ว่าเข้าข้อใดข้อหนึ่งนี้:
  - บ่น/ตำหนิตรงๆ ล้วนๆ ไม่มีการแกล้งชมหรือแกล้งขอบคุณเลย  ["ลบตรงๆ != ประชด"]
  - ชมจริงใจล้วนๆ ไม่มีนัยเหน็บ
  - รีวิวสมดุลตรงไปตรงมา: ชมจริงบางจุด ติจริงบางจุด ตามจริง ไม่ได้เสแสร้ง
  - แค่เล่าเหตุการณ์/อ้างคำพูดคนอื่น ไม่มีโทนเหน็บของผู้เขียนเอง

อย่าเชื่อฝ่ายไหนเพียงเพราะเขียนน่าเชื่อ -- กลับไปดู "ข้อความต้นฉบับ" เป็นหลัก
ถ้าฝ่ายไหนอ้างหลักฐานที่ไม่มีจริงในข้อความ ให้ตัดทิ้ง

ถ้า **ไม่ชัด** -- แม้จะก้ำกึ่งหรืออ่านได้สองแง่ -- ให้ **คงไว้เป็นประชด (1)**
เหตุผล: ประชดไทยมักแนบเนียน อ่านได้สองแง่เป็นเรื่องปกติของประชด ลังเล = น่าจะประชด

ตอบ JSON เท่านั้น: {"verdict": "1" หรือ "0"}"""

JUDGE_SCHEMA = {"type": "object",
                "properties": {"verdict": {"type": "string", "enum": ["0", "1"]}},
                "required": ["verdict"], "additionalProperties": False}


def _make_client():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ไม่พบ OPENAI_API_KEY")
    from openai import OpenAI
    return OpenAI()


def _judge(client, text, pros, defe):
    user = (f"ข้อความต้นฉบับ:\n{text}\n\n"
            f"คำแถลงอัยการ (ว่าประชด):\n{pros}\n\n"
            f"คำแถลงทนาย (ว่าไม่ประชด):\n{defe}\n\n"
            f"ตัดสิน (ปัดตกได้อย่างเดียว):")
    r = client.chat.completions.create(
        model=MODELS[PROVIDER], max_tokens=20,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE_SYS},
                  {"role": "user", "content": user}],
    )
    raw = (r.choices[0].message.content or "").strip()
    return str(json.loads(raw).get("verdict", "")).strip(), r.usage.prompt_tokens, r.usage.completion_tokens


def run_hybrid(client, text):
    t0 = time.perf_counter()
    ti = to = calls = 0
    pros = defe = verdict = ""

    det, i, o = _ask(client, DETECT_SYS, DETECT_SCHEMA, "label", text)
    ti += i; to += o; calls += 1
    if det not in ("0", "1"):
        return {"pred": "err", "detect": str(det), "pros": "", "defe": "", "judge": "",
                "in_tok": ti, "out_tok": to, "calls": calls,
                "latency_ms": round((time.perf_counter() - t0) * 1000), "err": "detector เพี้ยน"}

    if det == "0":
        # ผู้พิพากษาเพิ่มประชดใหม่ไม่ได้อยู่แล้ว -> ยิงไปก็เปลี่ยนอะไรไม่ได้ ประหยัด 2 calls
        return {"pred": "0", "detect": "0", "pros": "", "defe": "", "judge": "",
                "in_tok": ti, "out_tok": to, "calls": calls,
                "latency_ms": round((time.perf_counter() - t0) * 1000), "err": ""}

    try:
        pros, i, o = _argue(client, PROS_SYS, f"ข้อความ: {text}")
        ti += i; to += o; calls += 1
        defe, i, o = _argue(client, DEFE_SYS, f"ข้อความ: {text}")
        ti += i; to += o; calls += 1
        verdict, i, o = _judge(client, text, pros, defe)
        ti += i; to += o; calls += 1
    except Exception as e:
        return {"pred": "err", "detect": det, "pros": pros[:300], "defe": defe[:300], "judge": "",
                "in_tok": ti, "out_tok": to, "calls": calls,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "err": f"{type(e).__name__}: {e}"[:200]}

    pred = verdict if verdict in ("0", "1") else "err"
    return {"pred": pred, "detect": det, "pros": pros[:300], "defe": defe[:300], "judge": verdict,
            "in_tok": ti, "out_tok": to, "calls": calls,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "err": "" if pred != "err" else f"bad verdict: {verdict[:40]}"}


def load():
    g = pd.read_csv(GOLD_CSV, dtype=str).fillna("")
    g["label"] = g["label"].str.strip()
    g = g[g["label"].isin(["0", "1"])].reset_index(drop=True)
    for c in COLS:
        g[c] = ""
    if os.path.exists(PRED_CSV):
        old = pd.read_csv(PRED_CSV, dtype=str).fillna("")
        m = old.set_index("text")
        for i, t in enumerate(g["text"]):
            if t in m.index:
                for c in COLS:
                    if c in old.columns:
                        g.at[i, c] = m.at[t, c]
    return g


def main():
    df = load()
    todo = df.index[~df["pred"].isin(["0", "1"])].tolist()
    if LIMIT:
        todo = todo[:LIMIT]
    print(f"gold {len(df)} ข้อ | ต้องรัน {len(todo)} ข้อ | โมเดล {MODELS[PROVIDER]}")
    print("HYBRID: detector -> (อัยการ vs ทนาย -> ผู้พิพากษาที่ปัดตกได้อย่างเดียว)\n")

    client = _make_client() if todo else None
    t0 = time.time()
    for n, idx in enumerate(todo, 1):
        out = run_hybrid(client, str(df.at[idx, "text"]))
        for c in COLS:
            df.at[idx, c] = str(out[c])
        flag = "!" if out["pred"] == "err" else " "
        flip = " (พลิกทิ้ง)" if out["detect"] == "1" and out["pred"] == "0" else ""
        print(f"{flag}[{n}/{len(todo)}] det={out['detect']} -> {out['pred']}{flip} "
              f"({out['calls']} calls, {out['latency_ms']}ms)")
        if n % SAVE_EVERY == 0:
            df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")
        time.sleep(SLEEP_SEC)
    df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")

    done = df[df["pred"].isin(["0", "1"])]
    n_err = int((df["pred"] == "err").sum())
    acc, prec, rec, f1, (tn, fp, fn, tp) = metrics(done["label"].tolist(), done["pred"].tolist())

    ti = pd.to_numeric(df["in_tok"], errors="coerce").fillna(0).sum()
    to = pd.to_numeric(df["out_tok"], errors="coerce").fillna(0).sum()
    calls = pd.to_numeric(df["calls"], errors="coerce").fillna(0).sum()
    lat = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
    ip, op = PRICE_PER_MTOK[PROVIDER]
    cost = ti / 1e6 * ip + to / 1e6 * op

    flips = df[(df["detect"] == "1") & (df["pred"] == "0")]
    good = int((flips["label"] == "0").sum())
    bad = int((flips["label"] == "1").sum())

    print("\n" + "=" * 60)
    print(f"HYBRID | วัด {len(done)} ข้อ | error {n_err}")
    print(f"  F1 {f1:.3f} | precision {prec:.3f} | recall {rec:.3f}")
    print(f"  TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    print(f"  ผู้พิพากษาปัดตก {len(flips)} ข้อ -> ถูก {good} (ฆ่า FP) / ผิด {bad} (ฆ่า TP)")
    print(f"  LLM calls {int(calls)} | token {int(ti)} in / {int(to)} out | ${cost:.3f}")
    if len(lat):
        print(f"  latency p50 {lat.median():.0f} ms")
    print(f"  เวลาเดินจริง {time.time()-t0:.0f}s")
    print("=" * 60)
    print("\nเทียบทุกระบบ (gold 127 ข้อ):")
    print("  เอเจนต์เดี่ยว        F1 0.690 | 127 calls | $0.094")
    print("  pipeline v2         F1 0.744 | 183 calls | $0.169  <- แชมป์เดิม")
    print("  debate              F1 0.694 | 381 calls | $0.695")
    print(f"  hybrid (นี่)        F1 {f1:.3f} | {int(calls)} calls | ${cost:.3f}")
    print("\nตีความ: hybrid ชนะ v2 -> 'การไตร่ตรอง' มีค่า")
    print("        hybrid เสมอ/แพ้ v2 -> 'ข้อจำกัด' ต่างหากที่สำคัญ ไม่ใช่ความลึก")


if __name__ == "__main__":
    main()
