# -*- coding: utf-8 -*-
"""ระบบ ④ -- Multi-agent แบบ DEBATE (สถาปัตยกรรมคนละแบบกับ pipeline)

pipeline (v2) : detector -> verifier   [verifier พลิกได้ทางเดียว 1->0 -> ซื้อได้แค่ precision]
debate  (v3)  : อัยการ + ทนาย -> ผู้พิพากษา  [ตัดสินใหม่ได้ทั้งสองทาง -> recall ขึ้นก็ได้ ลงก็ได้]

ทำไมต้องรันทุกข้อ (ไม่ใช่แค่ข้อที่ detector ชี้):
  ถ้ารันแค่ข้อที่ detector ชี้ = มันคือ "verifier ที่แพงขึ้น" ไม่ใช่สถาปัตยกรรมใหม่
  จะเทียบ paradigm กันไม่ได้จริง -> ต้องปล่อยให้ debate ตัดสินตั้งแต่ต้นเองทั้ง 127 ข้อ

agents:
  1) อัยการ (prosecution) -- หาเหตุผลที่ดีที่สุดว่า "นี่คือประชด"
  2) ทนาย   (defense)     -- หาเหตุผลที่ดีที่สุดว่า "นี่ไม่ใช่ประชด"
     * ทั้งคู่ถูกบังคับให้เถียงข้างตัวเอง (adversarial) ไม่ใช่ให้ตัดสิน
  3) ผู้พิพากษา (judge)    -- อ่านข้อความ + คำแถลงทั้งสองฝ่าย + rubric แล้วตัดสิน

หมายเหตุความเป็นธรรม: ผู้พิพากษาได้ rubric ชุดเดียวกับที่ verifier ของ v2 ได้
-> ไม่ได้เปรียบเรื่องข้อมูล ต่างกันแค่ "โครงสร้างการตัดสินใจ"

รัน: python multiagent_debate.py
"""
import json
import os
import sys
import time

import pandas as pd

from baseline import MODELS, PRICE_PER_MTOK, metrics  # harness เดียวกับทุกระบบ

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_CSV = os.path.join(HERE, "gold.csv")

PROVIDER = "gpt"
VARIANT = "debate"
PRED_CSV = os.path.join(HERE, f"multiagent_preds_{PROVIDER}_{VARIANT}.csv")

LIMIT = None
SAVE_EVERY = 10
SLEEP_SEC = 0.2
ARG_TOKENS = 120          # จำกัดความยาวคำแถลง -- กันค่าใช้จ่ายบานปลาย

COLS = ["pred", "pros", "defe", "judge", "in_tok", "out_tok", "calls", "latency_ms", "err"]

# ---------- agent 1: อัยการ ----------
PROS_SYS = """คุณคือ "อัยการ" ในการโต้แย้งเรื่องข้อความภาษาไทย
หน้าที่: หาเหตุผลที่ "ดีที่สุด" ที่จะบอกว่าข้อความนี้ **เป็นประชด/เสียดสี**
ประชด = เจตนาจริงตรงข้ามกับความหมายผิวเผิน โดยมี "การเสแสร้ง" (แกล้งชม/แกล้งขอบคุณ) เพื่อเหน็บ

ชี้หลักฐานในข้อความให้ชัด (คำที่แกล้งชม, ความขัดแย้งระหว่างคำชมกับเนื้อหาจริง, น้ำเสียงเกินจริง)
ถ้าหลักฐานอ่อน ให้ยอมรับตรงๆ ว่าอ่อน -- ห้ามแต่งหลักฐานที่ไม่มีในข้อความ
เขียนสั้นๆ ไม่เกิน 3 ประโยค ตอบเป็นข้อความธรรมดา"""

# ---------- agent 2: ทนาย ----------
DEFE_SYS = """คุณคือ "ทนายจำเลย" ในการโต้แย้งเรื่องข้อความภาษาไทย
หน้าที่: หาเหตุผลที่ "ดีที่สุด" ที่จะบอกว่าข้อความนี้ **ไม่ใช่ประชด**

เหตุผลที่ใช้ได้ (ถ้าตรงกับข้อความจริง):
  - บ่น/ตำหนิ "ตรงๆ" ล้วนๆ ไม่มีการแกล้งชมเลย  -> ลบตรงๆ != ประชด
  - ชมจริงใจล้วนๆ ไม่มีนัยเหน็บ
  - รีวิวสมดุล: ชมจริงบางจุด ติจริงบางจุด ตามความเป็นจริง ไม่ได้เสแสร้ง
  - แค่เล่าเหตุการณ์/อ้างคำพูดคนอื่น ไม่มีโทนเหน็บของผู้เขียนเอง

ถ้าข้อความมีการเสแสร้งชัดเจนจนแก้ต่างไม่ขึ้น ให้ยอมรับตรงๆ -- ห้ามบิดเบือนข้อความ
เขียนสั้นๆ ไม่เกิน 3 ประโยค ตอบเป็นข้อความธรรมดา"""

# ---------- agent 3: ผู้พิพากษา ----------
JUDGE_SYS = """คุณคือ "ผู้พิพากษา" ตัดสินว่าข้อความภาษาไทยนี้เป็นประชด/เสียดสีหรือไม่
คุณจะได้: ข้อความต้นฉบับ + คำแถลงของอัยการ + คำแถลงของทนาย

กฎการตัดสิน (สำคัญ):
  ประชด (1) = ต้องมี "การเสแสร้ง" -- แกล้งชม/แกล้งขอบคุณ เพื่อเหน็บ
  ไม่ประชด (0) = บ่นตรงๆ ล้วน / ชมจริงใจล้วน / รีวิวสมดุลตามจริง / แค่เล่าเรื่อง

อย่าเชื่อฝ่ายใดเพียงเพราะเขาเขียนน่าเชื่อ -- ให้กลับไปดู "ข้อความต้นฉบับ" เป็นหลัก
ถ้าฝ่ายไหนอ้างหลักฐานที่ไม่มีอยู่จริงในข้อความ ให้ตัดทิ้ง
ถ้าก้ำกึ่งจริงๆ: ประชดไทยมักแนบเนียนและอ่านได้สองแง่ -> เอนไปทาง "ประชด (1)"

ตอบ JSON เท่านั้น: {"verdict": "1" หรือ "0"}"""

JUDGE_SCHEMA = {"type": "object",
                "properties": {"verdict": {"type": "string", "enum": ["0", "1"]}},
                "required": ["verdict"], "additionalProperties": False}


def _make_client():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ไม่พบ OPENAI_API_KEY -- ตั้งก่อน:  $env:OPENAI_API_KEY=\"sk-...\"")
    from openai import OpenAI
    return OpenAI()


def _argue(client, system, user):
    """ให้ agent เขียนคำแถลง (ข้อความธรรมดา ไม่ใช่ JSON)"""
    r = client.chat.completions.create(
        model=MODELS[PROVIDER], max_tokens=ARG_TOKENS,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return ((r.choices[0].message.content or "").strip(),
            r.usage.prompt_tokens, r.usage.completion_tokens)


def _judge(client, text, pros, defe):
    user = (f"ข้อความต้นฉบับ:\n{text}\n\n"
            f"คำแถลงอัยการ (ฝ่ายว่าประชด):\n{pros}\n\n"
            f"คำแถลงทนาย (ฝ่ายว่าไม่ประชด):\n{defe}\n\n"
            f"ตัดสิน:")
    r = client.chat.completions.create(
        model=MODELS[PROVIDER], max_tokens=20,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE_SYS},
                  {"role": "user", "content": user}],
    )
    raw = (r.choices[0].message.content or "").strip()
    v = str(json.loads(raw).get("verdict", "")).strip()
    return v, r.usage.prompt_tokens, r.usage.completion_tokens


def run_debate(client, text):
    """คืน dict. ถ้าพังที่ด่านไหน -> pred="err" ไม่เดา "0" (เดา 0 ถูกฟรี 76% -> ตัวเลขสวยปลอม)"""
    t0 = time.perf_counter()
    ti = to = 0
    calls = 0
    pros = defe = verdict = ""
    try:
        pros, i, o = _argue(client, PROS_SYS, f"ข้อความ: {text}")
        ti += i; to += o; calls += 1

        defe, i, o = _argue(client, DEFE_SYS, f"ข้อความ: {text}")
        ti += i; to += o; calls += 1

        verdict, i, o = _judge(client, text, pros, defe)
        ti += i; to += o; calls += 1
    except Exception as e:
        return {"pred": "err", "pros": pros[:300], "defe": defe[:300], "judge": "",
                "in_tok": ti, "out_tok": to, "calls": calls,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "err": f"{type(e).__name__}: {e}"[:200]}

    pred = verdict if verdict in ("0", "1") else "err"
    return {"pred": pred, "pros": pros[:300], "defe": defe[:300], "judge": verdict,
            "in_tok": ti, "out_tok": to, "calls": calls,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "err": "" if pred != "err" else f"bad verdict: {verdict[:40]}"}


def load():
    """เริ่มจาก gold เสมอ แล้ว merge ผลเก่ากลับเข้ามา (resume ได้ และไม่ตกข้อที่เพิ่งเพิ่มใน gold)"""
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
    print("สถาปัตยกรรม: อัยการ + ทนาย -> ผู้พิพากษา  (3 calls/ข้อ)\n")
    if not todo:
        print("ครบแล้ว -- ข้ามไปสรุปผล\n")

    client = _make_client() if todo else None
    t0 = time.time()
    for n, idx in enumerate(todo, 1):
        out = run_debate(client, str(df.at[idx, "text"]))
        for c in COLS:
            df.at[idx, c] = str(out[c])   # pandas 3 เข้ม dtype: คอลัมน์เป็น str จะยัด int ไม่ได้
        flag = "!" if out["pred"] == "err" else " "
        print(f"{flag}[{n}/{len(todo)}] pred={out['pred']} "
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

    print("\n" + "=" * 58)
    print(f"DEBATE (อัยการ+ทนาย+ผู้พิพากษา) | วัด {len(done)} ข้อ | error {n_err}")
    print(f"  F1 {f1:.3f} | precision {prec:.3f} | recall {rec:.3f}")
    print(f"  TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    print(f"  LLM calls {int(calls)} | token {int(ti)} in / {int(to)} out | ${cost:.3f}")
    if len(lat):
        print(f"  latency p50 {lat.median():.0f} ms | รวม {lat.sum()/1000:.0f} s")
    print(f"  เวลาเดินจริง {time.time()-t0:.0f}s")
    print("=" * 58)
    print("\nเทียบ (gold ชุดเดียวกัน 127 ข้อ):")
    print("  baseline เดี่ยว      F1 0.690 | 127 calls | $0.094")
    print("  pipeline v2         F1 0.744 | 183 calls | $0.169")
    print(f"  debate (นี่)        F1 {f1:.3f} | {int(calls)} calls | ${cost:.3f}")
    print("\nรัน compare_systems.py เพื่อทดสอบนัยสำคัญแบบ paired")


if __name__ == "__main__":
    main()
