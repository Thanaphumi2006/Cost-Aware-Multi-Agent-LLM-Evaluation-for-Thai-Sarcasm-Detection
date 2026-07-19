# -*- coding: utf-8 -*-
"""ระบบ ④-async — DEBATE เดิม แต่ให้อัยการกับทนายพูดพร้อมกัน

multiagent_debate.py รันเรียงกัน:  อัยการ -> ทนาย -> ผู้พิพากษา
  latency = t(อัยการ) + t(ทนาย) + t(ผู้พิพากษา)

แต่ **อัยการกับทนายไม่ได้อ่านคำแถลงของกันและกัน** (ดู _argue: user prompt คือ "ข้อความ: {text}"
เท่านั้น ไม่มีการส่งคำแถลงฝ่ายตรงข้ามเข้าไป) -> สองตัวนี้เป็นอิสระต่อกันจริง ยิงพร้อมกันได้
  latency = max(t(อัยการ), t(ทนาย)) + t(ผู้พิพากษา)

**ราคาไม่เปลี่ยนเลย** -- จำนวน call เท่าเดิม (3/ข้อ) token เท่าเดิม เปลี่ยนแค่ "รอ" เท่านั้น

ความถูกต้อง: import prompt/schema จาก multiagent_debate โดยตรง ไม่ก๊อปมาวาง
-> prompt ดริฟต์ไม่ได้ ผลลัพธ์ต้องเทียบกับตัว sequential ได้แบบ apples-to-apples

CONCURRENCY คุมสองชั้น:
  1) ในข้อเดียว: asyncio.gather(อัยการ, ทนาย)          -> ลด latency ต่อข้อ
  2) ข้ามข้อ: semaphore --concurrency ข้อพร้อมกัน       -> ลด wall-clock รวม
ชั้นที่ 2 คือตัวที่ชน rate limit ได้ -> ดีฟอลต์ 4 (อนุรักษ์นิยม) ปรับขึ้นได้ถ้า tier สูง

รัน:
  python async_debate.py --dry-run              ดูว่าจะยิงกี่ call (ฟรี ไม่ยิง API)
  python async_debate.py --concurrency 4        รันจริง (ต้องมี OPENAI_API_KEY)
  python async_debate.py --limit 10 --compare   รัน 10 ข้อทั้ง async และ sequential แล้วเทียบเวลา
"""
import argparse
import asyncio
import json
import os
import sys
import time

import pandas as pd

import envload  # noqa: F401  -- โหลด OPENAI_API_KEY จาก .env ถ้ามี (ต้องมาก่อน import ที่ใช้คีย์)
from baseline import MODELS, PRICE_PER_MTOK, metrics
from multiagent_debate import (ARG_TOKENS, COLS, DEFE_SYS, JUDGE_SYS, PROS_SYS,
                               GOLD_CSV)

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
PROVIDER = "gpt"
PRED_CSV = os.path.join(HERE, f"multiagent_preds_{PROVIDER}_debate_async.csv")
METRICS_JSON = os.path.join(HERE, "async_debate_metrics.json")


def load(fresh=False):
    """เหมือน multiagent_debate.load() แต่ merge ผลเก่าจาก **ไฟล์ของ async เอง**

    ห้าม import load() จาก multiagent_debate มาใช้ตรงๆ -- ฟังก์ชันนั้นอ่าน PRED_CSV ของ *module มัน*
    (ผล sequential ที่รันครบ 127 ข้อแล้ว) -> todo จะเป็น 0 เสมอ แล้ว async จะไม่มีวันได้รัน
    """
    g = pd.read_csv(GOLD_CSV, dtype=str).fillna("")
    g["label"] = g["label"].str.strip()
    g = g[g["label"].isin(["0", "1"])].reset_index(drop=True)
    for c in COLS + ["arg_ms"]:
        g[c] = ""
    if os.path.exists(PRED_CSV) and not fresh:
        old = pd.read_csv(PRED_CSV, dtype=str).fillna("")
        m = old.set_index("text")
        for i, t in enumerate(g["text"]):
            if t in m.index:
                for c in COLS:
                    if c in old.columns:
                        g.at[i, c] = m.at[t, c]
    return g


def _make_async_client():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ไม่พบ OPENAI_API_KEY -- ตั้งก่อน:  $env:OPENAI_API_KEY=\"sk-...\"")
    from openai import AsyncOpenAI
    return AsyncOpenAI()


async def _argue(client, system, user):
    r = await client.chat.completions.create(
        model=MODELS[PROVIDER], max_tokens=ARG_TOKENS,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return ((r.choices[0].message.content or "").strip(),
            r.usage.prompt_tokens, r.usage.completion_tokens)


async def _judge(client, text, pros, defe):
    user = (f"ข้อความต้นฉบับ:\n{text}\n\n"
            f"คำแถลงอัยการ (ฝ่ายว่าประชด):\n{pros}\n\n"
            f"คำแถลงทนาย (ฝ่ายว่าไม่ประชด):\n{defe}\n\n"
            f"ตัดสิน:")
    r = await client.chat.completions.create(
        model=MODELS[PROVIDER], max_tokens=20,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE_SYS},
                  {"role": "user", "content": user}],
    )
    raw = (r.choices[0].message.content or "").strip()
    v = str(json.loads(raw).get("verdict", "")).strip()
    return v, r.usage.prompt_tokens, r.usage.completion_tokens


async def run_debate(client, text, sem):
    """เหมือน multiagent_debate.run_debate ทุกอย่าง ยกเว้นอัยการ+ทนายยิงพร้อมกัน
    err handling ตรงกัน: พังที่ด่านไหน -> pred="err" ไม่เดา "0" """
    async with sem:
        t0 = time.perf_counter()
        ti = to = 0
        calls = 0
        pros = defe = verdict = ""
        try:
            # ---- ชั้นที่ 1: สองฝ่ายพูดพร้อมกัน (อิสระต่อกัน) ----
            t_arg = time.perf_counter()
            (pros, i1, o1), (defe, i2, o2) = await asyncio.gather(
                _argue(client, PROS_SYS, f"ข้อความ: {text}"),
                _argue(client, DEFE_SYS, f"ข้อความ: {text}"),
            )
            arg_ms = round((time.perf_counter() - t_arg) * 1000)
            ti += i1 + i2; to += o1 + o2; calls += 2

            # ---- ผู้พิพากษาต้องรอทั้งคู่ -> ยังเรียงกันตามเดิม ----
            verdict, i, o = await _judge(client, text, pros, defe)
            ti += i; to += o; calls += 1
        except Exception as e:
            return {"pred": "err", "pros": pros[:300], "defe": defe[:300], "judge": "",
                    "in_tok": ti, "out_tok": to, "calls": calls,
                    "latency_ms": round((time.perf_counter() - t0) * 1000),
                    "arg_ms": 0, "err": f"{type(e).__name__}: {e}"[:200]}

        pred = verdict if verdict in ("0", "1") else "err"
        return {"pred": pred, "pros": pros[:300], "defe": defe[:300], "judge": verdict,
                "in_tok": ti, "out_tok": to, "calls": calls,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "arg_ms": arg_ms,
                "err": "" if pred != "err" else f"bad verdict: {verdict[:40]}"}


async def run_all(df, todo, concurrency):
    client = _make_async_client()
    sem = asyncio.Semaphore(concurrency)
    done_n = 0

    async def one(idx):
        nonlocal done_n
        out = await run_debate(client, str(df.at[idx, "text"]), sem)
        done_n += 1
        flag = "!" if out["pred"] == "err" else " "
        print(f"{flag}[{done_n}/{len(todo)}] pred={out['pred']} "
              f"({out['calls']} calls, {out['latency_ms']}ms, args {out['arg_ms']}ms)")
        return idx, out

    return await asyncio.gather(*(one(i) for i in todo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=4, help="กี่ข้อพร้อมกัน (ชั้นที่ชน rate limit)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="ไม่ยิง API -- แค่ดูว่าจะยิงกี่ call")
    ap.add_argument("--fresh", action="store_true", help="ไม่ต่อจากผลเก่า -- รันใหม่ทั้งหมด")
    a = ap.parse_args()

    df = load(fresh=a.fresh)
    todo = df.index[~df["pred"].isin(["0", "1"])].tolist()
    if a.limit:
        todo = todo[:a.limit]

    if a.dry_run:
        ip, op = PRICE_PER_MTOK[PROVIDER]
        # ค่าเฉลี่ยจริงต่อข้อจาก multiagent_preds_gpt_debate.csv ถ้ามี
        est = ""
        old = os.path.join(HERE, f"multiagent_preds_{PROVIDER}_debate.csv")
        if os.path.exists(old):
            o = pd.read_csv(old, dtype=str).fillna("")
            ti = pd.to_numeric(o["in_tok"], errors="coerce").fillna(0)
            to = pd.to_numeric(o["out_tok"], errors="coerce").fillna(0)
            lat = pd.to_numeric(o["latency_ms"], errors="coerce").dropna()
            n_ok = int((ti > 0).sum())
            if n_ok:
                pi, po = ti.sum() / n_ok, to.sum() / n_ok
                est = (f"\n  ต่อข้อเฉลี่ย (จากรัน sequential เดิม): {pi:.0f} in / {po:.0f} out tok"
                       f"\n  คาดราคา: ${len(todo)*(pi/1e6*ip + po/1e6*op):.3f}"
                       f"\n  latency sequential p50: {lat.median():.0f} ms/ข้อ"
                       f"\n  คาด async p50: ~{lat.median()*0.67:.0f} ms/ข้อ (ตัด 1 ใน 3 ด่านออกจากเส้นทางวิกฤต)")
        print(f"dry-run: จะรัน {len(todo)} ข้อ x 3 calls = {len(todo)*3} calls")
        print(f"  concurrency {a.concurrency} ข้อพร้อมกัน -> in-flight สูงสุด {a.concurrency*2} calls{est}")
        print("\nราคาไม่ต่างจาก sequential เลย -- async เปลี่ยนแค่เวลารอ ไม่ได้เปลี่ยนจำนวน call")
        return

    print(f"gold {len(df)} ข้อ | ต้องรัน {len(todo)} ข้อ | โมเดล {MODELS[PROVIDER]}")
    print(f"สถาปัตยกรรม: (อัยการ || ทนาย) -> ผู้พิพากษา | concurrency {a.concurrency}\n")
    if not todo:
        print("ครบแล้ว -- ข้ามไปสรุปผล\n")
        results = []
    else:
        t0 = time.time()
        results = asyncio.run(run_all(df, todo, a.concurrency))
        wall = time.time() - t0

    arg_ms = []
    for idx, out in results:
        for c in COLS:
            df.at[idx, c] = str(out[c])
        if out["arg_ms"]:
            arg_ms.append(out["arg_ms"])
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
    print(f"DEBATE async | วัด {len(done)} ข้อ | error {n_err}")
    print(f"  F1 {f1:.3f} | precision {prec:.3f} | recall {rec:.3f}")
    print(f"  TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    print(f"  LLM calls {int(calls)} | token {int(ti)} in / {int(to)} out | ${cost:.3f}")
    if len(lat):
        print(f"  latency p50 {lat.median():.0f} ms/ข้อ")
    if arg_ms:
        print(f"  ขั้นโต้แย้ง (อัยการ||ทนาย) p50 {pd.Series(arg_ms).median():.0f} ms "
              f"= max ของสองฝ่าย ไม่ใช่ผลรวม")
    if results:
        print(f"  wall-clock รวม {wall:.0f}s ({wall/len(results):.2f}s/ข้อ ที่ concurrency {a.concurrency})")

    if results:
        json.dump({"n": len(results), "wall_sec": round(wall, 1),
                   "concurrency": a.concurrency, "f1": round(f1, 4),
                   "p50_latency_ms": float(lat.median()) if len(lat) else None,
                   "p50_argue_ms": float(pd.Series(arg_ms).median()) if arg_ms else None,
                   "calls": int(calls), "cost_usd": round(cost, 4)},
                  open(METRICS_JSON, "w"), ensure_ascii=False, indent=2)
        print(f"  metrics -> {os.path.basename(METRICS_JSON)}")
    print("=" * 58)
    print("เทียบกับ sequential: python multiagent_debate.py (prompt/โมเดลชุดเดียวกัน)")
    print("F1 ควรใกล้เคียงกัน -- ถ้าต่างมาก แปลว่าไม่ใช่แค่เรื่องเวลา ต้องไปหาสาเหตุ")


if __name__ == "__main__":
    main()
