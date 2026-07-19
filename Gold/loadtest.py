# -*- coding: utf-8 -*-
"""Load test ระบบ router -> LLM: วัด latency / cost / routing ภายใต้ traffic พร้อมกันหลายเส้น

ตอบคำถามว่า "ระบบ cost-aware นี้เอาไปใช้จริงแล้วพังตรงไหน" ด้วยตัวเลข ไม่ใช่ความรู้สึก

สองโหมด:
  --mock (ดีฟอลต์)  ฟรี ไม่ยิง API เลย -- จำลอง latency จากค่าที่ *วัดมาจริง* ในโปรเจกต์นี้
                    (WCB 26ms, GPT 1 call 751ms, debate 4557ms -- ดู SOURCES ข้างล่าง)
                    ใช้ตรวจ orchestration / semaphore / เส้นทาง routing ได้ครบโดยไม่เสียเงิน
  --live            ยิง API จริง (ต้องมี OPENAI_API_KEY) -- ราคาจริงตามจำนวนข้อที่ escalate

ทำไมต้องมี mock: gold มี 127 ข้อ ยิงจริงรอบละ ~$0.13 การจูน concurrency ต้องรันหลายรอบ
-> จูนบน mock ให้จบก่อน แล้วค่อยยืนยันด้วย --live รอบเดียว

ข้อจำกัดที่ต้องพูดตรงๆ: n=127 ข้อ ไม่ใช่ traffic ระดับ production จริง
สคริปต์นี้ตอบได้แค่ "orchestration ถูกไหม / p95 เป็นเท่าไร / cost ต่อนาทีเท่าไร"
มันตอบ **ไม่ได้** ว่าระบบทน 10k req/s ไหม -- อย่าเคลมเกินนั้นในรายงาน

SOURCES (latency ที่ใช้ใน mock -- มาจากไฟล์ผลจริงในรีโป ไม่ได้เดา):
  WCB screener   ~26 ms   docstring ของ cascade.py
  GPT 1 call     ~751 ms  RESULTS.md (GPT bot p50)
  debate 3 calls ~4557 ms multiagent_preds_gpt_debate.csv (p50 latency_ms)

รัน:
  python loadtest.py --rps 5 --duration 30
  python loadtest.py --rps 5 --duration 30 --concurrency 8 --budget 0.20
  python loadtest.py --live --rps 2 --duration 20      (เสียเงินจริง)
  python loadtest.py --rps 5 --duration 30 --prom metrics.prom   (ให้ Grafana ดูด)
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

import envload  # noqa: F401  -- โหลด OPENAI_API_KEY จาก .env ถ้ามี (ใช้ตอน --live)

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_CSV = os.path.join(HERE, "gold.csv")
OOF_CSV = os.path.join(HERE, "wcb_oof_probs.csv")
OUT_JSON = os.path.join(HERE, "loadtest_result.json")
OUT_CSV = os.path.join(HERE, "loadtest_requests.csv")

# latency ที่วัดมาจริง (ms) -- ดู SOURCES ใน docstring
LAT_WCB = 26.0
LAT_GPT_CALL = 751.0
LAT_DEBATE = 4557.0
COST_PER_CALL = 391 / 1e6 * 2.50 + 7 / 1e6 * 10.0   # เท่ากับ router.py / cascade.py


def lognormal_ms(median, sigma=0.35):
    """latency จริงเบ้ขวา (หางยาว) -- lognormal สมจริงกว่า gaussian
    median คือค่ากลางที่วัดมา, sigma คุมความกระจาย (0.35 -> p95 ~ 1.8x median)"""
    return float(np.random.lognormal(np.log(median), sigma))


class Metrics:
    def __init__(self):
        self.rows = []
        self.t0 = time.time()

    def add(self, **kw):
        kw["t_rel"] = round(time.time() - self.t0, 3)
        self.rows.append(kw)

    def df(self):
        return pd.DataFrame(self.rows)


async def handle(text, prob, tau, delta, tau_gpt, gpt_prob, live, client, m, sem):
    """1 request ผ่านระบบ: WCB router ก่อน -> escalate เฉพาะที่ 'ไม่แน่ใจ'"""
    async with sem:
        t0 = time.perf_counter()
        # --- ชั้น 1: router ในเครื่อง (ฟรีเสมอ ทั้ง mock และ live) ---
        await asyncio.sleep(lognormal_ms(LAT_WCB) / 1000)
        unsure = abs(prob - tau) < delta

        calls = 0
        cost = 0.0
        err = ""
        if not unsure:
            pred = int(prob >= tau)
            route = "auto"
        else:
            route = "escalated"
            try:
                if live:
                    from multiagent import _ask, VERIFY_SYS, VERIFY_SCHEMA
                    loop = asyncio.get_running_loop()
                    v = await loop.run_in_executor(
                        None, lambda: _ask(client, VERIFY_SYS, VERIFY_SCHEMA, "verdict", text))
                    pred = int(v[0] == "1")
                    calls, cost = 1, COST_PER_CALL
                else:
                    await asyncio.sleep(lognormal_ms(LAT_GPT_CALL) / 1000)
                    pred = int(gpt_prob >= tau_gpt)
                    calls, cost = 1, COST_PER_CALL
            except Exception as e:
                pred, err = -1, f"{type(e).__name__}: {e}"[:120]

        ms = (time.perf_counter() - t0) * 1000
        m.add(route=route, pred=pred, latency_ms=round(ms, 1), calls=calls,
              cost=cost, err=err)
        return pred


async def drive(items, rps, duration, concurrency, tau_map, live, client, m):
    """ยิง request แบบ Poisson arrival (traffic จริงไม่ได้มาเป็นจังหวะสม่ำเสมอ)"""
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    end = time.time() + duration
    i = 0
    while time.time() < end:
        it = items[i % len(items)]
        tau, delta, tau_gpt = tau_map
        tasks.append(asyncio.create_task(
            handle(it["text"], it["prob"], tau, delta, tau_gpt, it["gpt_prob"],
                   live, client, m, sem)))
        i += 1
        # Poisson: ช่วงห่างเป็น exponential ไม่ใช่คงที่
        await asyncio.sleep(random.expovariate(rps) if rps > 0 else 0)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return len(tasks)


def prom_export(path, d):
    """Prometheus textfile format -- ให้ node_exporter textfile collector หรือ Pushgateway ดูดต่อ
    ชื่อ metric ตามธรรมเนียม: _seconds สำหรับเวลา, _total สำหรับ counter"""
    lines = [
        "# HELP sarcasm_requests_total จำนวน request ทั้งหมดใน load test",
        "# TYPE sarcasm_requests_total counter",
        f"sarcasm_requests_total {d['n']}",
        "# HELP sarcasm_escalated_total request ที่ router ตัดสินเองไม่ได้ ต้องส่งให้ LLM",
        "# TYPE sarcasm_escalated_total counter",
        f"sarcasm_escalated_total {d['escalated']}",
        "# HELP sarcasm_escalation_ratio สัดส่วนที่ต้องจ่ายเงิน",
        "# TYPE sarcasm_escalation_ratio gauge",
        f"sarcasm_escalation_ratio {d['escalation_ratio']:.4f}",
        "# HELP sarcasm_llm_calls_total จำนวน LLM call",
        "# TYPE sarcasm_llm_calls_total counter",
        f"sarcasm_llm_calls_total {d['calls']}",
        "# HELP sarcasm_cost_usd_total ค่าใช้จ่ายสะสม (USD)",
        "# TYPE sarcasm_cost_usd_total counter",
        f"sarcasm_cost_usd_total {d['cost_usd']:.6f}",
        "# HELP sarcasm_cost_usd_per_minute ต้นทุนต่อนาทีที่ throughput นี้",
        "# TYPE sarcasm_cost_usd_per_minute gauge",
        f"sarcasm_cost_usd_per_minute {d['cost_per_min']:.6f}",
        "# HELP sarcasm_throughput_rps request ต่อวินาทีที่ทำได้จริง",
        "# TYPE sarcasm_throughput_rps gauge",
        f"sarcasm_throughput_rps {d['throughput_rps']:.3f}",
        "# HELP sarcasm_errors_total request ที่พัง",
        "# TYPE sarcasm_errors_total counter",
        f"sarcasm_errors_total {d['errors']}",
        "# HELP sarcasm_latency_seconds latency ของ request (quantile)",
        "# TYPE sarcasm_latency_seconds summary",
    ]
    for q, key in ((0.5, "p50"), (0.95, "p95"), (0.99, "p99")):
        lines.append(f'sarcasm_latency_seconds{{quantile="{q}"}} {d["latency_ms"][key]/1000:.4f}')
    for r in ("auto", "escalated"):
        if r in d["latency_by_route"]:
            v = d["latency_by_route"][r]
            lines.append(f'sarcasm_route_latency_seconds{{route="{r}",quantile="0.5"}} {v["p50"]/1000:.4f}')
            lines.append(f'sarcasm_route_requests_total{{route="{r}"}} {v["n"]}')
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", type=float, default=5.0, help="อัตรายิงเป้าหมาย (Poisson)")
    ap.add_argument("--duration", type=float, default=30.0, help="วินาที")
    ap.add_argument("--concurrency", type=int, default=8, help="request พร้อมกันสูงสุด")
    ap.add_argument("--budget", type=float, default=0.20, help="escalation budget (ดู router.py)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--live", action="store_true", help="ยิง API จริง (เสียเงิน)")
    ap.add_argument("--prom", default=None, help="เขียน Prometheus textfile ไปที่ path นี้")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)

    if not os.path.exists(OOF_CSV):
        sys.exit(f"ไม่พบ {OOF_CSV} -- รัน wcb_oof_probs.py ก่อน")
    df = pd.read_csv(OOF_CSV, encoding="utf-8-sig")
    gpt_path = os.path.join(HERE, "frontier_probs_gpt-4.1-mini.csv")
    gpt = pd.read_csv(gpt_path, encoding="utf-8-sig")
    if len(gpt) != len(df):
        sys.exit("จำนวนแถวไม่ตรง -- join ตามตำแหน่งไม่ได้")

    probs = df[f"prob_seed{a.seed}"].values.astype(float)
    y = df["label"].astype(int).values
    gpt_prob = gpt["prob"].values.astype(float)

    # tau = 0.5 ตรงกับดีฟอลต์ของ router.py (--tau-mode fixed) -> จำลอง config ที่ใช้จริง
    # (finding 15: จูน tau ที่ n=127 แล้วแย่ลง -> โปรดักชันใช้ 0.5)
    # delta/tau_gpt ใช้ทั้งชุดได้ เพราะ load test วัด *ระบบ* ไม่ได้วัด *ความแม่น*
    # (ตัวเลข F1 ที่รายงานได้จริงมาจาก router.py ซึ่ง leave-fold-out)
    from router import best_tau
    tau = 0.5
    delta = float(np.quantile(np.abs(probs - tau), a.budget)) if a.budget > 0 else -1.0
    tau_gpt = best_tau(gpt_prob, y)

    items = [{"text": t, "prob": p, "gpt_prob": g}
             for t, p, g in zip(df["text"].astype(str), probs, gpt_prob)]

    client = None
    if a.live:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("--live ต้องมี OPENAI_API_KEY")
        import multiagent
        client = multiagent._make_client()

    mode = "LIVE (เสียเงินจริง)" if a.live else "MOCK (ฟรี)"
    print(f"load test [{mode}] | rps {a.rps} | {a.duration}s | concurrency {a.concurrency} "
          f"| budget {a.budget}")
    print(f"router: tau {tau:.3f} delta {delta:.3f} -> คาดว่า escalate ~{100*a.budget:.0f}%\n")

    m = Metrics()
    t0 = time.time()
    n = asyncio.run(drive(items, a.rps, a.duration, a.concurrency, (tau, delta, tau_gpt),
                          a.live, client, m))
    wall = time.time() - t0

    r = m.df()
    if r.empty:
        sys.exit("ไม่มี request สำเร็จเลย")
    r.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lat = r["latency_ms"]
    esc = int((r["route"] == "escalated").sum())
    cost = float(r["cost"].sum())
    d = {
        "mode": "live" if a.live else "mock",
        "n": int(len(r)), "wall_sec": round(wall, 2),
        "target_rps": a.rps, "throughput_rps": len(r) / wall,
        "concurrency": a.concurrency, "budget": a.budget,
        "escalated": esc, "escalation_ratio": esc / len(r),
        "calls": int(r["calls"].sum()), "cost_usd": cost,
        "cost_per_min": cost / wall * 60,
        "errors": int((r["err"] != "").sum()),
        "latency_ms": {"p50": float(lat.quantile(0.5)), "p95": float(lat.quantile(0.95)),
                       "p99": float(lat.quantile(0.99)), "max": float(lat.max())},
        "latency_by_route": {
            k: {"n": int(len(g)), "p50": float(g["latency_ms"].quantile(0.5)),
                "p95": float(g["latency_ms"].quantile(0.95))}
            for k, g in r.groupby("route")},
    }
    json.dump(d, open(OUT_JSON, "w"), ensure_ascii=False, indent=2)

    print("=" * 66)
    print(f"request {d['n']} ข้อ ใน {d['wall_sec']}s -> throughput {d['throughput_rps']:.2f} rps "
          f"(เป้า {a.rps})")
    print(f"escalate {esc}/{d['n']} = {100*d['escalation_ratio']:.1f}%  (ตั้ง budget ไว้ {100*a.budget:.0f}%)")
    print(f"latency  p50 {d['latency_ms']['p50']:.0f} ms | p95 {d['latency_ms']['p95']:.0f} ms "
          f"| p99 {d['latency_ms']['p99']:.0f} ms")
    for k, v in d["latency_by_route"].items():
        tag = "ฟรี ไม่ยิง API" if k == "auto" else "ยิง LLM"
        print(f"  {k:>10} n={v['n']:>4}  p50 {v['p50']:>7.0f} ms  p95 {v['p95']:>7.0f} ms   ({tag})")
    print(f"cost     ${cost:.4f} รวม | ${d['cost_per_min']:.4f}/นาที | {d['calls']} calls")
    print(f"error    {d['errors']}")
    print("=" * 66)
    print(f"บันทึก -> {os.path.basename(OUT_JSON)} , {os.path.basename(OUT_CSV)}")
    if a.prom:
        prom_export(a.prom, d)
        print(f"Prometheus -> {a.prom}  (ชี้ node_exporter --collector.textfile.directory มาที่โฟลเดอร์นี้)")
    if not a.live:
        print("\nหมายเหตุ: นี่คือ MOCK -- latency จำลองจากค่าที่วัดจริง ไม่ได้ยิง API")
        print("ยืนยันด้วยของจริงรอบเดียวได้ที่:  python loadtest.py --live --rps 2 --duration 20")


if __name__ == "__main__":
    main()
