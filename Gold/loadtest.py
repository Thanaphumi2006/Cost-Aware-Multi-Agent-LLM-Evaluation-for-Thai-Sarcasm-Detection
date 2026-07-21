# -*- coding: utf-8 -*-
"""Load test the router -> LLM system: measure latency / cost / routing under concurrent multi-lane traffic

Answers "where does this cost-aware system break in real use" with numbers, not feelings

Two modes:
  --mock (default)  free, never calls the API -- simulates latency from values *actually measured* in this project
                    (WCB 26ms, GPT 1 call 751ms, debate 4557ms -- see SOURCES below)
                    lets you check orchestration / semaphore / routing paths fully without spending money
  --live            calls the real API (needs OPENAI_API_KEY) -- real cost per number of escalated items

Why mock exists: gold has 127 items, each live run ~$0.13, tuning concurrency needs many runs
-> tune on mock first, then confirm with a single --live run

A limitation to state plainly: n=127 items is not real production-level traffic
This script can only answer "is the orchestration correct / what is p95 / what is cost per minute"
It **cannot** answer whether the system withstands 10k req/s -- don't claim beyond that in the report

SOURCES (latency used in mock -- from real result files in the repo, not guessed):
  WCB screener   ~26 ms   cascade.py docstring
  GPT 1 call     ~751 ms  RESULTS.md (GPT bot p50)
  debate 3 calls ~4557 ms multiagent_preds_gpt_debate.csv (p50 latency_ms)

Run:
  python loadtest.py --rps 5 --duration 30
  python loadtest.py --rps 5 --duration 30 --concurrency 8 --budget 0.20
  python loadtest.py --live --rps 2 --duration 20      (costs real money)
  python loadtest.py --rps 5 --duration 30 --prom metrics.prom   (for Grafana to scrape)
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

import envload  # noqa: F401  -- load OPENAI_API_KEY from .env if present (used with --live)

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_CSV = os.path.join(HERE, "gold.csv")
OOF_CSV = os.path.join(HERE, "wcb_oof_probs.csv")
OUT_JSON = os.path.join(HERE, "loadtest_result.json")
OUT_CSV = os.path.join(HERE, "loadtest_requests.csv")

# actually-measured latency (ms) -- see SOURCES in the docstring
LAT_WCB = 26.0
LAT_GPT_CALL = 751.0
LAT_DEBATE = 4557.0
COST_PER_CALL = 391 / 1e6 * 2.50 + 7 / 1e6 * 10.0   # same as router.py / cascade.py


def lognormal_ms(median, sigma=0.35):
    """real latency is right-skewed (long tail) -- lognormal is more realistic than gaussian
    median is the measured center, sigma controls the spread (0.35 -> p95 ~ 1.8x median)"""
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
    """1 request through the system: WCB router first -> escalate only the 'uncertain' ones"""
    async with sem:
        t0 = time.perf_counter()
        # --- layer 1: local router (always free, both mock and live) ---
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
    """fire requests with Poisson arrival (real traffic doesn't come at a steady rhythm)"""
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
        # Poisson: inter-arrival is exponential, not constant
        await asyncio.sleep(random.expovariate(rps) if rps > 0 else 0)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return len(tasks)


def prom_export(path, d):
    """Prometheus textfile format -- for the node_exporter textfile collector or Pushgateway to scrape
    metric names follow convention: _seconds for time, _total for counters"""
    lines = [
        "# HELP sarcasm_requests_total total requests in the load test",
        "# TYPE sarcasm_requests_total counter",
        f"sarcasm_requests_total {d['n']}",
        "# HELP sarcasm_escalated_total requests the router could not decide, sent to the LLM",
        "# TYPE sarcasm_escalated_total counter",
        f"sarcasm_escalated_total {d['escalated']}",
        "# HELP sarcasm_escalation_ratio the fraction that costs money",
        "# TYPE sarcasm_escalation_ratio gauge",
        f"sarcasm_escalation_ratio {d['escalation_ratio']:.4f}",
        "# HELP sarcasm_llm_calls_total number of LLM calls",
        "# TYPE sarcasm_llm_calls_total counter",
        f"sarcasm_llm_calls_total {d['calls']}",
        "# HELP sarcasm_cost_usd_total cumulative cost (USD)",
        "# TYPE sarcasm_cost_usd_total counter",
        f"sarcasm_cost_usd_total {d['cost_usd']:.6f}",
        "# HELP sarcasm_cost_usd_per_minute cost per minute at this throughput",
        "# TYPE sarcasm_cost_usd_per_minute gauge",
        f"sarcasm_cost_usd_per_minute {d['cost_per_min']:.6f}",
        "# HELP sarcasm_throughput_rps requests per second actually achieved",
        "# TYPE sarcasm_throughput_rps gauge",
        f"sarcasm_throughput_rps {d['throughput_rps']:.3f}",
        "# HELP sarcasm_errors_total requests that failed",
        "# TYPE sarcasm_errors_total counter",
        f"sarcasm_errors_total {d['errors']}",
        "# HELP sarcasm_latency_seconds request latency (quantile)",
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
    ap.add_argument("--rps", type=float, default=5.0, help="target fire rate (Poisson)")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds")
    ap.add_argument("--concurrency", type=int, default=8, help="max concurrent requests")
    ap.add_argument("--budget", type=float, default=0.20, help="escalation budget (see router.py)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--live", action="store_true", help="call the real API (costs money)")
    ap.add_argument("--prom", default=None, help="write Prometheus textfile to this path")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)

    if not os.path.exists(OOF_CSV):
        sys.exit(f"{OOF_CSV} not found -- run wcb_oof_probs.py first")
    df = pd.read_csv(OOF_CSV, encoding="utf-8-sig")
    gpt_path = os.path.join(HERE, "frontier_probs_gpt-4.1-mini.csv")
    gpt = pd.read_csv(gpt_path, encoding="utf-8-sig")
    if len(gpt) != len(df):
        sys.exit("row counts do not match -- cannot join by position")

    probs = df[f"prob_seed{a.seed}"].values.astype(float)
    y = df["label"].astype(int).values
    gpt_prob = gpt["prob"].values.astype(float)

    # tau = 0.5 matches router.py default (--tau-mode fixed) -> simulates the config actually used
    # (finding 15: tuning tau at n=127 made it worse -> production uses 0.5)
    # delta/tau_gpt can use the whole set, because the load test measures the *system*, not *accuracy*
    # (the real reportable F1 comes from router.py, which is leave-fold-out)
    from router import best_tau
    tau = 0.5
    delta = float(np.quantile(np.abs(probs - tau), a.budget)) if a.budget > 0 else -1.0
    tau_gpt = best_tau(gpt_prob, y)

    items = [{"text": t, "prob": p, "gpt_prob": g}
             for t, p, g in zip(df["text"].astype(str), probs, gpt_prob)]

    client = None
    if a.live:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("--live needs OPENAI_API_KEY")
        import multiagent
        client = multiagent._make_client()

    mode = "LIVE (real money)" if a.live else "MOCK (free)"
    print(f"load test [{mode}] | rps {a.rps} | {a.duration}s | concurrency {a.concurrency} "
          f"| budget {a.budget}")
    print(f"router: tau {tau:.3f} delta {delta:.3f} -> expect escalate ~{100*a.budget:.0f}%\n")

    m = Metrics()
    t0 = time.time()
    n = asyncio.run(drive(items, a.rps, a.duration, a.concurrency, (tau, delta, tau_gpt),
                          a.live, client, m))
    wall = time.time() - t0

    r = m.df()
    if r.empty:
        sys.exit("no request succeeded")
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
    print(f"request {d['n']} items in {d['wall_sec']}s -> throughput {d['throughput_rps']:.2f} rps "
          f"(target {a.rps})")
    print(f"escalate {esc}/{d['n']} = {100*d['escalation_ratio']:.1f}%  (budget set at {100*a.budget:.0f}%)")
    print(f"latency  p50 {d['latency_ms']['p50']:.0f} ms | p95 {d['latency_ms']['p95']:.0f} ms "
          f"| p99 {d['latency_ms']['p99']:.0f} ms")
    for k, v in d["latency_by_route"].items():
        tag = "free, no API" if k == "auto" else "LLM call"
        print(f"  {k:>10} n={v['n']:>4}  p50 {v['p50']:>7.0f} ms  p95 {v['p95']:>7.0f} ms   ({tag})")
    print(f"cost     ${cost:.4f} total | ${d['cost_per_min']:.4f}/min | {d['calls']} calls")
    print(f"error    {d['errors']}")
    print("=" * 66)
    print(f"saved -> {os.path.basename(OUT_JSON)} , {os.path.basename(OUT_CSV)}")
    if a.prom:
        prom_export(a.prom, d)
        print(f"Prometheus -> {a.prom}  (point node_exporter --collector.textfile.directory at this folder)")
    if not a.live:
        print("\nNote: this is MOCK -- latency simulated from measured values, no API calls")
        print("confirm with a single real run:  python loadtest.py --live --rps 2 --duration 20")


if __name__ == "__main__":
    main()
