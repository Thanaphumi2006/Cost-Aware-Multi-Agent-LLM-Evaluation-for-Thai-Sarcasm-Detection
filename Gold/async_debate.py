# -*- coding: utf-8 -*-
"""System ④-async — the same DEBATE, but prosecutor and defender speak concurrently

multiagent_debate.py runs sequentially:  prosecutor -> defender -> judge
  latency = t(prosecutor) + t(defender) + t(judge)

But **prosecutor and defender never read each other's statements** (see _argue: the user prompt is just "ข้อความ: {text}",
no opposing statement is passed in) -> the two are truly independent and can fire concurrently.
  latency = max(t(prosecutor), t(defender)) + t(judge)

**Cost is unchanged** -- same call count (3/item), same tokens; only the "waiting" changes.

Correctness: import the prompts/schema from multiagent_debate directly, no copy-paste
-> prompts can't drift; results must compare apples-to-apples with the sequential version.

CONCURRENCY controlled at two levels:
  1) within one item: asyncio.gather(prosecutor, defender)   -> reduce per-item latency
  2) across items: semaphore --concurrency items at once     -> reduce total wall-clock
level 2 is what can hit rate limits -> default 4 (conservative), raise it on a higher tier

Run:
  python async_debate.py --dry-run              see how many calls it would fire (free, no API)
  python async_debate.py --concurrency 4        run for real (needs OPENAI_API_KEY)
  python async_debate.py --limit 10 --compare   run 10 items both async and sequential and compare timing
"""
import argparse
import asyncio
import json
import os
import sys
import time

import pandas as pd

import envload  # noqa: F401  -- load OPENAI_API_KEY from .env if present (must precede imports that use the key)
from baseline import MODELS, PRICE_PER_MTOK, metrics
from multiagent_debate import (ARG_TOKENS, COLS, DEFE_SYS, JUDGE_SYS, PROS_SYS,
                               GOLD_CSV)

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
PROVIDER = "gpt"
PRED_CSV = os.path.join(HERE, f"multiagent_preds_{PROVIDER}_debate_async.csv")
METRICS_JSON = os.path.join(HERE, "async_debate_metrics.json")


def load(fresh=False):
    """Like multiagent_debate.load() but merges old results from **async's own file**

    Don't import load() from multiagent_debate directly -- that function reads *its module's* PRED_CSV
    (the sequential results, already complete for 127 items) -> todo would always be 0 and async would never run.
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
        sys.exit("OPENAI_API_KEY not found -- set it first:  $env:OPENAI_API_KEY=\"sk-...\"")
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
    """Same as multiagent_debate.run_debate except prosecutor+defender fire concurrently.
    Same err handling: any stage failure -> pred="err", not a guessed "0" """
    async with sem:
        t0 = time.perf_counter()
        ti = to = 0
        calls = 0
        pros = defe = verdict = ""
        try:
            # ---- level 1: the two sides speak concurrently (independent) ----
            t_arg = time.perf_counter()
            (pros, i1, o1), (defe, i2, o2) = await asyncio.gather(
                _argue(client, PROS_SYS, f"ข้อความ: {text}"),
                _argue(client, DEFE_SYS, f"ข้อความ: {text}"),
            )
            arg_ms = round((time.perf_counter() - t_arg) * 1000)
            ti += i1 + i2; to += o1 + o2; calls += 2

            # ---- the judge must wait for both -> still sequential ----
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
    ap.add_argument("--concurrency", type=int, default=4, help="items at once (the level that hits rate limits)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="no API -- just see how many calls it would fire")
    ap.add_argument("--fresh", action="store_true", help="don't resume -- run everything fresh")
    a = ap.parse_args()

    df = load(fresh=a.fresh)
    todo = df.index[~df["pred"].isin(["0", "1"])].tolist()
    if a.limit:
        todo = todo[:a.limit]

    if a.dry_run:
        ip, op = PRICE_PER_MTOK[PROVIDER]
        # real per-item averages from multiagent_preds_gpt_debate.csv if present
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
                est = (f"\n  per-item average (from the sequential run): {pi:.0f} in / {po:.0f} out tok"
                       f"\n  estimated cost: ${len(todo)*(pi/1e6*ip + po/1e6*op):.3f}"
                       f"\n  sequential latency p50: {lat.median():.0f} ms/item"
                       f"\n  estimated async p50: ~{lat.median()*0.67:.0f} ms/item (one of three stages off the critical path)")
        print(f"dry-run: will run {len(todo)} items x 3 calls = {len(todo)*3} calls")
        print(f"  concurrency {a.concurrency} items at once -> max in-flight {a.concurrency*2} calls{est}")
        print("\nCost is identical to sequential -- async only changes wait time, not the call count")
        return

    print(f"gold {len(df)} items | to run {len(todo)} | model {MODELS[PROVIDER]}")
    print(f"architecture: (prosecutor || defender) -> judge | concurrency {a.concurrency}\n")
    if not todo:
        print("all done -- skipping to the summary\n")
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
    print(f"DEBATE async | measured {len(done)} items | error {n_err}")
    print(f"  F1 {f1:.3f} | precision {prec:.3f} | recall {rec:.3f}")
    print(f"  TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    print(f"  LLM calls {int(calls)} | token {int(ti)} in / {int(to)} out | ${cost:.3f}")
    if len(lat):
        print(f"  latency p50 {lat.median():.0f} ms/item")
    if arg_ms:
        print(f"  argue stage (prosecutor||defender) p50 {pd.Series(arg_ms).median():.0f} ms "
              f"= max of the two, not the sum")
    if results:
        print(f"  total wall-clock {wall:.0f}s ({wall/len(results):.2f}s/item at concurrency {a.concurrency})")

    if results:
        json.dump({"n": len(results), "wall_sec": round(wall, 1),
                   "concurrency": a.concurrency, "f1": round(f1, 4),
                   "p50_latency_ms": float(lat.median()) if len(lat) else None,
                   "p50_argue_ms": float(pd.Series(arg_ms).median()) if arg_ms else None,
                   "calls": int(calls), "cost_usd": round(cost, 4)},
                  open(METRICS_JSON, "w"), ensure_ascii=False, indent=2)
        print(f"  metrics -> {os.path.basename(METRICS_JSON)}")
    print("=" * 58)
    print("compare to sequential: python multiagent_debate.py (same prompts/model)")
    print("F1 should be close -- a large difference means it's not just timing; investigate")


if __name__ == "__main__":
    main()
