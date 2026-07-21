# -*- coding: utf-8 -*-
"""
Multi-agent: detector -> verifier (filters false positives per the rubric)
Compared against single-agent baseline.py on the same harness (exactly the same metrics).

Why this design (based on the real baseline results):
  baseline GPT-4o gets recall 1.000 (catches all sarcasm) but precision only 0.526
  -> the problem is false positives (27 items called sarcastic that aren't, mostly "balanced reviews")
  -> so the verifier is a "reject stage," not a "find-more stage"

Pipeline per item:
  stage 1 detector : decide sarcastic/not (plain prompt like baseline -> preserve recall)
  stage 2 verifier : runs only on items stage 1 called "sarcastic," applies the rubric decision tree
                     if it doesn't meet the "pretense" condition -> flip to 0
  items stage 1 calls "not sarcastic" -> pass straight through, no verifier call (cheaper + can't flip back anyway)

Measures the same 4 dimensions as baseline + LLM calls per item (to answer "is it worth it?")

Run: set PROVIDER, then  python multiagent.py
"""

import json
import os
import statistics
import sys
import time

import pandas as pd

# use the same metrics + prices + model names as baseline (important: must not diverge)
from baseline import metrics, MODELS, PRICE_PER_MTOK

# ================== configurable ==================
PROVIDER = "gpt"           # "claude" or "gpt" -- must match the baseline being compared
VARIANT = "conservative"   # tag in the output filename -- change the verifier prompt, change the tag, to avoid overwriting old results
LIMIT = None
SLEEP_SEC = 0.2
SAVE_EVERY = 10
MAX_RETRY = 4
# opt-in: use a "cheaper" model as the screener (detector) while the verifier stays the main model
#   empty = use MODELS[PROVIDER] for both stages as before (old results reproduce)
#   experiment:  SCREENER_MODEL=gpt-4o-mini python multiagent.py
# cost note: PRICE_PER_MTOK is the verifier's price -- if the screener is cheaper, the real cost is "lower" than printed
SCREENER_MODEL = os.getenv("SCREENER_MODEL") or None
# =================================================

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.environ.get("EVAL_DIR", HERE)
os.makedirs(EVAL_DIR, exist_ok=True)
GOLD_CSV = os.environ.get("GOLD_CSV", os.path.join(HERE, "gold.csv"))
# if using a different screener, append a tag to the filename to avoid overwriting the original conservative results
_SCREEN_TAG = f"_screen-{SCREENER_MODEL}" if SCREENER_MODEL else ""
PRED_CSV = os.path.join(EVAL_DIR, f"multiagent_preds_{PROVIDER}_{VARIANT}{_SCREEN_TAG}.csv")
BASE_PRED = os.path.join(EVAL_DIR, f"baseline_preds_{PROVIDER}.csv")  # for comparison

# ---- stage 1: detector -- plain prompt, same as baseline (keeps recall the same) ----
# (the Thai prompt is the experimental instruction to the model and is kept as-is)
DETECT_SYS = """ตัดสินว่าข้อความภาษาไทยนี้ "ประชด/เสียดสี" หรือไม่
ประชด = เจตนาจริงตรงข้ามกับความหมายผิวเผิน เพื่อเหน็บหรือแสดงความไม่พอใจ

ตอบเป็น JSON เท่านั้น: {"label": "1" หรือ "0"}
1 = ประชด, 0 = ไม่ประชด"""

# ---- stage 2: verifier -- decision tree from labeling_rubric.md ----
# role: take an item the detector called "sarcastic" and check whether there is real "pretense"
# v2 conservative: default = keep; overturn only when it is "clearly" not sarcasm
#   (the old v1 overturned everything borderline -> lost 10 recall items; instead give the benefit of the doubt to the detector)
# (the Thai prompt is the experimental instruction to the model and is kept as-is)
VERIFY_SYS = """มีคนตัดสินว่าข้อความไทยนี้ "ประชด" มาแล้ว หน้าที่ของคุณคือ "ตรวจจับความผิดพลาดชัดๆ" เท่านั้น
ไม่ใช่ตัดสินใหม่ตั้งแต่ต้น -- คนก่อนหน้าจับประชดเก่งมาก (แทบไม่พลาด) ให้เชื่อเขาไว้ก่อน

พลิกเป็น "ไม่ประชด (0)" เฉพาะเมื่อ **มั่นใจชัดเจน** ว่าเข้าข้อใดข้อหนึ่งนี้:
  - บ่น/ตำหนิตรงๆ ล้วนๆ ไม่มีการแกล้งชมหรือแกล้งขอบคุณเลย  ["ลบตรงๆ ≠ ประชด"]
  - ชมจริงใจล้วนๆ ไม่มีนัยเหน็บ
  - รีวิวสมดุลตรงไปตรงมา: ชมจริงบางจุด ติจริงบางจุด อยู่ด้วยกันตามจริง ไม่ได้เสแสร้ง
  - แค่เล่าเหตุการณ์/อ้างคำพูดคนอื่น ไม่มีโทนเหน็บของผู้เขียนเอง

ถ้า **ไม่ชัด** ว่าเข้าข้อไหนข้างบน -- แม้จะก้ำกึ่ง หรืออ่านได้สองแง่ -- ให้ **คงไว้เป็นประชด (1)**
เหตุผล: ประชดไทยมักแนบเนียน อ่านได้สองแง่เป็นเรื่องปกติของประชด ถ้าลังเลแปลว่าน่าจะประชด

ตอบ JSON เท่านั้น: {"verdict": "1" หรือ "0"}  (1 = คงเป็นประชด, 0 = พลิกทิ้งเพราะชัดว่าไม่ใช่)"""

DETECT_SCHEMA = {"type": "object",
                 "properties": {"label": {"type": "string", "enum": ["0", "1"]}},
                 "required": ["label"], "additionalProperties": False}
VERIFY_SCHEMA = {"type": "object",
                 "properties": {"verdict": {"type": "string", "enum": ["0", "1"]}},
                 "required": ["verdict"], "additionalProperties": False}


# ---------------- LLM plumbing (accepts system + schema, unlike baseline which fixes them) ----------------

def _make_client():
    if PROVIDER not in MODELS:
        sys.exit(f"PROVIDER must be 'claude' or 'gpt', not {PROVIDER!r}")
    pkg, key = {"claude": ("anthropic", "ANTHROPIC_API_KEY"),
                "gpt": ("openai", "OPENAI_API_KEY")}[PROVIDER]
    try:
        if PROVIDER == "claude":
            import anthropic
            return anthropic.Anthropic(max_retries=MAX_RETRY)
        from openai import OpenAI
        return OpenAI(max_retries=MAX_RETRY)
    except ImportError:
        sys.exit(f"{pkg} not installed  ->  pip install {pkg}")
    except Exception as e:
        hint = f"\n(is the key set? {key})" if not os.getenv(key) else ""
        sys.exit(f"could not create client: {type(e).__name__}: {e}{hint}")


def _ask(client, system, schema, key, text, model=None):
    """Call the LLM once, return (value, in_tok, out_tok) ; value=None on failure
    model=None -> use MODELS[PROVIDER] (default) ; pass a model name to override just this call (e.g. a cheap screener)"""
    mdl = model or MODELS[PROVIDER]
    if PROVIDER == "claude":
        r = client.messages.create(
            model=mdl, max_tokens=256, system=system,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": f"ข้อความ: {text}"}],
        )
        raw = next((b.text for b in r.content if b.type == "text"), "")
        in_tok, out_tok = r.usage.input_tokens, r.usage.output_tokens
    else:
        r = client.chat.completions.create(
            model=mdl, max_tokens=20,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": f"ข้อความ: {text}"}],
        )
        raw = (r.choices[0].message.content or "").strip()
        in_tok, out_tok = r.usage.prompt_tokens, r.usage.completion_tokens
    try:
        v = str(json.loads(raw).get(key, "")).strip()
    except json.JSONDecodeError:
        return None, in_tok, out_tok
    return (v if v in ("0", "1") else None), in_tok, out_tok


def run_pipeline(client, text):
    """detector -> (if sarcastic) verifier. Return a dict with all dimensions."""
    t0 = time.perf_counter()
    in_tok = out_tok = calls = 0
    try:
        det, i, o = _ask(client, DETECT_SYS, DETECT_SCHEMA, "label", text, model=SCREENER_MODEL)
        in_tok += i; out_tok += o; calls += 1
        if det is None:
            raise ValueError("detector returned malformed output")

        verdict = ""
        if det == "1":
            # re-check only the items called sarcastic
            ver, i, o = _ask(client, VERIFY_SYS, VERIFY_SCHEMA, "verdict", text)
            in_tok += i; out_tok += o; calls += 1
            if ver is None:
                raise ValueError("verifier returned malformed output")
            verdict = ver
            final = ver          # verifier can flip: 1->keep, 0->overturn
        else:
            final = "0"          # detector says not sarcastic -> pass straight through

        return {"pred": final, "detect": det, "verdict": verdict,
                "in_tok": in_tok, "out_tok": out_tok, "calls": calls,
                "latency_ms": round((time.perf_counter() - t0) * 1000), "err": ""}
    except Exception as e:
        return {"pred": "err", "detect": "", "verdict": "",
                "in_tok": in_tok, "out_tok": out_tok, "calls": calls,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "err": f"{type(e).__name__}: {e}"[:200]}


# ---------------- load/save (resume, always starting from gold, like baseline) ----------------

COLS = ["pred", "detect", "verdict", "in_tok", "out_tok", "calls", "latency_ms", "err"]


def load():
    if not os.path.exists(GOLD_CSV):
        sys.exit(f"file not found: {GOLD_CSV}")
    g = pd.read_csv(GOLD_CSV, dtype=str).fillna("")
    g["label"] = g["label"].str.strip()
    g = g[g["label"].isin(["0", "1"])].reset_index(drop=True)
    for c in COLS:
        g[c] = ""
    if os.path.exists(PRED_CSV):
        old = pd.read_csv(PRED_CSV, dtype=str).fillna("")
        prev = {r["text"]: r for _, r in old.iterrows() if r.get("pred", "") in ("0", "1")}
        hit = 0
        for i in g.index:
            r = prev.get(g.at[i, "text"])
            if r is not None:
                for c in COLS:
                    g.at[i, c] = r.get(c, "")
                hit += 1
        print(f"resuming: reused {hit} old results")
    return g


def save(df):
    df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")


# ---------------- main ----------------

def _sum_int(series):
    return sum(int(x) for x in series if str(x).lstrip("-").isdigit())


def main():
    price = PRICE_PER_MTOK[PROVIDER]
    df = load()
    todo = [i for i in df.index if df.at[i, "pred"] not in ("0", "1")]
    if LIMIT:
        todo = todo[:LIMIT]

    print(f"\nMULTI-AGENT {PROVIDER} ({MODELS[PROVIDER]}) | gold {len(df)} | to do {len(todo)} items")
    print("pipeline: detector -> verifier (only on items called sarcastic)")
    if todo:
        client = _make_client()
        t0 = time.perf_counter()
        for n, idx in enumerate(todo, 1):
            out = run_pipeline(client, str(df.at[idx, "text"]))
            for c in COLS:
                df.at[idx, c] = str(out[c])
            if out["err"]:
                print(f"  [{n}/{len(todo)}] failed: {out['err'][:70]}")
            if n % SAVE_EVERY == 0 or n == len(todo):
                save(df); print(f"  ...{n}/{len(todo)}")
            time.sleep(SLEEP_SEC)
        print(f"total API time: {time.perf_counter() - t0:.1f}s")
    save(df)

    # ---- evaluate ----
    done = df[df["pred"].isin(["0", "1"])]
    bad = df[df["pred"] == "err"]
    acc, prec, rec, f1, (tn, fp, fn, tp) = metrics(done["label"].tolist(), done["pred"].tolist())
    in_tok, out_tok = _sum_int(df["in_tok"]), _sum_int(df["out_tok"])
    n_calls = _sum_int(df["calls"])
    lat = [int(x) for x in df["latency_ms"] if str(x).isdigit()]
    n_verified = int((df["verdict"] != "").sum())
    n_flipped = int(((df["detect"] == "1") & (df["verdict"] == "0")).sum())

    print("\n" + "=" * 58)
    print(f"MULTI-AGENT (detector->verifier) — {PROVIDER} / {MODELS[PROVIDER]}")
    print("=" * 58)
    print(f"\n[1] quality (n = {len(done)})")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(f"  F1       : {f1:.3f}")
    print("\n  Confusion matrix (row=true, col=pred)")
    print("              pred:0   pred:1")
    print(f"    true:0     {tn:>5}    {fp:>5}")
    print(f"    true:1     {fn:>5}    {tp:>5}")
    print(f"\n  verifier ran on {n_verified} items | flipped 1->0 on {n_flipped}")

    print(f"\n[2] cost")
    print(f"  LLM calls    : {n_calls}  (baseline = {len(done)} calls)")
    print(f"  input tokens : {in_tok:,}")
    print(f"  output tokens: {out_tok:,}")
    if price:
        cost = in_tok / 1e6 * price[0] + out_tok / 1e6 * price[1]
        print(f"  cost         : ${cost:.4f}")

    print(f"\n[3] time")
    if lat:
        p95 = sorted(lat)[max(0, int(len(lat) * 0.95) - 1)]
        print(f"  latency/item: p50 {statistics.median(lat):.0f} ms | p95 {p95} ms | total {sum(lat)/1000:.1f}s")

    print(f"\n[4] reliability: err {len(bad)}/{len(df)}")
    for e in bad["err"].head(3):
        print(f"     {e[:70]}")

    # ---- compare to baseline directly (if the file exists) ----
    if os.path.exists(BASE_PRED):
        b = pd.read_csv(BASE_PRED, dtype=str).fillna("")
        b = b[b["pred"].isin(["0", "1"])]
        bacc, bprec, brec, bf1, _ = metrics(b["label"].tolist(), b["pred"].tolist())
        b_in, b_out = _sum_int(b["in_tok"]), _sum_int(b["out_tok"])
        b_cost = (b_in / 1e6 * price[0] + b_out / 1e6 * price[1]) if price else 0
        print("\n" + "=" * 58)
        print("vs baseline (is it worth it?)")
        print("=" * 58)
        print(f"  {'':<12}{'baseline':>12}{'multi-agent':>14}{'diff':>10}")
        print(f"  {'F1':<12}{bf1:>12.3f}{f1:>14.3f}{f1-bf1:>+10.3f}")
        print(f"  {'precision':<12}{bprec:>12.3f}{prec:>14.3f}{prec-bprec:>+10.3f}")
        print(f"  {'recall':<12}{brec:>12.3f}{rec:>14.3f}{rec-brec:>+10.3f}")
        if price:
            print(f"  {'cost $':<12}{b_cost:>12.4f}{cost:>14.4f}{f'{cost/b_cost:.2f}x':>10}")
        print(f"  {'LLM calls':<12}{len(b):>12}{n_calls:>14}{f'{n_calls/len(b):.2f}x':>10}")
        print("\n  -> F1 must exceed 0.793 (baseline upper CI) to claim a real win, not chance")
        if f1 > 0.793:
            print(f"     F1 = {f1:.3f} > 0.793  ok, beyond the range of chance")
        else:
            print(f"     F1 = {f1:.3f} not above 0.793 -> can't yet claim it's better")

    print(f"\nsaved to: {os.path.basename(PRED_CSV)}")


if __name__ == "__main__":
    main()
