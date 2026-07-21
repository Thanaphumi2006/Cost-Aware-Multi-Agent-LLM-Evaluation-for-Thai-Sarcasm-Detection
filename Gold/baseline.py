# -*- coding: utf-8 -*-
"""
Baseline: single-agent zero-shot LLM classifier -> measure 4 dimensions

  1. quality     : accuracy / precision / recall / F1 (sarcastic side) + confusion matrix
  2. cost        : input/output tokens (+ estimated dollars if prices are set)
  3. time        : latency per item (p50 / p95) + total time
  4. reliability : count items the LLM couldn't answer / answered malformed, separate from real predictions

Can run two providers (switch PROVIDER) to see how much the gold bias matters:
  - "claude" : anthropic SDK, claude-opus-4-8
  - "gpt"    : openai SDK, gpt-4o   <- the same one used to mine gold (see PROVENANCE.md)

Input : gold.csv (text, label)
Output: baseline_preds_<provider>.csv  (every item + pred + tokens + latency)

Install: pip install pandas anthropic openai
Set key: ANTHROPIC_API_KEY or OPENAI_API_KEY
Run    : python baseline.py
"""

import json
import os
import statistics
import sys
import time

import pandas as pd

# ================== configurable ==================
PROVIDER = "gpt"           # "claude" or "gpt"
LIMIT = None               # limit item count (for testing before a full run), e.g. 10 ; None = all
SLEEP_SEC = 0.2
SAVE_EVERY = 10            # save every N items (to avoid losing work mid-run)
MAX_RETRY = 4
# =================================================

MODELS = {"claude": "claude-opus-4-8", "gpt": "gpt-4o"}

# price in USD per 1M tokens (input, output)
# claude-opus-4-8 = 5 / 25  (from Anthropic docs)
# gpt-4o = 2.50 / 10.00  (check OpenAI pricing 2026-07 -- standard/real-time tier)
PRICE_PER_MTOK = {"claude": (5.0, 25.0), "gpt": (2.50, 10.0)}

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
# override via env: GOLD_CSV=another set, EVAL_DIR=separate output folder (avoid overwriting the original)
EVAL_DIR = os.environ.get("EVAL_DIR", HERE)
os.makedirs(EVAL_DIR, exist_ok=True)
GOLD_CSV = os.environ.get("GOLD_CSV", os.path.join(HERE, "gold.csv"))
PRED_CSV = os.path.join(EVAL_DIR, f"baseline_preds_{PROVIDER}.csv")

# deliberately plain prompt -- a fair baseline, stating only the definition, no sub-rules like at pre-label time
# (the Thai prompt is the experimental instruction to the model and is kept as-is)
SYSTEM = """ตัดสินว่าข้อความภาษาไทยนี้ "ประชด/เสียดสี" หรือไม่
ประชด = เจตนาจริงตรงข้ามกับความหมายผิวเผิน เพื่อเหน็บหรือแสดงความไม่พอใจ

ตอบเป็น JSON เท่านั้น: {"label": "1" หรือ "0"}
1 = ประชด, 0 = ไม่ประชด"""

LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["0", "1"]}},
    "required": ["label"],
    "additionalProperties": False,
}


# ---------------- LLM callers: return (label, in_tok, out_tok) ----------------

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


def _call_claude(client, text):
    # no thinking: the baseline must be a single plain LLM call, to compare against multi-agent
    # no temperature: claude-opus-4-8 rejects it (400)
    # output_config forces JSON format -> nearly eliminates malformed-parse problems
    r = client.messages.create(
        model=MODELS["claude"],
        max_tokens=256,
        system=SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
        messages=[{"role": "user", "content": f"ข้อความ: {text}"}],
    )
    raw = next((b.text for b in r.content if b.type == "text"), "")
    return raw, r.usage.input_tokens, r.usage.output_tokens


def _call_gpt(client, text):
    r = client.chat.completions.create(
        model=MODELS["gpt"],
        max_tokens=20,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"ข้อความ: {text}"},
        ],
    )
    raw = (r.choices[0].message.content or "").strip()
    return raw, r.usage.prompt_tokens, r.usage.completion_tokens


def predict_one(client, text):
    """Return dict: pred ("0"/"1"/"err"), in_tok, out_tok, latency_ms, err
    Important: on failure -> pred="err", not "0".
    Guessing "0" on failure would be right by chance often (gold is 76% zeros) -> inflated numbers."""
    t0 = time.perf_counter()
    try:
        call = _call_claude if PROVIDER == "claude" else _call_gpt
        raw, in_tok, out_tok = call(client, text)
    except Exception as e:  # SDK retries exhausted and still failing
        return {"pred": "err", "in_tok": 0, "out_tok": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "err": f"{type(e).__name__}: {e}"[:200]}

    latency_ms = round((time.perf_counter() - t0) * 1000)
    try:
        lab = str(json.loads(raw).get("label", "")).strip()
    except json.JSONDecodeError:
        return {"pred": "err", "in_tok": in_tok, "out_tok": out_tok,
                "latency_ms": latency_ms, "err": f"bad json: {raw[:80]}"}
    if lab not in ("0", "1"):
        return {"pred": "err", "in_tok": in_tok, "out_tok": out_tok,
                "latency_ms": latency_ms, "err": f"bad label: {lab[:40]}"}
    return {"pred": lab, "in_tok": in_tok, "out_tok": out_tok,
            "latency_ms": latency_ms, "err": ""}


# ---------------- metrics (hand-written, no sklearn needed) ----------------

def metrics(y_true, y_pred, positive="1"):
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))
    tn = sum(t != positive and p != positive for t, p in zip(y_true, y_pred))
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, prec, rec, f1, (tn, fp, fn, tp)


# ---------------- load / save (resume without forgetting new gold items) ----------------

COLS = ["pred", "in_tok", "out_tok", "latency_ms", "err"]


def load():
    if not os.path.exists(GOLD_CSV):
        sys.exit(f"file not found: {GOLD_CSV}")
    gold = pd.read_csv(GOLD_CSV, dtype=str).fillna("")
    gold["label"] = gold["label"].str.strip()
    gold = gold[gold["label"].isin(["0", "1"])].reset_index(drop=True)

    # always start from gold, then paste old predictions in by text
    # (the old version read from the preds file only -> newly added gold items would never be predicted)
    for c in COLS:
        gold[c] = ""
    if os.path.exists(PRED_CSV):
        old = pd.read_csv(PRED_CSV, dtype=str).fillna("")
        prev = {r["text"]: r for _, r in old.iterrows() if r.get("pred", "") in ("0", "1")}
        hit = 0
        for i in gold.index:
            r = prev.get(gold.at[i, "text"])
            if r is not None:
                for c in COLS:
                    gold.at[i, c] = r.get(c, "")
                hit += 1
        print(f"resuming: reused {hit} old predictions (from {len(old)} rows in the old file)")
    return gold


def save(df):
    df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")


# ---------------- main ----------------

def main():
    price = PRICE_PER_MTOK[PROVIDER]
    df = load()
    todo = [i for i in df.index if df.at[i, "pred"] not in ("0", "1")]
    if LIMIT:
        todo = todo[:LIMIT]

    print(f"\nBASELINE {PROVIDER} ({MODELS[PROVIDER]}) | gold {len(df)} items | to predict {len(todo)} items")
    if not todo:
        print("all predicted, skipping to evaluation")
    else:
        client = _make_client()
        t_start = time.perf_counter()
        for n, idx in enumerate(todo, 1):
            out = predict_one(client, str(df.at[idx, "text"]))
            for c in COLS:
                df.at[idx, c] = str(out[c])
            if out["err"]:
                print(f"  [{n}/{len(todo)}] failed: {out['err'][:70]}")
            if n % SAVE_EVERY == 0 or n == len(todo):
                save(df)
                print(f"  ...{n}/{len(todo)}")
            time.sleep(SLEEP_SEC)
        print(f"total API time: {time.perf_counter() - t_start:.1f}s")
    save(df)

    # ---- evaluate: exclude failed items (err) from the metrics, report them separately ----
    done = df[df["pred"].isin(["0", "1"])]
    bad = df[df["pred"] == "err"]
    acc, prec, rec, f1, (tn, fp, fn, tp) = metrics(done["label"].tolist(), done["pred"].tolist())

    lat = [int(x) for x in df["latency_ms"] if str(x).isdigit()]
    in_tok = sum(int(x) for x in df["in_tok"] if str(x).isdigit())
    out_tok = sum(int(x) for x in df["out_tok"] if str(x).isdigit())

    print("\n" + "=" * 58)
    print(f"BASELINE single agent — {PROVIDER} / {MODELS[PROVIDER]}")
    print("=" * 58)
    print(f"\n[1] quality  (n = {len(done)} successfully predicted)")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}   (positive class = sarcasm)")
    print(f"  Recall   : {rec:.3f}")
    print(f"  F1       : {f1:.3f}")
    print("\n  Confusion matrix (row=true, col=pred)")
    print("              pred:0   pred:1")
    print(f"    true:0     {tn:>5}    {fp:>5}")
    print(f"    true:1     {fn:>5}    {tp:>5}")

    print(f"\n[2] cost")
    print(f"  input tokens : {in_tok:,}")
    print(f"  output tokens: {out_tok:,}")
    if price:
        cost = in_tok / 1e6 * price[0] + out_tok / 1e6 * price[1]
        print(f"  estimated cost: ${cost:.4f}  (${price[0]}/${price[1]} per 1M tokens)")
    else:
        print(f"  (no price set for {PROVIDER} in PRICE_PER_MTOK -> skipping cost)")

    print(f"\n[3] time")
    if lat:
        p95 = sorted(lat)[max(0, int(len(lat) * 0.95) - 1)]
        print(f"  latency/item: p50 {statistics.median(lat):.0f} ms | p95 {p95} ms | total {sum(lat)/1000:.1f}s")

    print(f"\n[4] reliability")
    print(f"  failed predictions (err): {len(bad)}/{len(df)}")
    if len(bad):
        print("  -> these are excluded from the metrics (not guessed as 0)")
        for e in bad["err"].head(3):
            print(f"     {e[:70]}")

    print(f"\nsaved to: {os.path.basename(PRED_CSV)}")
    print("-> keep this F1 + tokens + latency as the baseline, to compare against multi-agent")
    print("-> read PROVENANCE.md before interpreting recall: the sarcastic side of gold was mined by GPT-4o + Claude")


if __name__ == "__main__":
    main()
