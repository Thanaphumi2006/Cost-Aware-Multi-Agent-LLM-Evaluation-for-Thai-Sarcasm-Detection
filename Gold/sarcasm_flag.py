# -*- coding: utf-8 -*-
"""Sarcasm-flagging for a Thai sentiment pipeline.

Sentiment models miscode sarcasm: "บริการดีมาก รอแค่ 2 ชั่วโมง" has a positive surface but a negative
intent, so a naive pipeline labels it POSITIVE when it is a complaint. This module adds a sarcasm stage
that flags exactly those items -- where a NON-negative sentiment is likely inverted -- for a human to
re-check. It does NOT relabel silently; it raises a flag (recall-triage, the safe use per finding 22).

The sarcasm signal is the project's cost-aware cascade: the free lexical cue answers confident cases,
and only cue-unsure text costs a gpt-4.1-mini call. WangchanBERTa is skipped here on purpose -- finding
21 showed the middle tier is inert, so cue -> LLM is both lighter (no torch) and just as accurate.

Programmatic use (drop into your pipeline):

    from sarcasm_flag import flag
    r = flag("บริการดีมาก รอแค่ 2 ชั่วโมง", sentiment="positive")
    # {'sarcastic': 1, 'prob': 0.98, 'by': 'gpt-4.1-mini',
    #  'trust_sentiment': False, 'flag': 'review',
    #  'reason': 'sarcasm likely inverts a non-negative sentiment'}
    if not r["trust_sentiment"]:
        route_to_human(text)          # your pipeline decides what "review" means

Batch use:

    export OPENAI_API_KEY=sk-...      # or put it in .env
    python sarcasm_flag.py --csv scored.csv --out flagged.csv
    #   input : a 'text' column, optionally a 'sentiment' column (positive/negative/neutral)
    #   output: adds sarcastic, sarcasm_prob, decided_by, flag, reason

Honest limits: it over-flags off-domain (finding 22, precision ~0.20 on political comments), so calibrate
on your domain first (calibrate_domain.py) and treat 'review' as "a human should look", not a verdict.
"""
import argparse
import math
import os
import sys

import pandas as pd

import envload  # noqa: F401  -- load OPENAI_API_KEY from .env if present
from cascade_eval import CUES

CUE_CUT = math.log(2.46)   # cue commits only on a strong signal (matches app.html, finding 21)
_det = None


def _cue(text):
    """-> 1 / 0 / None(abstain): the shipped cue tier."""
    hits = [lift for _, rx, lift in CUES if rx.search(text)]
    if not hits:
        return None
    s = sum(math.log(max(l, 0.05)) for l in hits)
    return None if abs(s) < CUE_CUT else (1 if s > 0 else 0)


def _detector():
    global _det
    if _det is None:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            return None
        import predict
        _det = predict.SarcasmDetector(operating="balanced", api_key=key)
    return _det


def sarcasm_signal(text):
    """(label, prob, by): cue answers confident cases free; cue-unsure text escalates to the LLM.
    Returns label=None when the cue is unsure AND no API key is available to resolve it."""
    c = _cue(text)
    if c is not None:
        return c, (1.0 if c == 1 else 0.0), "cue"
    det = _detector()
    if det is None:
        return None, None, "cue-only (no key)"
    p = det.prob(text)
    return (1 if p >= det.t else 0), p, "gpt-4.1-mini"


def _norm_sentiment(v):
    """map many sentiment encodings to positive / negative / neutral / None."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    if s in ("positive", "pos", "p", "+", "1", "บวก", "ดี") or s.startswith("pos"):
        return "positive"
    if s in ("negative", "neg", "n", "-", "-1", "ลบ", "แย่") or s.startswith("neg"):
        return "negative"
    return "neutral"


def flag(text, sentiment=None):
    """decide whether a sentiment label should be trusted for this text.

    sarcasm inverts surface sentiment, so a NON-negative label on sarcastic text is the risky case."""
    label, prob, by = sarcasm_signal(text)
    out = {"sarcastic": label, "prob": None if prob is None else round(prob, 4), "by": by}
    sent = _norm_sentiment(sentiment)

    if label is None:                                   # unresolved (no key, cue unsure) -> be conservative
        out.update(trust_sentiment=False, flag="review",
                   reason="sarcasm unresolved (no API key) -- review to be safe")
    elif label == 0:
        out.update(trust_sentiment=True, flag="ok", reason="no sarcasm signal")
    elif sent == "negative":
        out.update(trust_sentiment=True, flag="ok",
                   reason="sarcastic, but the sentiment is already negative")
    elif sent is None:
        out.update(trust_sentiment=False, flag="review",
                   reason="sarcastic -- re-check its sentiment (surface likely inverted)")
    else:                                               # positive / neutral + sarcastic = the inverted case
        out.update(trust_sentiment=False, flag="review",
                   reason=f"sarcasm likely inverts a {sent} sentiment")
    return out


def run(csv, out, text_col, sent_col):
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print("note: no OPENAI_API_KEY -> cue-only; cue-unsure rows are flagged 'review' unresolved.\n")
    df = pd.read_csv(csv, dtype=str).fillna("")
    if text_col not in df.columns:
        sys.exit(f"input has no '{text_col}' column (columns: {list(df.columns)})")
    has_sent = sent_col in df.columns
    rows, calls, flagged = [], 0, 0
    for _, r in df.iterrows():
        res = flag(r[text_col], r[sent_col] if has_sent else None)
        if res["by"] == "gpt-4.1-mini":
            calls += 1
        if res["flag"] == "review":
            flagged += 1
        row = {text_col: r[text_col]}
        if has_sent:
            row[sent_col] = r[sent_col]
        row.update(sarcastic=res["sarcastic"], sarcasm_prob=res["prob"],
                   decided_by=res["by"], flag=res["flag"], reason=res["reason"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    n = len(rows)
    print(f"{n} rows · {calls} LLM calls ({100*calls/n:.0f}%) · flagged for review {flagged}/{n} "
          f"({100*flagged/n:.0f}%) -> {out}")
    if not has_sent:
        print(f"(no '{sent_col}' column: flags every sarcastic item for a sentiment re-check)")


def main():
    ap = argparse.ArgumentParser(description="flag likely-inverted sentiment (sarcasm) for a Thai pipeline")
    ap.add_argument("--csv", required=True, help="input CSV")
    ap.add_argument("--out", required=True, help="output CSV (adds sarcastic/flag/reason columns)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--sentiment-col", default="sentiment", help="your pipeline's label column (optional)")
    a = ap.parse_args()
    run(a.csv, a.out, a.text_col, a.sentiment_col)


if __name__ == "__main__":
    main()
