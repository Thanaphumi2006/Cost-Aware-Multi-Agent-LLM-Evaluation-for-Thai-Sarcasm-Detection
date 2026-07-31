# -*- coding: utf-8 -*-
"""WangchanBERTa Thai sentiment + a sarcasm guard, wired end to end.

A plain sentiment model reads surface form, so it labels a sarcastic complaint
("บริการดีมากกก รอแค่ชั่วโมงเดียวเอง 555") as POSITIVE. This pipeline runs the sentiment model and then
passes each item through sarcasm_flag: when a NON-negative sentiment lands on sarcastic text, the label
is likely inverted, so the row is flagged for a human. It never rewrites the label silently.

Sentiment model: WangchanBERTa fine-tuned on Thai sentiment (default
`phoner45/wangchan-sentiment-thai-text-model`, labels pos/neu/neg), auto-downloaded from HF on first run.
Override with --model or the WCB_SENTIMENT_MODEL env var. Guard: the project's cue -> gpt-4.1-mini cascade
(finding 21), so the free cue answers most rows and only cue-unsure text costs an API call.

Programmatic:

    from sentiment_pipeline import analyze
    r = analyze("บริการดีมากกก รอแค่ชั่วโมงเดียวเอง 555")
    # {'sentiment': 'pos', 'sentiment_conf': 0.97, 'sarcastic': 1, 'flag': 'review',
    #  'reason': 'sarcasm likely inverts a positive sentiment', 'by': 'cue'}
    if r["flag"] == "review":
        route_to_human(text)          # the sentiment label is not safe to trust here

Batch:

    export OPENAI_API_KEY=sk-...      # or .env; without it, cue-unsure rows are flagged 'review'
    python sentiment_pipeline.py --csv texts.csv --out scored.csv     # input needs a 'text' column
"""
import argparse
import os
import sys

import pandas as pd

import envload  # noqa: F401  -- load OPENAI_API_KEY from .env
import sarcasm_flag

DEFAULT_MODEL = os.environ.get("WCB_SENTIMENT_MODEL", "phoner45/wangchan-sentiment-thai-text-model")
_sent = None


def _load(model=DEFAULT_MODEL):
    global _sent
    if _sent is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        mdl = AutoModelForSequenceClassification.from_pretrained(model)
        mdl.eval()
        _sent = (tok, mdl, torch)
    return _sent


def sentiment(text):
    """-> (label, confidence). label is the model's own class name (pos / neu / neg)."""
    tok, mdl, torch = _load()
    with torch.no_grad():
        enc = tok([text], return_tensors="pt", truncation=True, max_length=256)
        probs = torch.softmax(mdl(**enc).logits[0], -1)
    i = int(probs.argmax())
    return mdl.config.id2label[i], float(probs[i])


def analyze(text):
    """WangchanBERTa sentiment, guarded by the sarcasm flag."""
    lab, conf = sentiment(text)
    g = sarcasm_flag.flag(text, sentiment=lab)     # lab (pos/neu/neg) is normalised inside flag()
    return {"sentiment": lab, "sentiment_conf": round(conf, 3),
            "sarcastic": g["sarcastic"], "flag": g["flag"], "reason": g["reason"], "by": g["by"]}


def run(csv, out, text_col):
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print("note: no OPENAI_API_KEY -> sarcasm guard is cue-only; cue-unsure rows flagged 'review'.\n")
    df = pd.read_csv(csv, dtype=str).fillna("")
    if text_col not in df.columns:
        sys.exit(f"input has no '{text_col}' column (columns: {list(df.columns)})")
    print(f"sentiment model: {DEFAULT_MODEL}\n")
    rows, calls, flagged = [], 0, 0
    for t in df[text_col]:
        r = analyze(t)
        if r["by"] == "gpt-4.1-mini":
            calls += 1
        if r["flag"] == "review":
            flagged += 1
        rows.append({text_col: t, "sentiment": r["sentiment"], "sentiment_conf": r["sentiment_conf"],
                     "sarcastic": r["sarcastic"], "flag": r["flag"], "reason": r["reason"]})
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    n = len(rows)
    print(f"{n} rows · {calls} sarcasm-LLM calls ({100*calls/max(n,1):.0f}%) · "
          f"sentiment flagged as untrustworthy {flagged}/{n} ({100*flagged/max(n,1):.0f}%) -> {out}")


def main():
    global DEFAULT_MODEL
    ap = argparse.ArgumentParser(description="WangchanBERTa sentiment + sarcasm guard")
    ap.add_argument("--csv", required=True, help="input CSV with a 'text' column")
    ap.add_argument("--out", required=True, help="output CSV: text, sentiment, sentiment_conf, sarcastic, flag, reason")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF id of the WangchanBERTa sentiment model")
    a = ap.parse_args()
    DEFAULT_MODEL = a.model
    run(a.csv, a.out, a.text_col)


if __name__ == "__main__":
    main()
