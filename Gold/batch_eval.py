# -*- coding: utf-8 -*-
"""Batch (offline) evaluation via the OpenAI Batch API -- 50% cheaper in exchange for "not real-time"

Use only for "fire the whole file at once" jobs (frontier/baseline/re-run on the expanded gold)
The live web (app.py/predict.py) still needs normal calls, because a batch can take up to 24h

Logic identical to predict.py: same prompt (DETECT_SYS), read logprob -> P(sarcasm),
compare against the threshold from OPERATING -> label. **No corrections applied** (evaluate the "base system" cleanly, no leak)

Steps (async -- a batch does not finish immediately):
  1) python batch_eval.py --csv in.csv --dry-run          write .batch.jsonl + count (free, no calls)
  2) python batch_eval.py --csv in.csv --out out.csv       fire + wait until done + write results (blocking)
     or split into two steps if you do not want to wait blocking:
  2a) python batch_eval.py --csv in.csv --no-wait          fire, print the batch id, then exit
  2b) python batch_eval.py --csv in.csv --out out.csv --fetch <batch_id>   fetch the results of a finished batch

Note: --fetch needs --csv "the same file, same order" because results are mapped back by custom_id = row-<index>
"""
import argparse
import json
import math
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from predict import DETECT_SYS, OPERATING   # use the same prompt + operating point (model/threshold) as the real thing


def build_request(i, text, model):
    """one line in the JSONL = one request (same body predict._call fires, but without corrections)"""
    return {
        "custom_id": f"row-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "max_tokens": 20,
            "response_format": {"type": "json_object"},
            "logprobs": True,
            "top_logprobs": 20,
            "messages": [
                {"role": "system", "content": DETECT_SYS},
                {"role": "user", "content": f"ข้อความ: {text}"},
            ],
        },
    }


def prob_from_body(body):
    """extract P(sarcasm) from the response body (dict from batch output) -- same logic as predict._call"""
    try:
        ch = body["choices"][0]
    except (KeyError, IndexError):
        return float("nan")
    lp = ch.get("logprobs") or {}
    for tok in (lp.get("content") or []):
        if tok.get("token", "").strip().strip('"') not in ("0", "1"):
            continue
        p0 = p1 = 0.0
        for alt in tok.get("top_logprobs") or []:
            t = alt.get("token", "").strip().strip('"')
            if t == "1":
                p1 += math.exp(alt["logprob"])
            elif t == "0":
                p0 += math.exp(alt["logprob"])
        if p0 + p1 > 0:
            return p1 / (p0 + p1)
    # cannot read the logprob -> use the hard answer from JSON
    try:
        return 1.0 if str(json.loads(ch["message"]["content"] or "{}").get("label", "")) == "1" else 0.0
    except (json.JSONDecodeError, KeyError, TypeError):
        return float("nan")


def load_texts(csv_path, text_col):
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if text_col not in df.columns:
        sys.exit(f"no column '{text_col}' (have: {list(df.columns)})")
    return df


def write_jsonl(path, df, text_col, model):
    with open(path, "w", encoding="utf-8") as f:
        for i, text in enumerate(df[text_col]):
            f.write(json.dumps(build_request(i, text, model), ensure_ascii=False) + "\n")


def parse_output(raw_text):
    """map custom_id 'row-<i>' -> prob from the batch result file (JSONL)"""
    probs = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cid = rec.get("custom_id", "")
        resp = rec.get("response") or {}
        if resp.get("status_code") != 200:
            probs[cid] = float("nan")
            continue
        probs[cid] = prob_from_body(resp.get("body") or {})
    return probs


def finish(df, probs, t, out_path):
    """take probs (dict custom_id->prob), compare against the threshold, and write out.csv"""
    labels, prob_col, dec = [], [], []
    for i in range(len(df)):
        p = probs.get(f"row-{i}", float("nan"))
        if p != p:
            labels.append(None); prob_col.append(None); dec.append("error"); continue
        lab = 1 if p >= t else 0
        labels.append(lab); prob_col.append(round(p, 3))
        dec.append("sarcasm" if lab else "not_sarcasm")
    df = df.copy()
    df["pred_prob"] = prob_col
    df["pred_label"] = labels
    df["pred_decision"] = dec
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n = len(df); ns = sum(1 for d in dec if d == "sarcasm"); ne = sum(1 for d in dec if d == "error")
    print(f"wrote {out_path} · {n} items · sarcasm {ns}" + (f" · error {ne}" if ne else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="input file (must have a text column)")
    ap.add_argument("--out", help="output file (required when writing results)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--op", default="balanced", choices=list(OPERATING))
    ap.add_argument("--model", help="override --op model (e.g. sweeping frontier)")
    ap.add_argument("--dry-run", action="store_true", help="write .jsonl + count, no API calls")
    ap.add_argument("--no-wait", action="store_true", help="fire, print the batch id, do not wait")
    ap.add_argument("--fetch", help="fetch results of a finished batch id (use with the same --csv + --out)")
    ap.add_argument("--poll", type=int, default=20, help="status polling interval (seconds)")
    a = ap.parse_args()

    df = load_texts(a.csv, a.text_col)
    model = a.model or OPERATING[a.op]["model"]
    t = OPERATING[a.op]["t"]

    # --dry-run: write the jsonl only
    jsonl_path = os.path.splitext(a.csv)[0] + ".batch.jsonl"
    if a.dry_run:
        write_jsonl(jsonl_path, df, a.text_col, model)
        print(f"[dry-run] wrote {jsonl_path} · {len(df)} requests · model={model} · not fired (batch = ~50% of normal price)")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY required (export OPENAI_API_KEY=sk-...)")
    from openai import OpenAI
    client = OpenAI(timeout=60.0, max_retries=3)

    # --fetch: skip firing, go fetch the results of an existing batch
    if a.fetch:
        b = client.batches.retrieve(a.fetch)
        if b.status != "completed":
            sys.exit(f"batch {a.fetch} not finished yet (status={b.status})")
        if not a.out:
            sys.exit("also pass --out with --fetch")
        raw = client.files.content(b.output_file_id).text
        finish(df, parse_output(raw), t, a.out)
        return

    # fresh fire: write jsonl -> upload -> create batch
    write_jsonl(jsonl_path, df, a.text_col, model)
    up = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h")
    print(f"[batch] fired · id={batch.id} · {len(df)} requests · model={model}", file=sys.stderr)

    if a.no_wait:
        print(batch.id)
        print(f"fetch later with:  python batch_eval.py --csv {a.csv} --out out.csv --fetch {batch.id}",
              file=sys.stderr)
        return

    if not a.out:
        sys.exit("also pass --out (or use --no-wait then --fetch)")

    # wait until done (real batches usually finish well under 24h, but it is async)
    while True:
        b = client.batches.retrieve(batch.id)
        rc = b.request_counts
        print(f"[batch] status={b.status} · {rc.completed}/{rc.total} done · fail {rc.failed}", file=sys.stderr)
        if b.status == "completed":
            break
        if b.status in ("failed", "expired", "cancelled"):
            sys.exit(f"batch ended unsuccessfully: {b.status}")
        time.sleep(a.poll)

    raw = client.files.content(b.output_file_id).text
    probs = parse_output(raw)
    if b.error_file_id:   # if some items errored, keep it so we know
        err = client.files.content(b.error_file_id).text
        print(f"[batch] some items errored -- see details, {len(err.splitlines())} lines", file=sys.stderr)
    finish(df, probs, t, a.out)


if __name__ == "__main__":
    main()
