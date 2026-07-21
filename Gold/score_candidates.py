# -*- coding: utf-8 -*-
"""Score P(sarcasm) for candidates in to_label_next.csv via logprob -> re-rank to hunt for positives

Reason: the old ranking used a keyword signal that is noisy (priority 2 = 142 items, poorly separated)
the model logprob ranks much better -> lift positive candidates (high P) to the top + surface the borderline band (0.2-0.8)
the bottleneck is positives (n=30) -> we want to label items "likely to be real positives" first

uses gpt-4.1-mini (the best + cheap model in finding 9) · the original DETECT_SYS · ~$0.03
Run: python score_candidates.py
"""
import math
import os
import sys

import pandas as pd

import multiagent

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "to_label_next.csv")
MODEL = "gpt-4.1-mini"


def score_one(client, text):
    r = client.chat.completions.create(
        model=MODEL, max_tokens=20, response_format={"type": "json_object"},
        logprobs=True, top_logprobs=20,
        messages=[{"role": "system", "content": multiagent.DETECT_SYS},
                  {"role": "user", "content": f"ข้อความ: {text}"}],
    )
    it, ot = r.usage.prompt_tokens, r.usage.completion_tokens
    for tok in (r.choices[0].logprobs.content or []):
        if tok.token.strip().strip('"') not in ("0", "1"):
            continue
        p0 = p1 = 0.0
        for alt in tok.top_logprobs:
            t = alt.token.strip().strip('"')
            if t == "1":
                p1 += math.exp(alt.logprob)
            elif t == "0":
                p0 += math.exp(alt.logprob)
        if p0 + p1 > 0:
            return p1 / (p0 + p1), it, ot
    return float("nan"), it, ot


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY required")
    df = pd.read_csv(CSV, dtype=str).fillna("")
    from openai import OpenAI
    client = OpenAI(timeout=30.0, max_retries=3)

    probs, ti, to = [], 0, 0
    for n, text in enumerate(df["text"], 1):
        try:
            p, i, o = score_one(client, text)
        except Exception as e:
            print(f"\n  item {n} failed: {type(e).__name__}"); p, i, o = float("nan"), 0, 0
        probs.append(p); ti += i; to += o
        print(f"  {n}/{len(df)}", end="\r", flush=True)

    df["P_sarcasm"] = [round(p, 3) if p == p else "" for p in probs]
    # borderline band 0.2-0.8 = items the model is unsure about = separates systems well + requires real human judgment
    df["band"] = ["borderline (0.2-0.8)" if (p == p and 0.2 <= p <= 0.8)
                  else ("likely sarcasm (>0.8)" if (p == p and p > 0.8)
                        else "likely not (<0.2)") for p in probs]
    # re-rank: high P first (hunt positives) -- high-P items are the best positive candidates
    df["_s"] = [p if p == p else -1 for p in probs]
    df = df.sort_values("_s", ascending=False).drop(columns="_s")
    df.to_csv(CSV, index=False, encoding="utf-8-sig")

    c = ti/1e6*0.40 + to/1e6*1.60
    hi = sum(1 for p in probs if p == p and p > 0.8)
    mid = sum(1 for p in probs if p == p and 0.2 <= p <= 0.8)
    print(f"\nscored {len(df)} items | ${c:.4f} | {ti} in / {to} out")
    print(f"  likely sarcasm (P>0.8): {hi} items  <- best positive candidates, label first")
    print(f"  borderline (0.2-0.8) : {mid} items  <- needs human judgment, separates systems well")
    print(f"overwrote {os.path.basename(CSV)} (sorted P high->low, added columns P_sarcasm + band)")


if __name__ == "__main__":
    main()
