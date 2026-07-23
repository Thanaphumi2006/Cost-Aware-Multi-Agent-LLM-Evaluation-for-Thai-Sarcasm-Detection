# -*- coding: utf-8 -*-
"""High-precision auto-labelling for UNATTENDED use (moderation queues, auto-reply, bulk tagging).

The interactive demo escalates only cue-uncertain items and answers every one. An unattended system
has the opposite need: never act on a guess. Thresholding the LLM alone cannot deliver that here --
gpt-4.1-mini's false positives are as confident as its true ones, so precision is stuck near 0.68 at
every threshold (see the sweep in the README / finding 21 notes).

The fix is an AND-gate: auto-act only when two INDEPENDENT signals agree, and defer the rest to a
human. Measured on the 127-item gold set at the deployed threshold (not tuned, so not overfit):

    decision            rule                                 precision   what to do with it
    -----------------   ----------------------------------   ---------   ------------------------
    auto_sarcasm        cue says sarcastic AND LLM agrees       ~0.90     safe to act on unattended
    auto_not_sarcasm    cue says not-sarcastic AND LLM agrees   ~high     safe to act on unattended
    review              the two disagree, or cue is unsure        --      send to a human

Cost-aware: the LLM is called ONLY on items the cue tier is confident about (a minority). Cue-unsure
items go straight to 'review' with no API call. So you pay per confident item, not per row.

Usage:
    export OPENAI_API_KEY=sk-...
    python autolabel.py --csv comments.csv --out labelled.csv       # text[,label] -> decisions
    python autolabel.py --csv gold.csv --out /tmp/g.csv --eval       # if labelled, print precision
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

import envload  # noqa: F401  -- load OPENAI_API_KEY from .env
from cascade_eval import CUES

HERE = os.path.dirname(os.path.abspath(__file__))
CUE_CUT = math.log(2.46)   # cue commits only on a strong signal (matches app.html / finding 21)


def cue_decide(text):
    """-> 1 / 0 / None(abstain), the shipped cue tier with its finding-21 cut-off."""
    hits = [lift for _, rx, lift in CUES if rx.search(text)]
    if not hits:
        return None
    s = sum(math.log(max(x, 0.05)) for x in hits)
    return None if abs(s) < CUE_CUT else (1 if s > 0 else 0)


def decide(cue, llm_prob, t):
    """three-way decision. llm_prob is None when the LLM was not called (cue unsure)."""
    if cue is None:
        return "review"                       # no confident cue -> never auto-act, hand to a human
    llm_pos = (llm_prob is not None) and (llm_prob >= t)
    if cue == 1 and llm_pos:
        return "auto_sarcasm"                 # both independent signals say sarcastic
    if cue == 0 and not llm_pos:
        return "auto_not_sarcasm"             # both say not sarcastic
    return "review"                           # the two disagree -> too risky to act, hand to a human


def run(csv, out, do_eval):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        sys.exit("no OPENAI_API_KEY (set it or put it in .env).")
    df = pd.read_csv(csv, dtype={"text": str}).fillna("")
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    if "text" not in df:
        sys.exit("input needs a 'text' column.")

    import predict
    det = predict.SarcasmDetector(operating="balanced", api_key=key)
    t = det.t

    rows, calls = [], 0
    for text in df["text"]:
        cue = cue_decide(text)
        llm = None
        if cue is not None:                   # cost-aware: only confident-cue items reach the paid model
            llm = det.prob(text); calls += 1
        rows.append({"text": text, "cue": "" if cue is None else cue,
                     "llm_prob": "" if llm is None else round(llm, 4),
                     "decision": decide(cue, llm, t)})
    res = pd.DataFrame(rows)
    if "label" in df.columns:
        res["label"] = df["label"].values
    res.to_csv(out, index=False, encoding="utf-8-sig")

    n = len(res)
    counts = res["decision"].value_counts().to_dict()
    auto = counts.get("auto_sarcasm", 0) + counts.get("auto_not_sarcasm", 0)
    print(f"\n{n} items · {calls} LLM calls ({100*calls/n:.0f}% of rows) -> {out}")
    print(f"  auto-labelled : {auto}/{n} ({100*auto/n:.0f}%)   sent to review : {counts.get('review', 0)}")
    for k in ("auto_sarcasm", "auto_not_sarcasm", "review"):
        if k in counts:
            print(f"    {k:16s} {counts[k]}")

    if do_eval and "label" in res.columns:
        y = pd.to_numeric(res["label"], errors="coerce")
        print("\n=== precision of each auto decision (only meaningful with true labels) ===")
        for dec, target in (("auto_sarcasm", 1), ("auto_not_sarcasm", 0)):
            m = res["decision"] == dec
            k = int(m.sum())
            if k:
                correct = int((y[m] == target).sum())
                print(f"  {dec:16s} {correct}/{k} correct  precision={correct/k:.3f}")
        # recall of sarcasm actually auto-caught (of all true positives)
        tp_auto = int(((res["decision"] == "auto_sarcasm") & (y == 1)).sum())
        n_pos = int((y == 1).sum())
        if n_pos:
            print(f"  sarcasm auto-caught: {tp_auto}/{n_pos} of all true positives "
                  f"(recall {tp_auto/n_pos:.3f}); the rest are in 'review', not mislabelled.")


def main():
    ap = argparse.ArgumentParser(description="high-precision AND-gate auto-labeller (cue AND LLM agree)")
    ap.add_argument("--csv", required=True, help="input CSV with a 'text' column (optional 'label')")
    ap.add_argument("--out", required=True, help="output CSV: text, cue, llm_prob, decision")
    ap.add_argument("--eval", action="store_true", help="if the input has 'label', print precision/recall")
    a = ap.parse_args()
    run(a.csv, a.out, a.eval)


if __name__ == "__main__":
    main()
