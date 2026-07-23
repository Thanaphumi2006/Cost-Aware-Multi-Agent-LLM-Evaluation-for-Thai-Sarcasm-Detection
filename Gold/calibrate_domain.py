# -*- coding: utf-8 -*-
"""Calibrate the detector to YOUR domain -- the make-or-break step for real use.

Finding 12 showed precision collapses 0.68 -> 0.40 when you move off the reviews/tweets gold was
built from. The fix is not a new model; it is re-picking the escalation threshold on a labelled
sample of your own target content. This tool does that end to end, reusing the EXACT deployed
scorer (predict.SarcasmDetector, operating="balanced" = gpt-4.1-mini @ t=0.095) so the numbers
reflect what the demo will actually do.

Three-step workflow (see CALIBRATE.md for the full runbook):

  1. COLLECT   python calibrate_domain.py --name myshop --fetch https://www.youtube.com/watch?v=...
               -> writes domain_myshop_labeled.csv  (text + an empty label column)

  2. LABEL     open domain_myshop_labeled.csv and fill label = 1 (sarcastic) / 0 (not) by hand,
               same rubric as gold (feigned praise -- see labeling_rubric.md). Aim for >=30 sarcastic.
               (or use label_ui.py:  DOMAIN_LABEL_FILE=domain_myshop_labeled.csv python label_ui.py)

  3. CALIBRATE python calibrate_domain.py --name myshop --csv domain_myshop_labeled.csv
               -> scores once (cached), then prints cue-only vs LLM@deployed vs LLM@domain-tuned,
                  and the exact threshold to deploy.

Everything the domain data touches is gitignored (domain_*), so scraped third-party text never ships.
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

import envload  # noqa: F401  -- loads OPENAI_API_KEY from .env on import
from cascade_eval import CUES, prf

HERE = os.path.dirname(os.path.abspath(__file__))
CUE_CUT = math.log(2.46)          # finding 21: cue commits only when a strong cue fires (matches app.html)
SEED, N_FOLDS = 42, 5


# ---- the shipped cue tier, so the "if you never escalated" floor is the real one ----
def cue_decide(text):
    """-> 1 / 0 / None(abstain), exactly as verdictOf() in app.html after the finding-21 cut-off."""
    hits = [lift for _, rx, lift in CUES if rx.search(text)]
    if not hits:
        return None
    s = sum(math.log(max(l, 0.05)) for l in hits)
    return None if abs(s) < CUE_CUT else (1 if s > 0 else 0)


def report_row(name, y, pred, calls, n):
    p, r, f = prf(y, pred)
    print(f"  {name:34s} P={p:.3f} R={r:.3f} F1={f:.3f}   LLM calls {calls}/{n} ({100*calls/n:.0f}%)")
    return f


# ---------------------------------------------------------------- collect
def do_fetch(name, url, limit):
    import fetch_social as fs
    try:
        texts, plat = fs.fetch_any(url, limit=limit)
    except fs.UnsupportedError:
        sys.exit(f"'{url}' is a login/paid-API platform. Supported: YouTube, Pantip, Reddit.")
    except fs.FetchError as e:
        sys.exit(f"fetch failed: {e}")
    if not texts:
        sys.exit("no Thai comments found at that link.")
    out = os.path.join(HERE, f"domain_{name}_labeled.csv")
    pd.DataFrame({"text": texts, "label": ""}).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"fetched {len(texts)} {plat} comments -> {out}")
    print("next: open it, fill the label column (1=sarcastic, 0=not), then re-run with --csv "
          f"domain_{name}_labeled.csv")


# ---------------------------------------------------------------- calibrate
def do_calibrate(name, csv):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        sys.exit("no OPENAI_API_KEY (set it or put it in .env) -- scoring needs the deployed model.")

    df = pd.read_csv(csv, dtype={"text": str, "label": str}).fillna("")
    df["label"] = df["label"].str.strip()
    df = df[df["label"].isin(["0", "1"])].reset_index(drop=True)
    if len(df) < 20:
        sys.exit(f"only {len(df)} labelled items -- label at least ~20 (and >=8 sarcastic) first.")
    y = df["label"].astype(int).values
    n, npos = len(df), int(y.sum())
    print(f"\ndomain '{name}': {n} labelled items, {npos} sarcastic ({100*npos/n:.1f}% base rate)")
    if npos < 8:
        print("  warning: <8 sarcastic items -> every number below has a very wide CI; label more.")

    # ---- score once with the DEPLOYED scorer (cached to disk, so re-runs are free) ----
    import predict
    prob_csv = os.path.join(HERE, f"domain_{name}_probs.csv")
    cached = {}
    if os.path.exists(prob_csv):
        pc = pd.read_csv(prob_csv, dtype={"text": str})
        cached = dict(zip(pc["text"], pc["prob"].astype(float)))
    det = predict.SarcasmDetector(operating="balanced", api_key=key)
    T_DEPLOY = det.t
    probs, fresh = [], 0
    for t in df["text"]:
        if t in cached:
            probs.append(cached[t])
        else:
            probs.append(det.prob(t)); fresh += 1
    probs = np.array(probs, dtype=float)
    pd.DataFrame({"text": df["text"], "label": df["label"], "prob": probs}).to_csv(
        prob_csv, index=False, encoding="utf-8-sig")
    print(f"scored {n} items ({fresh} new API calls, {n-fresh} from cache) -> {prob_csv}\n")

    # ---- baselines on THIS domain -------------------------------------------------
    print("=== on your domain ===")
    cue = np.array([cue_decide(t) for t in df["text"]], dtype=object)
    cue_full = np.array([0 if c is None else int(c) for c in cue])   # never-escalate floor (None->0)
    n_cue = int((cue != None).sum())                                 # noqa: E711
    report_row(f"cue-only floor (answers {n_cue}/{n})", y, cue_full, 0, n)
    report_row(f"LLM @ deployed t={T_DEPLOY:.3f}", y, (probs >= T_DEPLOY).astype(int), n, n)

    # ---- the calibration: leave-fold-out tuned threshold on this domain ----
    from sklearn.model_selection import StratifiedKFold
    folds = min(N_FOLDS, npos, n - npos)
    if folds < 2:
        print("\n  too few of one class to cross-validate a threshold -- label more, especially the rare class.")
        return
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    pred, taus = np.zeros(n, dtype=int), []
    for tr, te in skf.split(probs, y):
        grid = np.unique(probs[tr])
        t = max(grid, key=lambda x: prf(y[tr], (probs[tr] >= x).astype(int))[2])
        pred[te] = (probs[te] >= t).astype(int); taus.append(t)
    f_tuned = report_row(f"LLM @ domain-tuned ({folds}-fold)", y, pred, n, n)

    # single deployable threshold (chosen on all domain data -- the value to actually ship)
    grid = np.unique(probs)
    t_ship = float(max(grid, key=lambda x: prf(y, (probs >= x).astype(int))[2]))
    f_deploy = prf(y, (probs >= T_DEPLOY).astype(int))[2]

    print("\n=== what to do ===")
    print(f"  per-fold thresholds: {', '.join(f'{t:.3f}' for t in taus)}  (spread shows stability)")
    print(f"  honest domain F1 (leave-fold-out): {f_tuned:.3f}   vs deployed threshold: {f_deploy:.3f}")
    print(f"  -> deploy threshold  t = {t_ship:.3f}  for domain '{name}'")
    print( "     set it in predict.py OPERATING['balanced']['t'], or pass operating point per call.")
    if f_tuned <= f_deploy + 0.01:
        print("     (note: tuning barely moved F1 here -- the deployed 0.095 already fits this domain.)")
    print( "     you can also skip tuning and just use the demo's 'correct it' button, which adapts in-context.")


def main():
    ap = argparse.ArgumentParser(description="calibrate the detector to your own domain")
    ap.add_argument("--name", required=True, help="short domain name (used in file names)")
    ap.add_argument("--fetch", metavar="URL", help="collect: pull Thai comments from a YouTube/Pantip/Reddit link")
    ap.add_argument("--limit", type=int, default=120, help="max comments to fetch (default 120)")
    ap.add_argument("--csv", help="calibrate: a labelled CSV with columns text,label")
    a = ap.parse_args()
    if a.fetch:
        do_fetch(a.name, a.fetch, a.limit)
    elif a.csv:
        do_calibrate(a.name, a.csv)
    else:
        ap.print_help()
        sys.exit("\nprovide --fetch URL (to collect) or --csv FILE (to calibrate)")


if __name__ == "__main__":
    main()
