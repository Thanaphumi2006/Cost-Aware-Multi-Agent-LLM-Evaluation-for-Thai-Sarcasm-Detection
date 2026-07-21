# -*- coding: utf-8 -*-
"""Fetch Thai comments from multiple links/platforms into one CSV (a text column) -- feeds distillation/eval

The "entry point" of the data-expansion pipeline: take any URL fetch_social supports (YouTube/Pantip/Reddit)
dump the raw text into a CSV, then have the teacher put silver labels on it

*** the key trick (fixes the cross-domain problem where precision drops 0.68->0.40) ***
fetch from "the domain you will actually deploy on" (e.g. the Pantip web board), not just Wongnai/Wisesight
-> silver from the target domain = teaches the student that domain = exactly where the model used to fail

Usage:
  python fetch_to_csv.py <url1> <url2> ... --out pool.csv --limit 100
Chain next:
  python batch_eval.py --csv pool.csv --out pool_pred.csv          # teacher labels (half price)
  python distill_label.py --pred pool_pred.csv --out silver.csv    # filter by confidence -> silver
  python distill_train_eval.py --silver silver.csv                 # train + eval OOF
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_social import fetch_any, UnsupportedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+", help="YouTube/Pantip/Reddit links (multiple allowed)")
    ap.add_argument("--out", default="pool.csv")
    ap.add_argument("--limit", type=int, default=80, help="number of comments per URL")
    a = ap.parse_args()

    import pandas as pd
    rows, seen = [], set()
    for url in a.urls:
        try:
            texts, plat = fetch_any(url, a.limit)
        except UnsupportedError as e:
            print(f"skipping {url} -- not freely accessible ({e})", file=sys.stderr)
            continue
        n = 0
        for t in texts:
            if t not in seen:
                seen.add(t); rows.append({"text": t, "source": plat, "url": url}); n += 1
        print(f"{plat}: +{n} texts  ({url})", file=sys.stderr)

    if not rows:
        sys.exit("no texts at all (links inaccessible / no Thai comments)")
    pd.DataFrame(rows).to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"wrote {a.out} · {len(rows)} texts (unique) from {len(a.urls)} links · "
          f"continue with batch_eval.py --csv {a.out}")


if __name__ == "__main__":
    main()
