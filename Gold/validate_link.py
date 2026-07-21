# -*- coding: utf-8 -*-
"""Cross-domain test from a link, all in one command: fetch comments -> label -> measure F1

Supports Pantip, YouTube, Reddit (see fetch_social.py). Good for "cross-domain validation":
  python validate_link.py "https://pantip.com/topic/XXXXXXXX"

It will:
  1) fetch Thai comments (skip if already fetched)
  2) open the labeling page (press 1=sarcasm 0=not u=skip b=back q=save&quit · saves every time · rerun to resume)
  3) once enough is labeled, auto-run eval, report F1 on this domain vs the original gold

Tip for picking Pantip threads with sarcasm: choose threads with debate/complaints/politics/drama
(plain Q&A threads usually have no sarcasm, hard to label, few positives)

Needs OPENAI_API_KEY at the eval step (not needed for the label step)
"""
import argparse
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def base_name(url, plat):
    h = hashlib.sha1(url.encode()).hexdigest()[:8]
    return os.path.join(HERE, f"domain_{plat}_{h}")


def counts(csv):
    if not os.path.exists(csv):
        return 0, 0
    import pandas as pd
    d = pd.read_csv(csv, dtype=str).fillna("")
    lab = d["label"].str.strip()
    return int(lab.isin(["0", "1"]).sum()), int((lab == "1").sum())


def main():
    ap = argparse.ArgumentParser(description="cross-domain validation from a link (Pantip/YouTube/Reddit)")
    ap.add_argument("url", help="thread/video link")
    ap.add_argument("-n", type=int, default=120, help="number of comments to fetch (default 120)")
    ap.add_argument("--min-label", type=int, default=30, help="label at least this many before eval (default 30)")
    ap.add_argument("--op", default="balanced", choices=["balanced", "high_recall"])
    a = ap.parse_args()

    import fetch_social as fs
    plat = fs.platform_of(a.url)
    base = base_name(a.url, plat)
    raw, labeled = base + "_raw.txt", base + "_raw_labeled.csv"

    # 1) fetch
    if not os.path.exists(raw):
        print(f"[1] fetching comments from {plat} ...")
        try:
            comments, plat = fs.fetch_any(a.url, a.n)
        except fs.UnsupportedError as e:
            sys.exit(f"cannot auto-fetch from {plat} ({e}). Supported: Pantip, YouTube, Reddit")
        except Exception as e:
            sys.exit(f"fetch failed: {type(e).__name__}: {e}")
        if not comments:
            sys.exit("no Thai comments found (try another thread)")
        with open(raw, "w", encoding="utf-8") as f:
            f.write("\n".join(comments))
        print(f"    got {len(comments)} comments -> {os.path.basename(raw)}\n")
    else:
        print(f"[1] comments already present ({os.path.basename(raw)}), skipping\n")

    # 2) label
    ntot, npos = counts(labeled)
    if ntot < a.min_label:
        print(f"[2] label (done {ntot} items, sarcasm {npos}) opening the label page ...\n")
        subprocess.run([PY, os.path.join(HERE, "label_any.py"), raw, "--out", labeled])
        ntot, npos = counts(labeled)
    else:
        print(f"[2] labeling complete ({ntot} items, sarcasm {npos}), skipping\n")

    # 3) eval
    if ntot < 10:
        print(f"\ntoo few labels ({ntot} items), rerun the same command to keep labeling, then eval")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print(f"\n[3] ready to eval ({ntot} items) but OPENAI_API_KEY is not set")
        print(f"    export OPENAI_API_KEY=sk-...  then run: python eval_domain.py {os.path.basename(labeled)}")
        return
    if npos < 10:
        print(f"[!] only {npos} sarcastic items, F1/CI will be rough (but still shows the direction)")
    print(f"[3] measuring F1 on the {plat} domain ({ntot} items, sarcasm {npos}) ...\n")
    subprocess.run([PY, os.path.join(HERE, "eval_domain.py"), labeled, "--op", a.op])


if __name__ == "__main__":
    main()
