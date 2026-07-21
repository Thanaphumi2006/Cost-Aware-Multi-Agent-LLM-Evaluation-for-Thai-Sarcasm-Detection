# -*- coding: utf-8 -*-
"""Cross-domain YouTube test, all in one command -- fetch -> label -> eval automatically

Instead of running 3 scripts one by one, a single command:
  python validate_yt.py "https://youtube.com/watch?v=XXXX"

It will:
  1) fetch Thai comments (skip if already fetched)
  2) open the labeling page (press 1/0/u/b/q · saves every time · quit and rerun to resume)
  3) once enough is labeled (sarcasm >= target), auto-run eval and conclude "does it work on YouTube"

Rerun the same command as many times as needed -- it remembers where it left off
Needs OPENAI_API_KEY at the eval step (not needed for the label step)
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def vid(url):
    m = re.search(r"(?:v=|youtu\.be/|/shorts/)([\w-]{6,})", url)
    return m.group(1) if m else "yt"


def count_pos(csv):
    if not os.path.exists(csv):
        return 0, 0
    import pandas as pd
    d = pd.read_csv(csv, dtype=str).fillna("")
    lab = d["label"].str.strip()
    return int((lab == "1").sum()), int(lab.isin(["0", "1"]).sum())


def main():
    ap = argparse.ArgumentParser(description="test the model on YouTube comments, all in one command")
    ap.add_argument("url", help="YouTube video link")
    ap.add_argument("-n", type=int, default=200, help="number of comments to fetch (default 200)")
    ap.add_argument("--target-pos", type=int, default=30, help="minimum sarcasm before eval (default 30)")
    ap.add_argument("--op", default="balanced", choices=["balanced", "high_recall"])
    a = ap.parse_args()

    base = os.path.join(HERE, f"yt_{vid(a.url)}")
    raw, labeled = base + "_raw.txt", base + "_raw_labeled.csv"

    # 1) fetch (skip if already present)
    if not os.path.exists(raw):
        print("① fetching comments...\n")
        r = subprocess.run([PY, os.path.join(HERE, "fetch_yt_comments.py"), a.url, "-n", str(a.n), "-o", raw])
        if r.returncode or not os.path.exists(raw):
            sys.exit("failed to fetch comments")
    else:
        print(f"① comments already present ({raw}) -- skipping\n")

    # 2) label (interactive)
    npos, ntot = count_pos(labeled)
    if npos < a.target_pos:
        print(f"② label (sarcasm {npos}/{a.target_pos} so far) -- opening the label page...\n")
        subprocess.run([PY, os.path.join(HERE, "label_any.py"), raw, "--out", labeled])
        npos, ntot = count_pos(labeled)
    else:
        print(f"② labeling complete (sarcasm {npos}) -- skipping\n")

    # 3) eval (if enough sarcasm)
    if npos < a.target_pos:
        print(f"\nnot enough sarcasm labeled yet ({npos}/{a.target_pos}) -- rerun the same command to keep labeling")
        print("(or eval now despite the small count, but the CI will be very wide)")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print(f"\n③ ready to eval (sarcasm {npos}) but OPENAI_API_KEY is not set")
        print(f"   export OPENAI_API_KEY=sk-...  then run: python eval_domain.py {os.path.basename(labeled)}")
        return
    print(f"③ eval on YouTube (sarcasm {npos} of {ntot})...\n")
    subprocess.run([PY, os.path.join(HERE, "eval_domain.py"), labeled, "--op", a.op])


if __name__ == "__main__":
    main()
