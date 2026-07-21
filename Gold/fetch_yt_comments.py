# -*- coding: utf-8 -*-
"""Fetch YouTube comments (Thai) into a .txt file, ready to feed label_any.py -- closes step 3 for the YouTube domain

Fetch public comments from the given video with yt-dlp -> keep only those containing Thai characters
-> drop too-short/too-long/duplicates -> write one per line into the .txt

Used only to "gather raw text" for cross-domain testing (humans label it themselves, with the same criteria)
Public data · fetched to validate one own model

Usage:
  python fetch_yt_comments.py "https://youtube.com/watch?v=XXXX" -n 200
  python fetch_yt_comments.py URL1 URL2 URL3 -o yt_raw.txt      # several videos combined
  then: python label_any.py yt_raw.txt   ->   python eval_domain.py yt_raw_labeled.csv
"""
import argparse
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
THAI = re.compile(r"[฀-๿]")


def is_thai(s, min_thai=3):
    return len(THAI.findall(s)) >= min_thai


def clean(s):
    s = re.sub(r"\s+", " ", s).strip()      # collapse newlines/spaces -> 1 comment = 1 line
    return s


def fetch(url, limit):
    import yt_dlp
    opts = {
        "getcomments": True, "skip_download": True, "quiet": True, "no_warnings": True,
        "extractor_args": {"youtube": {"comment_sort": ["top"], "max_comments": [str(limit * 3)]}},
    }
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=False)
    return [c.get("text", "") for c in (info.get("comments") or [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+", help="YouTube video links (multiple allowed)")
    ap.add_argument("-n", type=int, default=200, help="target number of Thai comments (across all videos)")
    ap.add_argument("-o", "--out", default="yt_raw.txt")
    ap.add_argument("--min-len", type=int, default=8, help="drop shorter than this (characters)")
    ap.add_argument("--max-len", type=int, default=300, help="drop longer than this")
    a = ap.parse_args()

    seen, kept = set(), []
    for url in a.urls:
        if len(kept) >= a.n:
            break
        print(f"fetching {url} ...", file=sys.stderr, flush=True)
        try:
            raw = fetch(url, a.n)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        n_thai = 0
        for c in raw:
            c = clean(c)
            if not is_thai(c) or not (a.min_len <= len(c) <= a.max_len):
                continue
            if c in seen:
                continue
            seen.add(c); kept.append(c); n_thai += 1
            if len(kept) >= a.n:
                break
        print(f"  got {n_thai} Thai comments (total {len(kept)})", file=sys.stderr)

    if not kept:
        sys.exit("no Thai comments at all -- the video may have comments disabled, or comments are not Thai")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(kept))
    print(f"\nwrote {len(kept)} comments -> {a.out}")
    print(f"next: python label_any.py {a.out}   (label ~150 items to get sarcasm >=30)")
    print(f"       python eval_domain.py {a.out.replace('.txt','_labeled.csv')}")


if __name__ == "__main__":
    main()
