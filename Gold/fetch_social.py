# -*- coding: utf-8 -*-
"""Fetch Thai comments from a link -- only platforms where comments are genuinely accessible "free, no login"

Supported (free, no API key / no login):
  - YouTube : yt-dlp (reliable)
  - Pantip  : the site own public JSON (Thai forum -- lots of sarcasm, the best fit for this task)
  - Reddit  : public .json (free, but some IPs get 403-blocked -> best-effort)

Platforms "dropped" because comments are not freely accessible (forced login/cookies/or paid API):
  Twitter/X, Instagram, TikTok, Facebook -> raise UnsupportedError so the caller tells the user to "paste it themselves"

Public data, fetched to test one own model · keeps only items containing Thai characters
"""
import json
import re
import urllib.request

THAI = re.compile(r"[฀-๿]")
TAG = re.compile(r"<[^>]+>")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

SUPPORTED = ("YouTube", "Pantip", "Reddit")   # shown in the web app as fetchable


class UnsupportedError(Exception):
    """this platform comments are not freely accessible (needs login / paid API)"""


def is_thai(s, min_thai=3):
    return len(THAI.findall(s)) >= min_thai


def clean(s):
    return re.sub(r"\s+", " ", TAG.sub(" ", s or "")).strip()


def _get(url, extra=None):
    h = {"User-Agent": UA}
    if extra:
        h.update(extra)
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=25)


# ---------- YouTube ----------
def fetch_youtube(url, limit):
    import yt_dlp
    opts = {"getcomments": True, "skip_download": True, "quiet": True, "no_warnings": True,
            "extractor_args": {"youtube": {"comment_sort": ["top"], "max_comments": [str(limit * 3)]}}}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=False)
    return [c.get("text", "") for c in (info.get("comments") or [])]


# ---------- Pantip (Thai forum) ----------
def fetch_pantip(url, limit):
    m = re.search(r"pantip\.com/topic/(\d+)", url)
    if not m:
        raise UnsupportedError("Pantip: the link must be pantip.com/topic/<number>")
    tid = m.group(1)
    out = []

    def walk(c):
        if c.get("message"):
            out.append(c["message"])
        for rep in (c.get("replies") or []):
            walk(rep)

    for page in range(1, 8):                       # up to ~7 pages (100/page), enough for the set limit
        u = f"https://pantip.com/forum/topic/render_comments?tid={tid}&param=page{page}"
        try:
            d = json.loads(_get(u, {"X-Requested-With": "XMLHttpRequest",
                                    "Referer": f"https://pantip.com/topic/{tid}"}).read().decode("utf-8-sig"))
        except Exception as e:
            if page == 1:
                raise UnsupportedError(f"Pantip: {type(e).__name__}")
            break
        coms = d.get("comments") or []
        if not coms:
            break
        for c in coms:
            walk(c)
        if len(out) >= limit * 3:
            break
    return out


# ---------- Reddit (best-effort) ----------
def fetch_reddit(url, limit):
    u = url.split("?")[0].rstrip("/")
    if not u.endswith(".json"):
        u += ".json"
    data = json.load(_get(u, {"Accept": "application/json"}))
    out = []

    def walk(node):
        if len(out) >= limit * 4:
            return
        if isinstance(node, dict):
            d = node.get("data", {})
            if node.get("kind") == "t1" and d.get("body"):
                out.append(d["body"])
            if isinstance(d.get("replies"), dict):
                walk(d["replies"])
            for ch in (d.get("children") or []):
                walk(ch)
        elif isinstance(node, list):
            for n in node:
                walk(n)

    walk(data[1] if isinstance(data, list) and len(data) > 1 else data)
    return out


# ---------- dispatch ----------
def platform_of(url):
    u = url.lower()
    if "youtu" in u:
        return "youtube"
    if "pantip.com" in u:
        return "pantip"
    if "reddit.com" in u or "redd.it" in u:
        return "reddit"
    if "twitter.com" in u or "x.com" in u:
        return "twitter"
    if "instagram.com" in u:
        return "instagram"
    if "tiktok.com" in u:
        return "tiktok"
    if "facebook.com" in u or "fb.watch" in u:
        return "facebook"
    return "other"


_FETCHERS = {"youtube": fetch_youtube, "pantip": fetch_pantip, "reddit": fetch_reddit}


def fetch_any(url, limit=80):
    """return (list of unique Thai texts, platform name) · raise UnsupportedError if not freely accessible"""
    plat = platform_of(url)
    fn = _FETCHERS.get(plat)
    if fn is None:
        raise UnsupportedError(plat)          # twitter/instagram/tiktok/facebook/other
    try:
        raw = fn(url, limit)
    except UnsupportedError:
        raise
    except Exception as e:
        raise UnsupportedError(f"{plat}: {type(e).__name__}")

    seen, out = set(), []
    for c in raw:
        c = clean(c)
        if is_thai(c) and 8 <= len(c) <= 400 and c not in seen:
            seen.add(c); out.append(c)
        if len(out) >= limit:
            break
    return out, plat
