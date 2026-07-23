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
import ipaddress
import json
import re
import socket
import urllib.request
from urllib.parse import urlparse

THAI = re.compile(r"[฀-๿]")
TAG = re.compile(r"<[^>]+>")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

SUPPORTED = ("YouTube", "Pantip", "Reddit")   # shown in the web app as fetchable

# registrable domains we are willing to fetch from, matched against the REAL hostname (not a substring).
# substring matching was an SSRF hole: "http://169.254.169.254/reddit.com" is not reddit -- its host is a
# metadata IP. Parsing the host and matching exact-or-subdomain closes that.
_DOMAINS = {"youtube": ("youtube.com", "youtu.be"),
            "pantip": ("pantip.com",),
            "reddit": ("reddit.com", "redd.it")}


class UnsupportedError(Exception):
    """this platform comments are not freely accessible (needs login / paid API)"""


class FetchError(Exception):
    """a supported platform (YouTube/Pantip/Reddit) that failed for a fixable reason
    (missing yt-dlp, network hiccup, 403, comments off) -- NOT a login/paid-API wall"""


def is_thai(s, min_thai=3):
    return len(THAI.findall(s)) >= min_thai


def clean(s):
    return re.sub(r"\s+", " ", TAG.sub(" ", s or "")).strip()


def _assert_public_host(url):
    """defence in depth for the direct-urlopen fetchers (pantip/reddit): even though the host is
    already domain-allowlisted, refuse if it resolves to a private/loopback/link-local address
    (guards against a DNS answer pointing inward). Raises FetchError on anything non-public."""
    host = _host_of(url)
    if not host:
        raise FetchError("bad url")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise FetchError("cannot resolve host")
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise FetchError("refusing to fetch a non-public address")


def _get(url, extra=None):
    _assert_public_host(url)
    h = {"User-Agent": UA}
    if extra:
        h.update(extra)
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=25)


# ---------- YouTube ----------
# YouTube now gates playback behind a Proof-of-Origin (PO) token: without one, every
# client returns UNPLAYABLE / "The page needs to be reloaded". The fix is (a) use the
# "web" client, which consumes a PO token, and (b) run a local PO-token provider
# (bgutil, on 127.0.0.1:4416) that yt-dlp's bundled plugin talks to automatically.
POT_URL = "http://127.0.0.1:4416/ping"


def _pot_provider_up():
    try:
        urllib.request.urlopen(POT_URL, timeout=2)
        return True
    except Exception:
        return False


def fetch_youtube(url, limit):
    try:
        import yt_dlp
    except ImportError:
        raise FetchError("YouTube needs yt-dlp -- run: pip install yt-dlp")
    opts = {"getcomments": True, "skip_download": True, "quiet": True, "no_warnings": True,
            "ignore_no_formats_error": True,        # we only want comments -> don't fail if no video format is offered
            "extractor_args": {"youtube": {"player_client": ["web"],       # the client that uses a PO token
                                           "comment_sort": ["top"], "max_comments": [str(limit * 3)]}}}
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(url, download=False)
    except Exception:
        # the usual cause today is a missing PO token -> point the user at the helper instead of a bare error
        if not _pot_provider_up():
            raise FetchError("YouTube needs the PO-token helper running (bgutil on port 4416). "
                             "Pantip works without it.")
        raise
    return [c.get("text", "") for c in (info.get("comments") or [])]


# ---------- Pantip (Thai forum) ----------
def fetch_pantip(url, limit):
    m = re.search(r"pantip\.com/topic/(\d+)", url)
    if not m:
        raise FetchError("Pantip: the link must be pantip.com/topic/<number>")
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
                raise FetchError(f"Pantip: {type(e).__name__}")
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
def _host_of(url):
    return (urlparse(url).hostname or "").lower().rstrip(".")


def _matches(host, domains):
    return any(host == d or host.endswith("." + d) for d in domains)


def platform_of(url):
    """map a URL to a supported platform by its real hostname (exact or subdomain match).
    anything else -> 'other' (raised as UnsupportedError). This is a security boundary, not just UX."""
    if urlparse(url).scheme not in ("http", "https"):
        return "other"
    host = _host_of(url)
    for plat, domains in _DOMAINS.items():
        if _matches(host, domains):
            return plat
    return "other"


_FETCHERS = {"youtube": fetch_youtube, "pantip": fetch_pantip, "reddit": fetch_reddit}


def fetch_any(url, limit=80):
    """return (list of unique Thai texts, platform name) · raise UnsupportedError if not freely accessible"""
    plat = platform_of(url)
    fn = _FETCHERS.get(plat)
    if fn is None:
        raise UnsupportedError(plat)          # twitter/instagram/tiktok/facebook/other: login/paid-API wall
    try:
        raw = fn(url, limit)
    except (UnsupportedError, FetchError):
        raise
    except Exception as e:                    # supported platform, fixable failure -> not a login wall
        raise FetchError(f"{plat}: {type(e).__name__}")

    seen, out = set(), []
    for c in raw:
        c = clean(c)
        if is_thai(c) and 8 <= len(c) <= 400 and c not in seen:
            seen.add(c); out.append(c)
        if len(out) >= limit:
            break
    return out, plat
