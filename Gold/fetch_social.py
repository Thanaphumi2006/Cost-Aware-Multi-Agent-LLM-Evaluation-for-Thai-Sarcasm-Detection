# -*- coding: utf-8 -*-
"""ดึงคอมเมนต์ไทยจากลิงก์ — เฉพาะแพลตฟอร์มที่ "ฟรี ไม่ต้องล็อกอิน" เข้าถึงคอมเมนต์ได้จริง

รองรับ (ฟรี ไม่ต้อง API key / ไม่ต้องล็อกอิน):
  - YouTube : yt-dlp (ชัวร์)
  - Pantip  : public JSON ของเว็บเอง (Thai forum — ประชดเยอะ เหมาะกับงานนี้ที่สุด)
  - Reddit  : public .json (ฟรี แต่บาง IP โดนบล็อก 403 -> best-effort)

แพลตฟอร์มที่ "ตัดออก" เพราะเข้าถึงคอมเมนต์ฟรีไม่ได้ (บังคับล็อกอิน/คุกกี้/หรือ API เสียเงิน):
  Twitter/X, Instagram, TikTok, Facebook -> โยน UnsupportedError ให้ผู้เรียกบอกผู้ใช้ "วางเอง"

ข้อมูลสาธารณะ ดึงมาเพื่อทดสอบโมเดลตัวเอง · กรองเฉพาะที่มีตัวอักษรไทย
"""
import json
import re
import urllib.request

THAI = re.compile(r"[฀-๿]")
TAG = re.compile(r"<[^>]+>")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

SUPPORTED = ("YouTube", "Pantip", "Reddit")   # ที่โชว์ในเว็บว่าดึงได้


class UnsupportedError(Exception):
    """แพลตฟอร์มนี้เข้าถึงคอมเมนต์ฟรีไม่ได้ (ต้องล็อกอิน/API เสียเงิน)"""


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
        raise UnsupportedError("Pantip: ลิงก์ต้องเป็น pantip.com/topic/<เลข>")
    tid = m.group(1)
    out = []

    def walk(c):
        if c.get("message"):
            out.append(c["message"])
        for rep in (c.get("replies") or []):
            walk(rep)

    for page in range(1, 8):                       # สูงสุด ~7 หน้า (100/หน้า) พอสำหรับ limit ที่ตั้งไว้
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
    """คืน (list ข้อความไทย unique, ชื่อแพลตฟอร์ม) · โยน UnsupportedError ถ้าเข้าถึงฟรีไม่ได้"""
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
