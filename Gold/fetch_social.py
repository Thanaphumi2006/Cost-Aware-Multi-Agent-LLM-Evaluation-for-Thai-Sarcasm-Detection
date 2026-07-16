# -*- coding: utf-8 -*-
"""ดึงคอมเมนต์ไทยจากลิงก์โซเชียล (หลายแพลตฟอร์ม) -> รายการข้อความ พร้อมป้อนตัวตรวจจับ

รองรับจริง (ดึงอัตโนมัติได้):
  - YouTube  : ผ่าน yt-dlp
  - Reddit   : ผ่าน public JSON (เติม .json ท้าย URL) — ไม่ต้องล็อกอิน ไม่ต้อง API key

พยายามแบบ best-effort (yt-dlp รองรับหลายเว็บ) แต่มักไม่ได้:
  - Twitter/X, Instagram, TikTok, Facebook : ต้องล็อกอิน/คุกกี้ หรือ API เสียเงิน -> ส่วนใหญ่ดึงไม่ได้
    ถ้าดึงไม่ได้ -> โยน UnsupportedError ให้ผู้เรียกบอกผู้ใช้ว่า "ก๊อปข้อความมาวางเอง" แทน

หมายเหตุ: เป็นข้อมูลสาธารณะ ดึงมาเพื่อทดสอบโมเดลตัวเอง · กรองเฉพาะที่มีตัวอักษรไทย
"""
import json
import re
import urllib.request

THAI = re.compile(r"[฀-๿]")


class UnsupportedError(Exception):
    """แพลตฟอร์มนี้ดึงอัตโนมัติไม่ได้ (ต้องล็อกอิน/API เสียเงิน)"""


def is_thai(s, min_thai=3):
    return len(THAI.findall(s)) >= min_thai


def clean(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _keep(c, seen):
    c = clean(c)
    return c if (is_thai(c) and 8 <= len(c) <= 400 and c not in seen) else None


# ---------- YouTube (yt-dlp) ----------
def fetch_youtube(url, limit):
    import yt_dlp
    opts = {"getcomments": True, "skip_download": True, "quiet": True, "no_warnings": True,
            "extractor_args": {"youtube": {"comment_sort": ["top"], "max_comments": [str(limit * 3)]}}}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=False)
    return [c.get("text", "") for c in (info.get("comments") or [])]


# ---------- Reddit (public .json) ----------
def fetch_reddit(url, limit):
    u = url.split("?")[0].rstrip("/")
    if not u.endswith(".json"):
        u += ".json"
    req = urllib.request.Request(u, headers={"User-Agent": "thai-sarcasm-eval/1.0 (research)"})
    with urllib.request.urlopen(req, timeout=25) as r:
        data = json.load(r)
    out = []

    def walk(node):
        if len(out) >= limit * 4:
            return
        if isinstance(node, dict):
            d = node.get("data", {})
            if node.get("kind") == "t1" and d.get("body"):     # t1 = comment
                out.append(d["body"])
            replies = d.get("replies")
            if isinstance(replies, dict):
                walk(replies)
            for ch in (d.get("children") or []):
                walk(ch)
        elif isinstance(node, list):
            for n in node:
                walk(n)

    # data = [post_listing, comments_listing]; เดินเฉพาะฝั่งคอมเมนต์
    walk(data[1] if isinstance(data, list) and len(data) > 1 else data)
    return out


# ---------- dispatch ----------
def platform_of(url):
    u = url.lower()
    if "youtu" in u:
        return "youtube"
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


def fetch_any(url, limit=80):
    """คืน list ข้อความไทย (unique) จากลิงก์ · โยน UnsupportedError ถ้าดึงไม่ได้"""
    plat = platform_of(url)
    try:
        if plat == "youtube":
            raw = fetch_youtube(url, limit)
        elif plat == "reddit":
            raw = fetch_reddit(url, limit)
        else:
            # best-effort ผ่าน yt-dlp (บางเว็บได้ text/คอมเมนต์) — ส่วนใหญ่ Twitter/IG จะพัง
            try:
                raw = fetch_youtube(url, limit)
            except Exception:
                raise UnsupportedError(plat)
    except UnsupportedError:
        raise
    except Exception as e:
        raise UnsupportedError(f"{plat}: {type(e).__name__}")

    seen, out = set(), []
    for c in raw:
        k = _keep(c, seen)
        if k:
            seen.add(k); out.append(k)
        if len(out) >= limit:
            break
    return out, plat
