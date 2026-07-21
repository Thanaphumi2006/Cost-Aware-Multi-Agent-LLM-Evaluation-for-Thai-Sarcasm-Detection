# -*- coding: utf-8 -*-
"""Render overview.html / app.html into PNG images for embedding in the README

Why images: GitHub shows images in the README directly **without enabling GitHub Pages**
the .html files 404 if Pages is off -> so images are always the path that works immediately
(the .html files are still all there, anyone who clones and opens them can use them; the images are just a preview)

How: use the headless Chrome/Edge already on the machine (no need to install a 300 MB puppeteer)
shoot at a very tall viewport then crop the empty bottom margin with Pillow

Run: python make_images.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

# (source file, output name, width, script injected before the shot)
DEMO_FILL = """
<script>
  document.getElementById('inp').value = 'บริการดีมากกก รอแค่ชั่วโมงเดียวเอง 555';
  render();
</script>
"""
JOBS = [
    ("overview.html", "overview", 1000, ""),
    ("app.html", "app", 1180, DEMO_FILL),
]


def find_browser():
    for b in BROWSERS:
        if os.path.exists(b):
            return b
    sys.exit("Chrome or Edge not found on this machine")


def prepare(src, theme, inject, tmpdir):
    """copy the html with a forced theme + injected script (if any) into a temp folder"""
    html = open(os.path.join(ROOT, src), encoding="utf-8").read()
    # force the theme via data-theme, which the page already supports (no browser flag needed)
    html = re.sub(r"<html([^>]*)>", rf'<html\1 data-theme="{theme}">', html, count=1)
    if inject:
        html = html.replace("</body>", inject + "</body>")
    out = os.path.join(tmpdir, f"{theme}_{src}")
    open(out, "w", encoding="utf-8").write(html)
    return out


def trim(path, bg_probe=(2, 2), pad=28):
    """crop the empty bottom area -- shoot overly tall first, then find the last row with content"""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    bg = im.getpixel(bg_probe)
    px = im.load()
    last = 0
    for y in range(h - 1, -1, -1):
        # scanning every 4 pixels horizontally is enough -- faster and precise enough to find the edge
        if any(px[x, y] != bg for x in range(0, w, 4)):
            last = y
            break
    im.crop((0, 0, w, min(h, last + pad))).save(path, optimize=True)
    return im.size, (w, min(h, last + pad))


def main():
    browser = find_browser()
    os.makedirs(DOCS, exist_ok=True)
    print(f"browser: {os.path.basename(browser)}\n")

    with tempfile.TemporaryDirectory() as tmp:
        for src, name, width, inject in JOBS:
            for theme in ("light", "dark"):
                page = prepare(src, theme, inject, tmp)
                out = os.path.join(DOCS, f"{name}-{theme}.png")
                profile = os.path.join(tmp, f"prof_{name}_{theme}")
                cmd = [
                    browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--user-data-dir={profile}",
                    f"--window-size={width},7000",
                    "--force-device-scale-factor=2",     # retina display -> crisp text
                    "--virtual-time-budget=3000",         # wait for JS/fonts to finish before the shot
                    f"--screenshot={out}",
                    "file:///" + page.replace("\\", "/"),
                ]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if not os.path.exists(out):
                    print(f"  [x] {name}-{theme}: capture failed\n{r.stderr[:300]}")
                    continue
                before, after = trim(out)
                kb = os.path.getsize(out) / 1024
                print(f"  ✓ docs/{name}-{theme}.png  {after[0]}×{after[1]}  {kb:.0f} KB")

    # remove any leftover profile folder
    for d in (DOCS,):
        for junk in ("prof_",):
            for f in os.listdir(d):
                if f.startswith(junk):
                    shutil.rmtree(os.path.join(d, f), ignore_errors=True)
    print(f"\ndone -> {DOCS}")


if __name__ == "__main__":
    main()
