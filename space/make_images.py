# -*- coding: utf-8 -*-
"""เรนเดอร์ overview.html / app.html เป็นไฟล์ภาพ PNG สำหรับฝังใน README

ทำไมต้องมีภาพ: GitHub แสดงรูปใน README ได้เลย **โดยไม่ต้องเปิด GitHub Pages**
ส่วนไฟล์ .html ถ้า Pages ไม่ได้เปิด ลิงก์จะ 404 -> ภาพจึงเป็นทางที่ทำงานได้ทันทีเสมอ
(ตัว .html ยังอยู่ครบ ใครโคลนไปเปิดเองก็ยังกดใช้ได้จริง ภาพเป็นแค่ตัวอย่างให้เห็นหน้าตา)

วิธีทำ: ใช้ Chrome/Edge headless ที่มีอยู่แล้วในเครื่อง (ไม่ต้องลง puppeteer 300 MB)
ถ่ายที่ viewport สูงมากแล้วค่อยตัดขอบล่างที่ว่างเปล่าทิ้งด้วย Pillow

รัน: python make_images.py
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

# (ไฟล์ต้นทาง, ชื่อผลลัพธ์, กว้าง, สคริปต์ที่ inject ก่อนถ่าย)
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
    sys.exit("ไม่พบ Chrome หรือ Edge ในเครื่อง")


def prepare(src, theme, inject, tmpdir):
    """ทำสำเนา html พร้อมบังคับธีม + inject สคริปต์ (ถ้ามี) ไว้ในโฟลเดอร์ชั่วคราว"""
    html = open(os.path.join(ROOT, src), encoding="utf-8").read()
    # บังคับธีมผ่าน data-theme ที่หน้าเว็บรองรับอยู่แล้ว (ไม่ต้องพึ่ง flag ของเบราว์เซอร์)
    html = re.sub(r"<html([^>]*)>", rf'<html\1 data-theme="{theme}">', html, count=1)
    if inject:
        html = html.replace("</body>", inject + "</body>")
    out = os.path.join(tmpdir, f"{theme}_{src}")
    open(out, "w", encoding="utf-8").write(html)
    return out


def trim(path, bg_probe=(2, 2), pad=28):
    """ตัดพื้นที่ว่างด้านล่างทิ้ง -- ถ่ายมาสูงเกินไว้ก่อน แล้วค่อยหาแถวสุดท้ายที่มีเนื้อหา"""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    bg = im.getpixel(bg_probe)
    px = im.load()
    last = 0
    for y in range(h - 1, -1, -1):
        # สแกนทีละ 4 พิกเซลตามแนวนอนก็พอ -- เร็วกว่าและแม่นพอสำหรับหาขอบ
        if any(px[x, y] != bg for x in range(0, w, 4)):
            last = y
            break
    im.crop((0, 0, w, min(h, last + pad))).save(path, optimize=True)
    return im.size, (w, min(h, last + pad))


def main():
    browser = find_browser()
    os.makedirs(DOCS, exist_ok=True)
    print(f"เบราว์เซอร์: {os.path.basename(browser)}\n")

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
                    "--force-device-scale-factor=2",     # จอ retina -> ตัวหนังสือคม
                    "--virtual-time-budget=3000",         # รอ JS/ฟอนต์ทำงานให้เสร็จก่อนถ่าย
                    f"--screenshot={out}",
                    "file:///" + page.replace("\\", "/"),
                ]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if not os.path.exists(out):
                    print(f"  ✗ {name}-{theme}: ถ่ายไม่สำเร็จ\n{r.stderr[:300]}")
                    continue
                before, after = trim(out)
                kb = os.path.getsize(out) / 1024
                print(f"  ✓ docs/{name}-{theme}.png  {after[0]}×{after[1]}  {kb:.0f} KB")

    # ลบโฟลเดอร์ profile ที่อาจหลงเหลือ
    for d in (DOCS,):
        for junk in ("prof_",):
            for f in os.listdir(d):
                if f.startswith(junk):
                    shutil.rmtree(os.path.join(d, f), ignore_errors=True)
    print(f"\nเสร็จ -> {DOCS}")


if __name__ == "__main__":
    main()
