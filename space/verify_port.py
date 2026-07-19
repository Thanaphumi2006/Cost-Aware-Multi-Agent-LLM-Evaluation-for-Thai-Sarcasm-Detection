# -*- coding: utf-8 -*-
"""ตรวจว่า app.html (JavaScript) ให้ผลตรงกับ space/app.py (Python) เป๊ะ

ทำไมต้องมี: หน้า static ที่ deploy จริง (app.html) เป็นการ *พอร์ตมือ* จากตัว Python
ถ้าสองฝั่งไม่ตรงกัน = หน้าเว็บที่คนใช้จริงจะให้คำตอบคนละอย่างกับตัวเลขที่รายงานในงานวิจัย
สคริปต์นี้ดึงตรรกะจริงออกจาก app.html (ไม่ใช่ก๊อปมาเขียนซ้ำ) แล้วรันด้วย node เทียบทีละข้อ

รัน: python verify_port.py     (ต้องมี node ในเครื่อง)
"""
import json
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(os.path.dirname(HERE), "app.html")

sys.path.insert(0, HERE)
import app as pyapp  # noqa: E402

# ประโยคทดสอบ: ครอบคลุมทั้งสามผลลัพธ์ + เคสขอบ
TESTS = [
    "บริการดีมากกก รอแค่ชั่วโมงเดียวเอง 555",
    "อาหารอร่อยมากค่ะ พนักงานน่ารัก จะกลับมาอีกแน่นอน",
    "ขอบคุณที่ยกเลิกออเดอร์ตอนรอมา 2 ชั่วโมง จ้า",
    "ราคาสมเหตุสมผล ของสดใหม่ แนะนำครับ",
    "ดีจังเลย ฝนตกตอนลืมร่มพอดี",
    "วันนี้อากาศดี ไปเดินเล่นสวนสาธารณะมา",
    "เก่งมากเลยนะ ทำพังได้ทุกครั้ง 555",
    "จริงเหรอ?? ราคานี้เนี่ยนะ",
    "ขอบคุณค่ะ สะดวกมากนะคะ",
    "โอ้โหหหห ดีจริงๆ ครับ 555 ??",   # cue ชนกันหลายทิศ
    "อะไรนะ",                          # ไม่มี cue เลย
    "555",                             # cue เดียวล้วน
]


def extract_js():
    """ดึง CUES + findCues + cueScore ออกจาก app.html โดยตรง
    (ตั้งใจอ่านจากไฟล์จริงที่ deploy ไม่ใช่เขียน logic ซ้ำ -- ไม่งั้นก็ไม่ได้ทดสอบของจริง)"""
    src = open(HTML, encoding="utf-8").read()
    m = re.search(r"const CUES = \[.*?\];", src, re.S)
    f1 = re.search(r"const findCues = .*?;", src, re.S)
    f2 = re.search(r"const cueScore = .*?;", src, re.S)
    if not (m and f1 and f2):
        sys.exit("ดึง logic จาก app.html ไม่สำเร็จ -- โครงสร้างไฟล์เปลี่ยนไปหรือเปล่า?")
    return "\n".join([m.group(0), f1.group(0), f2.group(0)])


def run_js(tests):
    js = extract_js() + """
const out = TESTS.map(t => {
  const hits = findCues(t);
  const s = cueScore(hits);
  const verdict = hits.length === 0 ? "unknown" : (s > 0 ? "sarcastic" : "genuine");
  return { score: Number(s.toFixed(6)), cues: hits.map(c => c.name), verdict };
});
console.log(JSON.stringify(out));
"""
    js = "const TESTS = " + json.dumps(tests, ensure_ascii=False) + ";\n" + js
    p = subprocess.run(["node", "-e", js], capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        sys.exit(f"node ล้มเหลว:\n{p.stderr}")
    return json.loads(p.stdout)


def py_result(t):
    hits = pyapp.find_cues(t)
    s = pyapp.cue_score(hits)
    verdict = "unknown" if not hits else ("sarcastic" if s > 0 else "genuine")
    return {"score": round(s, 6), "cues": [h[0] for h in hits], "verdict": verdict}


def main():
    js = run_js(TESTS)
    bad = 0
    print(f"เทียบ Python (space/app.py) กับ JavaScript (app.html) · {len(TESTS)} ประโยค\n")
    print(f"{'':2} {'verdict':<10} {'score':>8}  ข้อความ")
    print("-" * 74)
    for t, j in zip(TESTS, js):
        p = py_result(t)
        same = (p["verdict"] == j["verdict"]
                and abs(p["score"] - j["score"]) < 1e-6
                and p["cues"] == j["cues"])
        bad += not same
        print(f"{'ok' if same else '!!':2} {p['verdict']:<10} {p['score']:>8.3f}  {t[:40]}")
        if not same:
            print(f"     PY  {p}")
            print(f"     JS  {j}")
    print("-" * 74)
    if bad:
        sys.exit(f"\n❌ ไม่ตรงกัน {bad}/{len(TESTS)} ข้อ -- หน้าเว็บจะให้คำตอบคนละอย่างกับตัว Python")
    print(f"\n✅ ตรงกันทั้ง {len(TESTS)}/{len(TESTS)} ข้อ (verdict + คะแนน + cue ที่เจอ)")


if __name__ == "__main__":
    main()
