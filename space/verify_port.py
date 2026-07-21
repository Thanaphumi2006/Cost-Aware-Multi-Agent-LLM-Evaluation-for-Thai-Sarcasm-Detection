# -*- coding: utf-8 -*-
"""Verify that app.html (JavaScript) matches space/app.py (Python) exactly

Why it exists: the deployed static page (app.html) is a *hand-port* of the Python
if the two diverge = the page people actually use gives different answers than the numbers reported in the research
this script extracts the real logic from app.html (not a re-copy) and runs it with node, comparing item by item

Run: python verify_port.py     (needs node installed)
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

# test sentences: cover all three outcomes + edge cases
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
    "โอ้โหหหห ดีจริงๆ ครับ 555 ??",   # cues collide in several directions
    "อะไรนะ",                          # no cue at all
    "555",                             # a single cue only
]


def extract_js():
    """extract CUES + findCues + cueScore directly from app.html
    (deliberately reads the real deployed file rather than re-implementing the logic -- otherwise it is not testing the real thing)"""
    src = open(HTML, encoding="utf-8").read()
    m = re.search(r"const CUES = \[.*?\];", src, re.S)
    f1 = re.search(r"const findCues = .*?;", src, re.S)
    f2 = re.search(r"const cueScore = .*?;", src, re.S)
    if not (m and f1 and f2):
        sys.exit("failed to extract logic from app.html -- did the file structure change?")
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
        sys.exit(f"node failed:\n{p.stderr}")
    return json.loads(p.stdout)


def py_result(t):
    hits = pyapp.find_cues(t)
    s = pyapp.cue_score(hits)
    verdict = "unknown" if not hits else ("sarcastic" if s > 0 else "genuine")
    return {"score": round(s, 6), "cues": [h[0] for h in hits], "verdict": verdict}


def main():
    js = run_js(TESTS)
    bad = 0
    print(f"comparing Python (space/app.py) with JavaScript (app.html) · {len(TESTS)} sentences\n")
    print(f"{'':2} {'verdict':<10} {'score':>8}  text")
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
        sys.exit(f"\n[FAIL] mismatch on {bad}/{len(TESTS)} items -- the page will give different answers than the Python")
    print(f"\n[OK] all {len(TESTS)}/{len(TESTS)} items match (verdict + score + cues found)")


if __name__ == "__main__":
    main()
