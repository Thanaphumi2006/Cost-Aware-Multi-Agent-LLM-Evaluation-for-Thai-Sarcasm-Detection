# -*- coding: utf-8 -*-
"""Does WCB_NEG=0.17 (calibrated on out-of-fold probs) still separate anything on the DEPLOYED model?
The deployed model trained on all 127 gold items, so its gold probs are memorised -- shown only to
see the scale. The honest signal is the held-out-ish text at the bottom."""
import math, os, re, sys
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

D = os.path.dirname(os.path.abspath(__file__)) + "/"   # this Gold/ folder
tok = AutoTokenizer.from_pretrained(D + "wcb_model")
mdl = AutoModelForSequenceClassification.from_pretrained(D + "wcb_model"); mdl.eval()

def p(texts):
    out = []
    for i in range(0, len(texts), 16):
        with torch.no_grad():
            e = tok(texts[i:i+16], truncation=True, padding=True, max_length=256, return_tensors="pt")
            out += torch.softmax(mdl(**e).logits, -1)[:, 1].tolist()
    return np.array(out)

CUES = [("555", re.compile(r"555"), 2.46), ("??", re.compile(r"[?]{2,}"), 2.54),
        ("ยืด", re.compile(r"(.)\1{2,}"), 1.69), ("จ้า", re.compile(r"จ้า"), 1.32),
        ("ค่ะ", re.compile(r"ค่ะ"), 0.40), ("นะคะ", re.compile(r"นะคะ"), 0.22),
        ("ครับ", re.compile(r"ครับ"), 0.05)]
CUT = math.log(2.46)
def cue_defers(t):
    h = [l for _, rx, l in CUES if rx.search(t)]
    return (not h) or abs(sum(math.log(max(l, 0.05)) for l in h)) < CUT

g = pd.read_csv(D + "gold.csv"); g.columns = [c.lstrip("﻿") for c in g.columns]
g["defer"] = [cue_defers(t) for t in g.text]
sub = g[g.defer].reset_index(drop=True)          # the 53 items that reach tier 2
pr = p(sub.text.tolist())
y = sub.label.values

print("=== DEPLOYED model on the 53 cue-deferred gold items (MEMORISED -- scale check only) ===")
print(f"  min={pr.min():.4f}  p10={np.percentile(pr,10):.4f}  median={np.percentile(pr,50):.4f}  "
      f"p90={np.percentile(pr,90):.4f}  max={pr.max():.4f}")
for cut in [0.17, 0.05, 0.02, 0.01]:
    m = pr < cut
    print(f"  cut {cut:<5} -> answers {m.sum():2d}/53   of those, truly-0: {int((y[m]==0).sum())}/{int(m.sum() or 0)}")

print("\n=== fresh text the model has never seen (the honest check) ===")
fresh = [
 ("ดีจริงๆ รอของ 3 อาทิตย์ ได้ของพังมา ขอบคุณมาก", 1),
 ("บริการเยี่ยม พนักงานยิ้มแย้ม ประทับใจ", 0),
 ("โอ้โห เก่งจังเลยนะ ทำพังได้ทุกครั้ง", 1),
 ("ส่งของไวมาก ของครบ แพ็คดี", 0),
 ("ขอบคุณที่ทำให้วันนี้แย่ลงกว่าเดิม", 1),
 ("ร้านสะอาด อาหารรสชาติกำลังดี", 0),
 ("เยี่ยมไปเลย ระบบล่มอีกแล้ว", 1),
 ("ราคาถูกและคุณภาพดี คุ้มค่ามาก", 0),
 ("นโยบายนี้ดีต่อประชาชนมากจริงๆ นะ", 1),
 ("วันนี้อากาศดี ไปเดินเล่นสวนสาธารณะมา", 0),
]
ft = [t for t, _ in fresh]; fy = np.array([l for _, l in fresh]); fp = p(ft)
print(f"  range: min={fp.min():.4f}  max={fp.max():.4f}")
for (t, l), q in zip(fresh, fp):
    flag = "TIER2 answers 'not sarcastic'" if q < 0.17 else "-> defers to LLM"
    ok = "" if q >= 0.17 else ("  OK" if l == 0 else "  ** WRONG, real label=1 **")
    print(f"   P={q:.4f}  true={l}  {flag}{ok}   {t[:42]}")
n_ans = int((fp < 0.17).sum())
print(f"\n  tier 2 fires on {n_ans}/10 fresh items at cut 0.17")
if n_ans:
    wrong = int(((fp < 0.17) & (fy == 1)).sum())
    print(f"  of those, wrong (真 sarcastic wrongly cleared): {wrong}")
