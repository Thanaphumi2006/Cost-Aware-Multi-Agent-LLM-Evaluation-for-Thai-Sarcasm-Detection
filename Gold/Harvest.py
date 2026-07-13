# -*- coding: utf-8 -*-
"""
ขุด "ผู้ต้องสงสัยประชด" เพิ่ม ด้วย LLM (แม่นกว่าคีย์เวิร์ด) เพื่อเพิ่มฝั่งประชดใน gold

ทำไมต้องมี: คีย์เวิร์ดเจอประชดจริงน้อย (รีวิว Wongnai ยาวๆ ปนเยอะ)
วิธีแก้: เน้น Wisesight + ตัดข้อความยาว + ให้ LLM คัดเฉพาะที่น่าจะประชด
ผลลัพธ์: harvest_to_review.csv = กองที่ LLM คิดว่าน่าจะประชด เรียงจากมั่นใจสุด
         -> คุณเอาไปตรวจด้วยมือ (blind) เฉพาะกองนี้ เจอประชดเร็วขึ้นมาก

อินพุต : to_label.csv (ต้นทาง) + gold.csv (ข้อที่ตรวจแล้ว จะได้ไม่ซ้ำ)
ติดตั้ง:  pip install openai pandas
ตั้งคีย์:  export OPENAI_API_KEY="sk-..."
รัน:       python harvest_positives_llm.py
"""

import os, json, time
import pandas as pd
from openai import OpenAI

# path แบบทนทาน: อ้างอิงตำแหน่งไฟล์สคริปต์ (รันจากที่ไหนก็ได้)
HERE = os.path.dirname(os.path.abspath(__file__))       # โฟลเดอร์ Gold/
BASE = os.path.dirname(HERE)                            # โฟลเดอร์โปรเจกต์

# ============ ปรับได้ ============
SRC_CSV = os.path.join(BASE, "scored_texts.csv")  # ขุดจากกองใหญ่ 68k (ไม่ใช่แค่ 400)
GOLD_CSV = os.path.join(HERE, "gold.csv")         # ใช้กันข้อที่ตรวจไปแล้ว
OUT_CSV = os.path.join(HERE, "harvest_to_review.csv")
MODEL = "gpt-4o"
ONLY_WISESIGHT = True                 # เน้นแหล่งที่ประชดหนาแน่น
MAX_LEN = 150                         # ตัดข้อความยาว (ประชดมักสั้น กระชับ)
MAX_SCAN = 800                        # สแกนกี่ข้อ (คุมค่า API)
SLEEP = 0.3
# =================================

client = OpenAI()

SYSTEM = """ดูข้อความไทยแล้วตอบว่า "น่าจะประชด/เสียดสี" หรือไม่
ประชด = ผิวเผินชม/ขอบคุณ แต่เจตนาจริงคือเหน็บ/บ่น (ความหมายจริงต่างจากผิวเผิน)
ตำหนิตรงๆ หรือชมจริงใจ = ไม่ประชด
ตอบ JSON เท่านั้น: {"maybe_sarcasm": true/false, "conf": 0.0-1.0}"""

def judge(text):
    # retry กันเน็ตหลุด/ลิมิตชั่วคราว (ไม่ให้ทั้งงานพังเพราะ blip เดียว)
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL, max_tokens=40,
                response_format={"type": "json_object"},
                messages=[{"role":"system","content":SYSTEM},
                          {"role":"user","content":f"ข้อความ: {text}"}],
            )
            o = json.loads(r.choices[0].message.content)
            return bool(o.get("maybe_sarcasm", False)), float(o.get("conf", 0))
        except json.JSONDecodeError:
            return False, 0.0
        except Exception as e:
            if attempt == 3:
                print(f"    (ข้าม 1 ข้อ หลัง retry: {type(e).__name__})")
                return False, 0.0
            time.sleep(2 * (attempt + 1))   # 2s, 4s, 6s

# ---- โหลด + กันข้อที่ตรวจแล้ว ----
df = pd.read_csv(SRC_CSV)
reviewed = set()
if os.path.exists(GOLD_CSV):
    reviewed = set(pd.read_csv(GOLD_CSV)["text"].astype(str))

df["text"] = df["text"].astype(str)
cand = df[~df["text"].isin(reviewed)]
if ONLY_WISESIGHT and "source" in cand.columns:
    cand = cand[cand["source"] == "wisesight"]
cand = cand[cand["text"].str.len() <= MAX_LEN]
# เรียงจาก suspect_score สูงก่อน (มีโอกาสประชดกว่า) แล้วจำกัดจำนวนสแกน
if "suspect_score" in cand.columns:
    cand = cand.sort_values("suspect_score", ascending=False)
cand = cand.head(MAX_SCAN).reset_index(drop=True)
print(f"จะสแกน {len(cand)} ข้อด้วย LLM...")

rows = []
for i, t in enumerate(cand["text"], 1):
    maybe, conf = judge(t)
    if maybe:
        rows.append({"text": t, "llm_conf": conf})
    if i % 25 == 0:
        print(f"  ...{i}/{len(cand)}  พบผู้ต้องสงสัย {len(rows)}")
        if rows:  # เซฟบางส่วนกันงานหาย
            pd.DataFrame(rows).sort_values("llm_conf", ascending=False).to_csv(
                OUT_CSV, index=False, encoding="utf-8-sig")
    time.sleep(SLEEP)

out = pd.DataFrame(rows).sort_values("llm_conf", ascending=False)
out["label"] = ""   # ช่องให้คุณตรวจด้วยมือ
out["note"] = ""
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nพบผู้ต้องสงสัยประชด {len(out)} ข้อ -> {OUT_CSV}")
print("ขั้นต่อไป: ตรวจด้วยมือเฉพาะกองนี้ (blind) แล้วรวมข้อที่เป็น 1 เข้า gold.csv")