# Cost-Aware Multi-Agent LLM Evaluation for Thai Sarcasm Detection

ระบบหลายเอเจนต์ (multi-agent) คุ้มค่ากว่าเอเจนต์เดี่ยวจริงหรือไม่ สำหรับงานตรวจจับ**ประชด/เสียดสีภาษาไทย**?
งานนี้วัดคำตอบด้วย 4 มิติพร้อมกัน: **คุณภาพ (F1) · ค่าใช้จ่าย · latency · จำนวน LLM calls**

ทุกระบบวัดบน gold ชุดเดียวกัน (127 ข้อ: ประชด 30 / ไม่ประชด 97), ใช้ GPT-4o และ harness เดียวกัน
เปรียบเทียบด้วย **paired bootstrap (5,000 รอบ) + McNemar** เพราะทุกระบบรันบนข้อมูลชุดเดียวกัน

## ผลการทดลอง

| ระบบ | F1 | precision | recall | LLM calls | ค่าใช้จ่าย | latency p50 |
|---|---|---|---|---|---|---|
| ① เอเจนต์เดี่ยว (baseline) | 0.690 | 0.526 | **1.000** | 127 | $0.094 | 751 ms |
| ② **pipeline v2 — ผู้คัดกรอง → ผู้ตรวจสอบ** ⭐ | **0.744** | 0.604 | 0.967 | 183 | $0.169 | 967 ms |
| ③ pipeline v1 (verifier พลิกอิสระ) | 0.714 | **0.769** | 0.667 | 180 | $0.157 | 721 ms |
| ④ debate (อัยการ + ทนาย + ผู้พิพากษา) | 0.694 | 0.595 | 0.833 | 381 | $0.695 | 4,557 ms |
| ⑤ hybrid (คัดกรอง + คณะโต้แย้ง 4 เอเจนต์) | 0.700 | 0.560 | 0.933 | 292 | $0.407 | 832 ms |
| ⑥ WangchanBERTa (5-fold CV × 3 seeds) | 0.620 ±0.005 | 0.553 | 0.700 | **0** | **$0.00** | **26 ms** |

## ข้อค้นพบหลัก

> **การจำกัดอำนาจของเอเจนต์ให้ถูกต้อง สำคัญกว่าจำนวนเอเจนต์หรือความลึกของการถกเถียง**

ระบบที่ชนะคือระบบที่**เรียบง่ายและถูกที่สุด** — เอเจนต์ตัวที่สองทำได้อย่างเดียวคือ**ปัดตก** (พลิก 1→0)
จึงรักษา recall = 1.000 ที่ตัวคัดกรองได้มาฟรีไว้เกือบครบ แล้วค่อยๆ ซื้อ precision

หลักฐาน 3 เส้นที่ชี้ไปทางเดียวกัน:
1. **v1 vs v2** — ระบบเดียวกัน ต่างแค่กฎ "เวลาไม่แน่ใจให้ทำยังไง" → recall 0.667 vs 0.967
2. **pipeline vs debate** — เอเจนต์ 3 เท่า จ่ายแพงกว่า 4.1× ช้ากว่า 4.7× แต่ **แพ้** (0.694 vs 0.744)
3. **hybrid** — ให้ถกเถียงกันภายใต้กรอบที่จำกัด ก็ยัง **แพ้** (0.700) เพราะผู้พิพากษาลังเลขึ้น

รายละเอียดทั้งหมด (พร้อม CI, McNemar, และข้อควรระวัง) → **[`Gold/RESULTS.md`](Gold/RESULTS.md)**

## ข้อควรระวัง (อ่านก่อนอ้างอิงตัวเลข)

- **ห้ามใช้ accuracy** — baseline ได้ 0.787 ซึ่งเกือบเท่ากับเดา "ไม่ประชด" ทุกข้อ (0.764) เพราะข้อมูลเอียง
- **self-selection bias**: ข้อประชดใน gold ส่วนหนึ่งถูกขุดมาด้วย GPT-4o → recall ของ GPT-4o อาจสูงเกินจริง
  แต่ bias นี้กระทบทุกระบบเท่ากัน การ *เปรียบเทียบ* จึงยังยุติธรรม (ดู [`Gold/PROVENANCE.md`](Gold/PROVENANCE.md))
- **n ประชด = 30** เท่านั้น → ทุกข้อสรุปต้องแนบ CI ไม่ใช่ point estimate เดี่ยว

## โครงสร้าง

```
Gold/
  gold.csv                 ชุดวัดผล 127 ข้อ (ที่มา: Wongnai + Wisesight)
  labeling_rubric.md       เกณฑ์ติดป้าย -- ประชดต้องมี "การเสแสร้ง"
  baseline.py              ① เอเจนต์เดี่ยว
  multiagent.py            ② + ③ pipeline (detector -> verifier)
  multiagent_debate.py     ④ debate
  multiagent_hybrid.py     ⑤ hybrid
  wangchanberta.py         ⑥ โมเดลเล็ก (5-fold CV -- ตัวเลขที่ใช้รายงาน)
  compare_systems.py       paired bootstrap + McNemar
  app.py                   เว็บทดลอง/เทียบระบบสดๆ
  *_preds_*.csv            ผลทำนายรายข้อของทุกระบบ (ตรวจย้อนได้)
  RESULTS.md REPORT.md SLIDES.md
```

## วิธีรัน

```powershell
pip install openai pandas numpy scikit-learn flask torch transformers sentencepiece protobuf
$env:OPENAI_API_KEY="sk-..."     # ห้าม commit คีย์ลง repo

cd Gold
python baseline.py               # ① เอเจนต์เดี่ยว
python multiagent.py             # ② pipeline
python compare_systems.py        # เทียบทุกระบบ + สถิติ
python app.py                    # เว็บที่ http://127.0.0.1:5000
```

**ไม่มีอยู่ใน repo** (สร้างใหม่ได้): `Gold/wcb_model/` (401 MB — รัน `train_final_wcb.py`)
และข้อมูลดิบ `raw_texts.csv` / `scored_texts.csv` (~230 MB) ซึ่งไม่จำเป็นต่อการทำซ้ำผลการทดลอง
