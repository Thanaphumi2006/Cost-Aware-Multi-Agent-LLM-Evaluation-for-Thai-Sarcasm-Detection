# Distillation — teacher (GPT) สอน student (WangchanBERTa) ได้ไหม?

**สถานะ: finding-in-progress** — pipeline พร้อมและ dry-check ผ่าน (wiring OK ไม่ใช้ API) ยังไม่ได้รันจริงบน pool จริง

คำถามวิจัย (finding ที่สองของเปเปอร์): *"ระบบ multi-agent/GPT ที่แพง สอนโมเดลเล็กฟรี ให้เก่งขึ้นได้ไหม"*

## ที่มา (ช่องว่างที่จะปิด)
| ระบบ | F1 | precision | recall | ต้นทุน/ข้อ |
|---|---|---|---|---|
| WangchanBERTa (baseline, ไม่มี silver) | ~0.62 | 0.55 | 0.70 | **$0** |
| deployed single-agent (gpt-4.1-mini) | ~0.72 | 0.68 | 0.83 | ~$0.0001 |
| teacher pipeline (gpt-4o detector→verifier) | ~0.74 | 0.60 | 0.97 | ~$0.0006 |

ถ้า distillation ปิดช่องว่าง 0.62→0.74 ได้แม้ครึ่งเดียว = ได้คุณภาพใกล้ pipeline ที่ต้นทุนต่อข้อ = 0

## Pipeline (4 ขั้น — ทุกสคริปต์ไม่ผูกกับแหล่งข้อมูล)
```
fetch_to_csv.py  <urls>          -> pool.csv        ดึงคอมเมนต์ไทย (ทุกแพลตฟอร์มที่ fetch_social รองรับ)
batch_eval.py    --csv pool.csv  -> pool_pred.csv   teacher ติดป้าย (Batch API ลดครึ่งราคา)
distill_label.py --pred ...      -> silver.csv      กรองเฉพาะข้อที่ teacher มั่นใจ (ทิ้งช่วงกลาง)
distill_train_eval.py --silver   -> OOF F1 vs 0.62  เทรน WCB + วัดผลแบบไม่ leak
```

## การออกแบบที่ซื่อสัตย์ (ทำไมถึงทำแบบนี้)
1. **กรองความมั่นใจ** (`--pos-conf`/`--neg-conf`): teacher precision จำกัด (~0.68) → ป้าย silver มี noise
   โดยเฉพาะช่วงก้ำกึ่ง เก็บเฉพาะสองหาง (มั่นใจสูง) ทิ้งกลาง → ลด noise ที่จะเข้าไปในน้ำหนัก student
2. **teacher = single-agent ไม่ใช่ pipeline v2**: pipeline ให้ป้าย hard ไม่มีความมั่นใจ (กรอง noise ไม่ได้)
   ส่วน single-agent มี logprob (คัดความมั่นใจได้) + precision สูงกว่า (0.68 vs 0.60) = teacher ที่ดีกว่าสำหรับ distill
3. **วัดผลไม่ leak (สำคัญสุด)**: 5-fold OOF เดียวกับ `wangchanberta.py` — silver ใส่ทุก training fold,
   วัดผลบน gold fold ที่โมเดลไม่เคยเห็น → เลข F1 เทียบกับ baseline 0.62 ได้ตรงๆ ไม่ใช่คะแนนปลอมจากการจำ
   (reuse `train_one_fold` → พฤติกรรมเทรนเหมือน baseline เป๊ะ ต่างแค่ "ข้อมูลเทรน")
4. **ดึงจากโดเมนเป้าหมาย**: failure จริงคือ cross-domain (precision 0.68→0.40 บน Pantip)
   silver จากโดเมนที่จะ deploy (เว็บบอร์ด) = สอน student ให้รู้จักโดเมนนั้น = ตรงจุดที่พัง
   silver จาก Wongnai/Wisesight เดิม = ตอกย้ำสิ่งที่โมเดลรู้อยู่แล้ว ไม่ช่วย cross-domain

## วิธีอ่านผล
- F1 (OOF) ขยับเข้าใกล้ 0.74 → **teacher สอน student ได้จริง** (finding เชิงบวก)
- F1 ไม่ขยับ/ตก, precision ตก → silver positive ปนเปื้อน (teacher over-flag) → เพิ่ม `--pos-conf` แล้วลองใหม่
- recall ขึ้นแต่ precision ตก → silver ช่วยจับเพิ่มแต่พาความมั่ว → ปรับ balance/threshold

## ข้อจำกัด/ความเสี่ยงที่รู้ก่อน (honest)
- teacher เองก็เพดาน F1 ~0.74 precision ~0.60 → student ไม่น่าจะเกิน teacher
- ประชดหายาก → บน pool ดิบส่วนใหญ่เป็น negative; positive ของ teacher ส่วนมากอาจเป็น false positive
  → ป้าย silver ฝั่งประชดเสี่ยงปนเปื้อนสุด (กรองด้วย pos-conf สูงๆ ช่วยได้แต่ไม่หมด)
- gold มี self-selection bias อยู่แล้ว (ดู PROVENANCE.md) → distillation รับ bias นั้นต่อ

## Dry-check ที่ผ่านแล้ว (ไม่ใช้ API)
- `distill_label.py` บน `gold_pred.csv` → silver 66 ข้อ (33/33), ทิ้งช่วงกลาง 61 ข้อ, คอลัมน์ถูกต้อง
- `distill_train_eval.py` การเตรียมข้อมูล: fold0 = gold-train 101 + silver 66 = 167 เทรน / 26 วัดผล, label/shape ถูก
- (ใช้ gold_pred.csv แค่ทดสอบ plumbing — ของจริงต้องใช้ pool ใหม่จาก fetch_to_csv.py ไม่ใช่ป้ายบน gold)

## ไฟล์
- `fetch_to_csv.py` — ดึงคอมเมนต์หลายแหล่ง → CSV (ปากทางขยายข้อมูล)
- `batch_eval.py` — teacher ติดป้าย (Batch API ครึ่งราคา)
- `distill_label.py` — กรองความมั่นใจ → silver
- `distill_train_eval.py` — เทรน+วัดผล OOF (เทียบ baseline 0.62 ได้)
- artifact ที่สร้าง (`pool*.csv`, `silver*.csv`, `wcb_distill_oof.csv`) — gitignore ไว้ (regenerable + ข้อมูลคนอื่น)
