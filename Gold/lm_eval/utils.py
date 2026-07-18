# -*- coding: utf-8 -*-
"""helper สำหรับ task thai_sarcasm ใน lm-evaluation-harness

- process_docs : เติมคอลัมน์ gold (int 0/1) จาก label (str)
- doc_to_text  : ประกอบ prompt = rubric + ข้อความ (rubric ชุดเดียวกับ DETECT_SYS ใน predict.py)
"""
import datasets

# ตรงกับ DETECT_SYS ใน Gold/predict.py (ให้เทียบกับผล GPT ได้ยุติธรรม)
RUBRIC = (
    'ตัดสินว่าข้อความภาษาไทยนี้ "ประชด/เสียดสี" หรือไม่\n'
    "ประชด = เจตนาจริงตรงข้ามกับความหมายผิวเผิน เพื่อเหน็บหรือแสดงความไม่พอใจ"
)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _add_gold(doc):
        doc["gold"] = 1 if str(doc.get("label", "")).strip() in ("1", "1.0") else 0
        doc["text"] = str(doc.get("text", "")).strip()
        return doc
    return dataset.map(_add_gold)


def doc_to_text(doc) -> str:
    return f"{RUBRIC}\n\nข้อความ: {doc['text']}\nเป็นการประชดหรือไม่? คำตอบ:"
