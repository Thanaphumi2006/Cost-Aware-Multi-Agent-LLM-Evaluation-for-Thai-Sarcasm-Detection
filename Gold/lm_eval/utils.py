# -*- coding: utf-8 -*-
"""helper for the thai_sarcasm task in lm-evaluation-harness

- process_docs : add the gold column (int 0/1) from label (str)
- doc_to_text  : build the prompt = rubric + text (same rubric as DETECT_SYS in predict.py)
"""
import datasets

# matches DETECT_SYS in Gold/predict.py (for a fair comparison with the GPT results)
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
