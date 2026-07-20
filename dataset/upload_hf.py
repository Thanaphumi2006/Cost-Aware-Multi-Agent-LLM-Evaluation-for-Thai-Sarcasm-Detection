# -*- coding: utf-8 -*-
"""อัปโหลดชุดข้อมูล gold ขึ้น Hugging Face เป็น dataset สาธารณะ

ใช้:  1) huggingface-cli login   (ครั้งเดียว ใช้ token แบบ write)
      2) python dataset/upload_hf.py
สร้าง/อัปเดต repo: <username>/thai-sarcasm-gold  (แก้ชื่อได้ด้วย --repo)
"""
import argparse
import os

import pandas as pd
from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="เช่น username/thai-sarcasm-gold")
    args = ap.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = args.repo or f"{user}/thai-sarcasm-gold"

    # เตรียมไฟล์สอง split จาก gold ปัจจุบัน (คอลัมน์ตรงตาม dataset card)
    cols = ["text", "label", "source", "suspect_score", "signals"]
    for src, out in [("Gold/gold.csv", "canonical.csv"), ("Gold/gold_v2.csv", "hard.csv")]:
        df = pd.read_csv(os.path.join(ROOT, src), encoding="utf-8-sig")
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df[cols].to_csv(os.path.join(HERE, out), index=False, encoding="utf-8")
        print(f"{out}: {len(df)} rows ({int(df['label'].astype(int).sum())} sarcastic)")

    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=HERE,
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["README.md", "canonical.csv", "hard.csv"],
        commit_message="Thai sarcasm gold set: canonical (127) + hard (302) splits",
    )
    print(f"\nเสร็จ -> https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
