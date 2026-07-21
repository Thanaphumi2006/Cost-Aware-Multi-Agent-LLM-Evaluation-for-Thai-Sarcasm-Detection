# -*- coding: utf-8 -*-
"""Train "our own" model that remembers corrections permanently in the weights (fine-tuning) -- unlike few-shot

few-shot (in the web app): stuff examples into the prompt every time -- persists across sessions but is limited in count and not embedded in the model
fine-tuning (this file): train on all of gold + corrections -> get a "new model" with the knowledge baked into the weights
  -> truly permanent, no need to stuff examples into the prompt anymore, faster/cheaper inference, and remembers "everything" without limit

Steps:
  1) python finetune.py --export            -> ft_train.jsonl (gold + corrections in training format)
  2) python finetune.py --train ft_train.jsonl   -> upload + start training (costs money), prints the job id
  3) python finetune.py --status <job_id>   -> check status, when done you get the model name ft:...
  4) put the model name into predict.py (OPERATING[...]['model']) or SarcasmDetector(model="ft:...")

Things to know first (honestly):
  - need >= 10 examples (OpenAI minimum), recommend >= 50 corrections to see a clear effect -- corrections are still few right now
  - costs money twice: training cost (by token count) + a fine-tuned model charges more per call than the plain one
  - the sarcastic side of gold has self-selection bias -> the model inherits that bias (like every system in the project)
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold.csv")
OUT_JSONL = os.path.join(HERE, "ft_train.jsonl")
BASE_DEFAULT = "gpt-4o-mini-2024-07-18"   # base model that can be fine-tuned (change with --base)


def _row(text, label):
    import predict
    return {"messages": [
        {"role": "system", "content": predict.DETECT_SYS},
        {"role": "user", "content": f"ข้อความ: {text}"},
        {"role": "assistant", "content": json.dumps({"label": label}, ensure_ascii=False)},
    ]}


def do_export(out):
    import pandas as pd
    import predict
    rows, seen = [], set()
    # 1) user corrections (the real domain we want it to remember) come first
    for c in predict.load_corrections():
        t = c["text"].strip()
        if t and t not in seen and c["label"] in ("0", "1"):
            rows.append(_row(t, c["label"])); seen.add(t)
    n_corr = len(rows)
    # 2) gold (high quality) reinforces the base
    g = pd.read_csv(GOLD, dtype=str).fillna("")
    g["label"] = g["label"].str.strip()
    for _, r in g[g["label"].isin(["0", "1"])].iterrows():
        t = r["text"].strip()
        if t and t not in seen:
            rows.append(_row(t, r["label"])); seen.add(t)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples -> {out}  (corrections {n_corr} + gold {len(rows)-n_corr})")
    if len(rows) < 10:
        print("fewer than 10 examples -- OpenAI cannot train, need to label/correct more")
    else:
        print(f"next: python finetune.py --train {os.path.basename(out)}")


def do_train(path, base):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY required")
    from openai import OpenAI
    c = OpenAI()
    print(f"uploading {path} ...")
    f = c.files.create(file=open(path, "rb"), purpose="fine-tune")
    print(f"creating job (base {base}) ...")
    job = c.fine_tuning.jobs.create(training_file=f.id, model=base)
    print(f"job id: {job.id}  ·  status: {job.status}")
    print(f"check: python finetune.py --status {job.id}")


def do_status(job_id):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY required")
    from openai import OpenAI
    j = OpenAI().fine_tuning.jobs.retrieve(job_id)
    print(f"status: {j.status}")
    if j.fine_tuned_model:
        print(f"resulting model: {j.fine_tuned_model}")
        print(f"use: SarcasmDetector(model='{j.fine_tuned_model}')  or edit OPERATING in predict.py")
    elif j.status in ("running", "validating_files", "queued"):
        print("still training -- wait a moment and check again")
    elif j.error:
        print(f"error: {j.error}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", nargs="?", const=OUT_JSONL, help="build a training file from gold + corrections")
    ap.add_argument("--train", help="upload + start training from a jsonl file")
    ap.add_argument("--status", help="check job status")
    ap.add_argument("--base", default=BASE_DEFAULT, help="base model")
    a = ap.parse_args()
    if a.export is not None:
        do_export(a.export)
    elif a.train:
        do_train(a.train, a.base)
    elif a.status:
        do_status(a.status)
    else:
        ap.print_help()
