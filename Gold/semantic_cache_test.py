# -*- coding: utf-8 -*-
"""Would a 95%-semantic-similarity cache return CORRECT answers on this task?

A semantic cache assumes: similar text => same answer. The retrieval test already
showed sarcasm is not a cluster in embedding space, so measure the consequence
directly: among text pairs above a similarity threshold, how often do the two
items carry DIFFERENT labels? Each such pair is a cache hit that serves a wrong answer.

Free: local encoder, no API calls.
"""
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

sys.stdout.reconfigure(encoding="utf-8")
G = r"C:\Users\thana\Downloads\Cost-Aware-Multi-Agent-LLM-Evaluation-for-Thai-Sarcasm-Detection\Gold"
ENC = "intfloat/multilingual-e5-large"   # strongest encoder from the sweep

d = pd.read_csv(f"{G}\\gold.csv")
y = d["label"].astype(int).values
txt = ["query: " + t for t in d["text"].astype(str)]
n = len(y)
dev = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(ENC)
mdl = AutoModel.from_pretrained(ENC, dtype=torch.float16).eval().to(dev)
out = []
with torch.no_grad():
    for i in range(0, n, 8):
        b = tok(txt[i:i + 8], padding=True, truncation=True, max_length=256,
                return_tensors="pt").to(dev)
        h = mdl(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
E = np.vstack(out)
E = E / np.linalg.norm(E, axis=1, keepdims=True)

S = E @ E.T
iu = np.triu_indices(n, 1)
sims = S[iu]
same = (y[iu[0]] == y[iu[1]])
print(f"gold n={n} · {len(sims)} text pairs · encoder {ENC.split('/')[-1]}")
print(f"similarity range {sims.min():.3f} .. {sims.max():.3f}  (median {np.median(sims):.3f})\n")

print("=" * 74)
print("CACHE COLLISION RATE — pairs the cache would treat as 'the same question'")
print("=" * 74)
print(f"{'threshold':>10} {'pairs above':>12} {'% of all':>9} {'WRONG answer served':>22}")
print("-" * 74)
for t in (0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90):
    m = sims >= t
    k = int(m.sum())
    if k == 0:
        print(f"{t:>10.2f} {0:>12} {0.0:>8.1f}% {'  (no pairs this similar)':>22}")
        continue
    wrong = float((~same[m]).mean())
    print(f"{t:>10.2f} {k:>12} {100*k/len(sims):>8.1f}% {100*wrong:>21.1f}%")

print("\nbaseline: two texts picked at random differ in label "
      f"{100*(1-same.mean()):.1f}% of the time")

print("\n" + "=" * 74)
print("WORST OFFENDERS — most similar pairs that DISAGREE on the label")
print("=" * 74)
dis = np.where(~same)[0]
top = dis[np.argsort(-sims[dis])][:5]
for r in top:
    i, j = iu[0][r], iu[1][r]
    print(f"\ncos-sim {sims[r]:.4f}   labels: {y[i]} vs {y[j]}")
    print(f"  [{y[i]}] {d['text'].iloc[i][:95]}")
    print(f"  [{y[j]}] {d['text'].iloc[j][:95]}")
