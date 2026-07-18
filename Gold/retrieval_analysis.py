# -*- coding: utf-8 -*-
"""Is the 'sarcasm is not a cluster' result specific to WangchanBERTa?

Repeats the retrieval test across stronger multilingual encoders. The decisive
number is the separability gap: mean(sarcasm-sarcasm sim) - mean(sarcasm-nonsarcasm sim).
Negative gap => retrieving semantically similar examples surfaces OPPOSITE-label
examples for sarcastic inputs, which is what breaks dynamic few-shot.

Free: local encoders, no API calls. Leakage guard: neighbours from other folds only.
"""
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel

sys.stdout.reconfigure(encoding="utf-8")
G = r"C:\Users\thana\Downloads\Cost-Aware-Multi-Agent-LLM-Evaluation-for-Thai-Sarcasm-Detection\Gold"
N_FOLDS, SEED = 5, 42

# (name, pooling, query prefix) - prefixes are what each model card asks for
ENCODERS = [
    ("airesearch/wangchanberta-base-att-spm-uncased", "mean", ""),
    ("BAAI/bge-m3", "cls", ""),
    ("intfloat/multilingual-e5-large", "mean", "query: "),
]

d = pd.read_csv(f"{G}\\gold.csv")
y = d["label"].astype(int).values
txt = d["text"].astype(str).tolist()
n, base = len(y), y.mean()
dev = "cuda" if torch.cuda.is_available() else "cpu"
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
print(f"gold n={n}  base rate={base:.3f}  chance neighbour agreement={base**2+(1-base)**2:.3f}\n")


def f1p(yt, pr):
    tp = int(((pr == 1) & (yt == 1)).sum()); fp = int(((pr == 1) & (yt == 0)).sum())
    fn = int(((pr == 0) & (yt == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def encode(name, pooling, prefix):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name, dtype=torch.float16).eval().to(dev)
    texts = [prefix + t for t in txt]
    out = []
    with torch.no_grad():
        for i in range(0, n, 8):
            b = tok(texts[i:i + 8], padding=True, truncation=True,
                    max_length=256, return_tensors="pt").to(dev)
            h = mdl(**b).last_hidden_state
            if pooling == "cls":
                v = h[:, 0]
            else:
                m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1)
            out.append(v.float().cpu().numpy())
    del mdl
    torch.cuda.empty_cache()
    E = np.vstack(out)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


results = []
for name, pooling, prefix in ENCODERS:
    short = name.split("/")[-1]
    print(f"--- {short} ---", flush=True)
    try:
        E = encode(name, pooling, prefix)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:120]}\n")
        continue

    agree = []
    for tr, te in skf.split(E, y):
        S = E[te] @ E[tr].T
        idx = np.argsort(-S, axis=1)[:, :2]
        for r_i, q in enumerate(te):
            agree.append((y[tr][idx[r_i]] == y[q]).mean())

    best = (-1, None)
    for k in (1, 3, 5):
        pred = np.zeros(n, int)
        for tr, te in skf.split(E, y):
            S = E[te] @ E[tr].T
            idx = np.argsort(-S, axis=1)[:, :k]
            pred[te] = (y[tr][idx].mean(axis=1) >= 0.5).astype(int)
        f = f1p(y, pred)
        if f[0] > best[0]:
            best = (f[0], k)

    pos, neg = E[y == 1], E[y == 0]
    pp = (pos @ pos.T)[np.triu_indices(len(pos), 1)].mean()
    nn_ = (neg @ neg.T)[np.triu_indices(len(neg), 1)].mean()
    pn = (pos @ neg.T).mean()
    gap = pp - pn
    results.append((short, np.mean(agree), best[0], best[1], pp, nn_, pn, gap))
    print(f"  top-2 agreement {np.mean(agree):.3f} | best kNN F1 {best[0]:.3f} (k={best[1]}) | gap {gap:+.4f}\n")

print("=" * 78)
print(f"{'encoder':34s} {'top2agr':>8} {'kNN_F1':>7} {'s-s':>7} {'n-n':>7} {'s-n':>7} {'GAP':>8}")
print("-" * 78)
for r in results:
    print(f"{r[0]:34s} {r[1]:>8.3f} {r[2]:>7.3f} {r[4]:>7.3f} {r[5]:>7.3f} {r[6]:>7.3f} {r[7]:>+8.4f}")
print("\nGAP = sarcasm-sarcasm sim minus sarcasm-nonsarcasm sim.")
print("GAP <= 0  ->  sarcastic items are NOT closer to each other than to non-sarcastic ones,")
print("             so top-k retrieval feeds opposite-label examples for sarcastic inputs.")
print("\nreference F1: GPT bot 0.727 | regex `555` 0.590 | open 7-8B 0.45-0.58 | always-yes 0.382")
