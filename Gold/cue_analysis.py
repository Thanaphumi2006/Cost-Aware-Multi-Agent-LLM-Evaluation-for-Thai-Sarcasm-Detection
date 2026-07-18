# -*- coding: utf-8 -*-
"""Two honest tests of surface cues, both free (no API calls).

A) Regex-cue classifier with leave-fold-out CUE SELECTION.
   The earlier F1 0.590 for "555" was picked as best-of-18 on the full set, so it is
   optimistically biased. Here the cue is chosen on 4 folds and scored on the 5th.

B) Do cues add anything the GPT bot does not already know?
   Fit logistic regression on the GPT probability alone vs GPT probability + cue flags,
   leave-fold-out, paired bootstrap. If cues add nothing here, prompt-tagging them is
   dead before spending a cent on an API rerun.

Protocol matches gpt_threshold.py:150-157 (StratifiedKFold 5, shuffle, seed 42).
"""
import re, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8")
G = r"C:\Users\thana\Downloads\Cost-Aware-Multi-Agent-LLM-Evaluation-for-Thai-Sarcasm-Detection\Gold"
N_FOLDS, SEED = 5, 42
rng = np.random.default_rng(SEED)

CUES = {
    "555": r"555", "จ้า": r"จ้า", "นะคะ": r"นะคะ", "ค่ะ": r"ค่ะ", "ครับ": r"ครับ",
    "elong": r"(.)\1{2,}", "??": r"[?]{2,}", "!!": r"[!]{2,}", "...": r"\.{3,}|…",
    "ไม้ตรี": r"๊", "ไม้จัตวา": r"๋", "จัง": r"จัง", "มาก": r"มาก", "ดี": r"ดี",
}


def f1p(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


d = pd.read_csv(f"{G}\\gold.csv")
y = d["label"].astype(int).values
txt = d["text"].astype(str).tolist()
X = np.array([[1 if re.search(p, t) else 0 for p in CUES.values()] for t in txt])
names = list(CUES)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ---------- A) leave-fold-out cue selection ----------
print("=" * 68)
print("A) REGEX-CUE CLASSIFIER — cue chosen on 4 folds, scored on the 5th")
print("=" * 68)
pred = np.zeros(len(y), int)
picked = []
for tr, te in skf.split(X, y):
    best, bf = None, -1
    for j in range(X.shape[1]):
        f = f1p(y[tr], X[tr, j])[0]
        if f > bf:
            best, bf = j, f
    picked.append(names[best])
    pred[te] = X[te, best]
f, p, r = f1p(y, pred)
print(f"cue picked per fold : {', '.join(picked)}")
print(f"leave-fold-out F1   : {f:.3f}  (prec {p:.3f}  rec {r:.3f})")
print(f"oracle best-of-18   : 0.590   <- biased, do not report")

# ---------- B) do cues add to the GPT bot? ----------
print()
print("=" * 68)
print("B) DOES CUE INFO ADD TO THE GPT BOT? (leave-fold-out logistic regression)")
print("=" * 68)
g = pd.read_csv(f"{G}\\frontier_probs_gpt-4.1-mini.csv")
assert (g["label"].astype(int).values == y).all(), "row order mismatch"
gp = g["prob"].astype(float).values.reshape(-1, 1)


def lfo_model(feats):
    out = np.zeros(len(y), int)
    for tr, te in skf.split(feats, y):
        m = LogisticRegression(max_iter=2000, C=1.0).fit(feats[tr], y[tr])
        ptr = m.predict_proba(feats[tr])[:, 1]
        taus = np.unique(ptr)
        t = max(taus, key=lambda x: f1p(y[tr], (ptr >= x).astype(int))[0])
        out[te] = (m.predict_proba(feats[te])[:, 1] >= t).astype(int)
    return out


pred_g = lfo_model(gp)
pred_gc = lfo_model(np.hstack([gp, X]))
fg = f1p(y, pred_g); fgc = f1p(y, pred_gc)
print(f"GPT prob only        : F1 {fg[0]:.3f}  (prec {fg[1]:.3f} rec {fg[2]:.3f})")
print(f"GPT prob + cue flags : F1 {fgc[0]:.3f}  (prec {fgc[1]:.3f} rec {fgc[2]:.3f})")

n = len(y)
diffs = []
for _ in range(5000):
    s = rng.integers(0, n, n)
    diffs.append(f1p(y[s], pred_gc[s])[0] - f1p(y[s], pred_g[s])[0])
diffs = np.array(diffs)
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"\nΔF1 (cues - no cues) = {fgc[0]-fg[0]:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
print(f"P(cues help) = {100*(diffs>0).mean():.1f}%")
print(f"McNemar: cues right/plain wrong {int(((pred_gc==y)&(pred_g!=y)).sum())} | "
      f"plain right/cues wrong {int(((pred_g==y)&(pred_gc!=y)).sum())}")

m_full = LogisticRegression(max_iter=2000).fit(np.hstack([gp, X]), y)
co = m_full.coef_[0]
print("\ncoefficients (full fit, direction only):")
print(f"  {'gpt_prob':12s} {co[0]:+.2f}")
for nm, c in sorted(zip(names, co[1:]), key=lambda t: -abs(t[1]))[:6]:
    print(f"  {nm:12s} {c:+.2f}")
