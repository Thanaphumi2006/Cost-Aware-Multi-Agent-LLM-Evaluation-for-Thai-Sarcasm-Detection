# -*- coding: utf-8 -*-
"""A "production-ready" Thai sarcasm detector — distilled from findings 1-11

Findings summary: no multi-agent needed. The best-value system = single agent, 1 call + read the logprob + threshold.
  - model gpt-4.1-mini (cheapest/best on the frontier, ~$0.0001/item)
  - one call, request logprobs -> P(sarcastic) -> compare to a threshold at the chosen "operating point"

The "achievable" operating points on this task (chosen from the PR curve on 127-item gold) — pick the "model" per task:
  balanced    : gpt-4.1-mini t=0.095  P≈0.68 R≈0.83 F1≈0.75  cheapest (~$0.0001/item) — default
  high_recall : gpt-4o       t=0.05   P≈0.43 R≈1.00           "must not miss" -> a human reviews FPs (~6x pricier)
  review_band : 0.05–0.50 = "send to a human"                 (answer confidently outside the band, defer inside it)

*** Limitations to know before deploying (honest — measured from real data, not guessed) ***
  - **gpt-4.1-mini has a recall ceiling of ~0.83**: for 5/30 sarcastic items it scores ~0 (can't see them); no threshold finds them
    -> tasks where "you can't miss sarcasm" must use gpt-4o (high_recall mode), not mini
  - precision ceiling ~0.68 on both models — you can't set "high precision (>0.8)" because balanced reviews are genuinely borderline
  - recall on gold is inflated (self-selection bias, see PROVENANCE.md) -> real-world will be lower
  - trained/measured on Wongnai (reviews) + Wisesight (tweets) — other domains are expected to drop
  - realistic expected F1 ~0.70, not 0.9x — don't over-promise

Usage:
  export OPENAI_API_KEY=sk-...
  python predict.py "ขอบคุณที่ให้รอ 2 ชม. บริการดีจริงๆ"        # one text
  python predict.py --csv in.csv --out out.csv --text-col text  # whole file (batch)
  python predict.py "..." --op high_recall                      # choose the operating point

Or import:  from predict import SarcasmDetector
"""
import argparse
import hashlib
import json
import math
import os
import sys
import threading

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, ".predict_cache.json")   # gitignored -- re-firing the same text = free
CORR_PATH = os.path.join(HERE, ".predict_corrections.json")  # gitignored -- where a human marked "the model decided wrong"
MODEL = "gpt-4.1-mini"
# (the Thai prompt is the experimental instruction to the model and is kept as-is)
DETECT_SYS = (
    'ตัดสินว่าข้อความภาษาไทยนี้ "ประชด/เสียดสี" หรือไม่\n'
    "ประชด = เจตนาจริงตรงข้ามกับความหมายผิวเผิน เพื่อเหน็บหรือแสดงความไม่พอใจ\n\n"
    'ตอบเป็น JSON เท่านั้น: {"label": "1" หรือ "0"}\n1 = ประชด, 0 = ไม่ประชด'
)

# ---------- corrections: learn from what a human marked "decided wrong" (in-context few-shot, not real retraining) ----------
# Corrections are stored "unlimited" (permanent, across sessions), then at predict time we pull only the "most relevant" ones into the prompt
_MAX_SHOTS = 10          # examples included per prediction (chosen from the most similar, to avoid token bloat)
_corr_lock = threading.Lock()


def _trigrams(s):
    s = "".join(s.split())
    return set(s[i:i+3] for i in range(len(s) - 2)) if len(s) >= 3 else {s}


def _relevant(corr, query, k=_MAX_SHOTS):
    """Pick the corrections "most relevant to this text" (Jaccard of char-trigrams — works for Thai, no word segmentation)
    -> store as many corrections as you like, but include only the ones that help this item = permanent, scalable learning"""
    if len(corr) <= k:
        return corr
    q = _trigrams(query)
    def sim(c):
        t = _trigrams(c["text"])
        return len(t & q) / (len(t | q) or 1)
    return sorted(corr, key=sim, reverse=True)[:k]


def load_corrections():
    if os.path.exists(CORR_PATH):
        try:
            with open(CORR_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def add_correction(text, correct_label):
    """A human clicks 'wrong' -> store (text, correct label). '1'=sarcasm '0'=not sarcasm.
    Dedupe by text (latest wins) · return the total correction count"""
    text = (text or "").strip()
    correct_label = str(correct_label).strip()
    if not text or correct_label not in ("0", "1"):
        raise ValueError("correct_label must be '0' or '1', and text must not be empty")
    with _corr_lock:
        corr = [c for c in load_corrections() if c.get("text") != text]
        corr.append({"text": text, "label": correct_label})
        tmp = CORR_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(corr, f, ensure_ascii=False)
        os.replace(tmp, CORR_PATH)
    return len(corr)


def _shots_block(shots):
    """Turn the (already-selected) correction list into a few-shot block appended to the system prompt"""
    if not shots:
        return ""
    # (the Thai few-shot header is part of the prompt sent to the model and is kept as-is)
    lines = ["\n\nตัวอย่างที่คนยืนยันคำตอบที่ถูกแล้ว (ให้ยึดตามนี้กับข้อความคล้ายๆ กัน):"]
    for c in shots:
        t = c["text"].replace("\n", " ")[:160]
        lines.append(f'  "{t}" -> {c["label"]}')
    return "\n".join(lines)


def _corr_sig(corr):
    """Signature of the *entire* corrections set -> part of the cache key
    (corrections change -> prompt changes -> old probs are invalid, must namespace separately)"""
    if not corr:
        return "0"
    raw = "|".join(f'{c["text"]}={c["label"]}' for c in corr)
    return hashlib.sha1(raw.encode()).hexdigest()[:10]

# operating points: model+threshold chosen from the PR curve on gold — see header (pick the model per task)
OPERATING = {
    "balanced":    {"model": "gpt-4.1-mini", "t": 0.095, "desc": "highest F1, cheapest (P≈0.68 R≈0.83)"},
    "high_recall": {"model": "gpt-4o",        "t": 0.050, "desc": "catch all R≈1.00, accept FP (P≈0.43)"},
}
REVIEW_LO, REVIEW_HI = 0.05, 0.50   # the "defer to a human" band for review_band mode


class _Cache:
    """JSON-file cache: (model,text) -> prob. Re-firing the same text = read from cache, no cost.
    The key is a hash (avoids file bloat / character issues) · written atomically (survives an interrupt)"""
    def __init__(self, path=CACHE_PATH):
        self.path, self.lock, self.d = path, threading.Lock(), {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.d = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.d = {}

    @staticmethod
    def key(model, text, sig="0"):
        return hashlib.sha1(f"{model}\x00{sig}\x00{text}".encode()).hexdigest()

    def get(self, model, text, sig="0"):
        return self.d.get(self.key(model, text, sig))

    def put(self, model, text, prob, sig="0"):
        if not self.path:
            return
        with self.lock:
            self.d[self.key(model, text, sig)] = prob
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.d, f)
            os.replace(tmp, self.path)


class SarcasmDetector:
    def __init__(self, operating="balanced", api_key=None, model=None, cache=True):
        from openai import OpenAI
        self.model = model or OPERATING[operating]["model"]
        self.t = OPERATING[operating]["t"]
        self.op = operating
        self.client = OpenAI(api_key=api_key, timeout=30.0, max_retries=3)
        self.cache = _Cache() if cache else None
        self.hits = self.misses = 0
        self.reload_corrections()

    def reload_corrections(self):
        """Load all corrections (permanent, from file), kept for selection at predict time.
        Call again after a human clicks 'wrong' so the next predictions use the new examples"""
        self.corr = load_corrections()
        self.corr_sig = _corr_sig(self.corr)
        self.n_corr = len(self.corr)

    def prob(self, text):
        """Return P(sarcastic) 0..1 from the label token's logprob (1 call) — check the cache first"""
        if self.cache is not None:
            c = self.cache.get(self.model, text, self.corr_sig)
            if c is not None:
                self.hits += 1
                return c
        self.misses += 1
        p = self._call(text)
        if self.cache is not None and p == p:      # don't cache NaN
            self.cache.put(self.model, text, p, self.corr_sig)
        return p

    def _call(self, text):
        # select the corrections most relevant to this text and append to the prompt (per-query retrieval)
        sys_prompt = DETECT_SYS + _shots_block(_relevant(self.corr, text))
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=20, response_format={"type": "json_object"},
            logprobs=True, top_logprobs=20,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": f"ข้อความ: {text}"}])
        for tok in (r.choices[0].logprobs.content or []):
            if tok.token.strip().strip('"') not in ("0", "1"):
                continue
            p0 = p1 = 0.0
            for alt in tok.top_logprobs:
                t = alt.token.strip().strip('"')
                if t == "1": p1 += math.exp(alt.logprob)
                elif t == "0": p0 += math.exp(alt.logprob)
            if p0 + p1 > 0:
                return p1 / (p0 + p1)
        # couldn't read the logprob -> use the hard answer
        try:
            return 1.0 if str(json.loads(r.choices[0].message.content or "{}").get("label", "")) == "1" else 0.0
        except json.JSONDecodeError:
            return float("nan")

    def predict(self, text, review_band=False):
        """Return dict: label, prob, decision. review_band=True enables 'defer to human' mode in the borderline band"""
        p = self.prob(text)
        if p != p:                      # NaN
            return {"label": None, "prob": None, "decision": "error"}
        if review_band and REVIEW_LO <= p <= REVIEW_HI:
            return {"label": None, "prob": round(p, 3), "decision": "review"}
        label = 1 if p >= self.t else 0
        return {"label": label, "prob": round(p, 3),
                "decision": "sarcasm" if label else "not_sarcasm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="one text")
    ap.add_argument("--csv", help="input file (batch)")
    ap.add_argument("--out", help="output file (with --csv)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--op", default="balanced", choices=list(OPERATING))
    ap.add_argument("--review-band", action="store_true", help="defer borderline items to a human")
    a = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY required (export OPENAI_API_KEY=sk-...)")
    det = SarcasmDetector(operating=a.op)
    print(f"[predict] {det.model} · operating point '{a.op}' (t={det.t}) · {OPERATING[a.op]['desc']}", file=sys.stderr)
    print("[predict] warning: measured only on reviews/tweets (F1~0.72) — other domains (YouTube/news/formal) untested",
          file=sys.stderr)

    if a.csv:
        import pandas as pd
        df = pd.read_csv(a.csv, dtype=str).fillna("")
        if a.text_col not in df.columns:
            sys.exit(f"no column '{a.text_col}' (have: {list(df.columns)})")
        res = [det.predict(t, review_band=a.review_band) for t in df[a.text_col]]
        df["pred_label"] = [r["label"] for r in res]
        df["pred_prob"] = [r["prob"] for r in res]
        df["pred_decision"] = [r["decision"] for r in res]
        out = a.out or (os.path.splitext(a.csv)[0] + "_pred.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        n = len(df); ns = sum(1 for r in res if r["decision"] == "sarcasm")
        nr = sum(1 for r in res if r["decision"] == "review")
        print(f"wrote {out} · {n} items · sarcasm {ns}" + (f" · deferred {nr}" if a.review_band else "")
              + f" · cache hit {det.hits}/{det.hits+det.misses}")
    elif a.text:
        r = det.predict(a.text, review_band=a.review_band)
        print(json.dumps(r, ensure_ascii=False))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
