# -*- coding: utf-8 -*-
"""Semantic cache (Redis) -- the real system + a measurement of whether it actually works on this task

Read before use: finding 14 in RESULTS.md **already measured that this kind of cache does not work for sarcasm**
   the highest similarity in the dataset is 0.942 -> setting threshold 0.95 gives a 0% hit rate (the cache never fires)
   lower the threshold to make it work and it starts returning wrong answers: at 0.90, hit 7.7% but wrong 15.2%
   this file **was not written to overturn that result** but to confirm it end-to-end
   semantic_cache_test.py measures "text pairs" (pairwise) -> this file measures a "system that actually runs"

Why it is still worth writing: a negative result from a *working system* is stronger than one from analysis alone
We can report "we built it, ran it for real, and measured that it does not work, for this linguistic reason"
which is stronger than "we did not try"

The task-level cause (not engineering -- cannot be fixed with code):
  a semantic cache rests on the assumption "similar text = same answer"
  but sarcasm is defined by *intent that runs counter to the surface content*, and the surface content is the only thing the embedding measures
  -> two restaurant reviews are similar because of the "same topic", not the "same stance" -> the cache returns a wrong answer

backend:
  Redis if reachable (--redis-url), otherwise fall back to an in-memory dict -> always runnable without standing up a server
  real production should use RediSearch (HNSW vector index) instead of a full scan with numpy
  here we scan directly because n=127 -- and because this task bottleneck is not speed, it is correctness

Run:
  python semantic_cache.py                       (in-memory, free)
  python semantic_cache.py --redis-url redis://localhost:6379
  python semantic_cache.py --thresholds 0.99 0.95 0.90 0.85
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_CSV = os.path.join(HERE, "gold.csv")
OUT_JSON = os.path.join(HERE, "semantic_cache_result.json")
ENC = "intfloat/multilingual-e5-large"      # same encoder as semantic_cache_test.py
COST_PER_CALL = 391 / 1e6 * 2.50 + 7 / 1e6 * 10.0


class RedisBackend:
    """store vector + answer in Redis. key: cache:<n> -> hash{vec(bytes), answer, text}"""

    def __init__(self, url, prefix="sarcasm"):
        import redis
        self.r = redis.Redis.from_url(url, decode_responses=False)
        self.r.ping()
        self.prefix = prefix
        self.r.delete(f"{prefix}:n")
        for k in self.r.scan_iter(f"{prefix}:item:*"):
            self.r.delete(k)

    def add(self, vec, answer, text):
        n = self.r.incr(f"{self.prefix}:n") - 1
        self.r.hset(f"{self.prefix}:item:{n}", mapping={
            b"vec": vec.astype(np.float32).tobytes(),
            b"answer": str(answer).encode(),
            b"text": text.encode("utf-8")})

    def all_vectors(self):
        vecs, answers = [], []
        n = int(self.r.get(f"{self.prefix}:n") or 0)
        for i in range(n):
            h = self.r.hgetall(f"{self.prefix}:item:{i}")
            if not h:
                continue
            vecs.append(np.frombuffer(h[b"vec"], dtype=np.float32))
            answers.append(h[b"answer"].decode())
        return (np.vstack(vecs) if vecs else np.empty((0, 0))), answers

    def name(self):
        return "Redis"


class MemoryBackend:
    def __init__(self):
        self.vecs, self.answers = [], []

    def add(self, vec, answer, text):
        self.vecs.append(vec.astype(np.float32))
        self.answers.append(str(answer))

    def all_vectors(self):
        return (np.vstack(self.vecs) if self.vecs else np.empty((0, 0))), self.answers

    def name(self):
        return "in-memory dict"


class SemanticCache:
    def __init__(self, backend, threshold):
        self.b = backend
        self.threshold = threshold
        self.hits = self.misses = 0

    def get(self, vec):
        """return (answer, sim) if a close-enough match is found, else (None, best_sim)"""
        M, answers = self.b.all_vectors()
        if M.size == 0:
            self.misses += 1
            return None, 0.0
        sims = M @ vec                       # vectors are already normalized -> dot = cosine
        j = int(np.argmax(sims))
        if float(sims[j]) >= self.threshold:
            self.hits += 1
            return answers[j], float(sims[j])
        self.misses += 1
        return None, float(sims[j])

    def put(self, vec, answer, text):
        self.b.add(vec, answer, text)


def embed(texts, batch=8):
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ENC)
    mdl = AutoModel.from_pretrained(ENC, dtype=torch.float16).eval().to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            b = tok(["query: " + t for t in texts[i:i + batch]], padding=True,
                    truncation=True, max_length=256, return_tensors="pt").to(dev)
            h = mdl(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    E = np.vstack(out)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def replay(E, y, threshold, backend_factory):
    """replay gold item by item like real traffic: ask the cache first, on a miss 'call the LLM' then store it
    measure: hit rate / what % of cache-returned answers are wrong / how much is saved"""
    cache = SemanticCache(backend_factory(), threshold)
    served_wrong = 0
    hit_sims = []
    for i in range(len(y)):
        ans, sim = cache.get(E[i])
        if ans is not None:
            hit_sims.append(sim)
            if str(ans) != str(y[i]):
                served_wrong += 1          # cache hit but returned an answer not matching this item true label
        else:
            cache.put(E[i], str(y[i]), "")  # miss -> "call the LLM" (assume it returns the right answer) then store
    n = len(y)
    return {
        "threshold": threshold,
        "hits": cache.hits, "misses": cache.misses,
        "hit_rate": cache.hits / n,
        "llm_calls_saved": cache.hits,
        "cost_saved_usd": cache.hits * COST_PER_CALL,
        "wrong_served": served_wrong,
        "wrong_rate_of_hits": (served_wrong / cache.hits) if cache.hits else None,
        "max_sim_seen": max(hit_sims) if hit_sims else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis-url", default=os.environ.get("REDIS_URL"))
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.99, 0.97, 0.95, 0.93, 0.90, 0.85])
    a = ap.parse_args()

    d = pd.read_csv(GOLD_CSV, encoding="utf-8-sig")
    y = d["label"].astype(int).tolist()
    texts = d["text"].astype(str).tolist()
    n = len(y)

    def factory():
        if a.redis_url:
            try:
                return RedisBackend(a.redis_url)
            except Exception as e:
                print(f"cannot connect to Redis ({type(e).__name__}) -> using in-memory instead")
        return MemoryBackend()

    probe = factory()
    print(f"semantic cache | gold {n} items | encoder {ENC.split('/')[-1]} | backend {probe.name()}")
    print("encoding ...")
    E = embed(texts)
    S = E @ E.T
    np.fill_diagonal(S, -1)
    print(f"highest similarity between distinct items in this set: {S.max():.4f}\n")

    print(f"{'threshold':>10} {'hit':>5} {'hit rate':>9} {'wrong':>7} {'% of hits':>10} "
          f"{'calls saved':>14} {'$ saved':>10}")
    print("-" * 76)
    rows = []
    for t in a.thresholds:
        r = replay(E, y, t, factory)
        rows.append(r)
        wr = "—" if r["wrong_rate_of_hits"] is None else f"{100*r['wrong_rate_of_hits']:.1f}%"
        print(f"{t:>10.2f} {r['hits']:>5} {100*r['hit_rate']:>8.1f}% {r['wrong_served']:>7} "
              f"{wr:>10} {r['llm_calls_saved']:>14} {'$%.4f' % r['cost_saved_usd']:>10}")

    json.dump({"n": n, "encoder": ENC, "backend": probe.name(),
               "max_pairwise_sim": float(S.max()), "results": rows},
              open(OUT_JSON, "w"), ensure_ascii=False, indent=2)

    at95 = next((r for r in rows if abs(r["threshold"] - 0.95) < 1e-9), None)
    print("\n" + "=" * 76)
    if at95 is not None:
        if at95["hits"] == 0:
            print("at threshold 0.95 (the commonly recommended value): 0 hits -> **the cache never fires**")
            print(f"because no pair of items in this set is similar up to 0.95 (max {S.max():.4f})")
        else:
            print(f"at threshold 0.95: {at95['hits']} hits "
                  f"({100*at95['hit_rate']:.1f}%) wrong {at95['wrong_served']}")
    worst = [r for r in rows if r["hits"] > 0 and (r["wrong_rate_of_hits"] or 0) > 0]
    if worst:
        w = worst[0]
        print(f"lowest threshold that still returns no wrong answers -> see the table above")
        print(f"once you lower the threshold until the cache starts working, it starts returning wrong answers "
              f"(at {w['threshold']:.2f}: wrong {100*w['wrong_rate_of_hits']:.1f}% of hits)")
    print("\nSummary: this is a working system, but the result says *do not enable it* for this task")
    print("the cause is the task, not the code -- sarcasm = intent counter to the surface content")
    print("which surface content is the only thing the embedding can measure (see finding 14)")
    print(f"saved -> {os.path.basename(OUT_JSON)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
