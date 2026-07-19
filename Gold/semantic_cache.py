# -*- coding: utf-8 -*-
"""Semantic cache (Redis) — ตัวระบบจริง + วัดว่ามันได้ผลจริงไหมบนงานนี้

⚠️ อ่านก่อนใช้: finding 14 ใน RESULTS.md **วัดไว้แล้วว่าแคชแบบนี้ใช้ไม่ได้กับงานประชด**
   ความคล้ายสูงสุดในชุดข้อมูลคือ 0.942 -> ตั้ง threshold 0.95 ได้ hit rate 0% (แคชไม่เคยทำงาน)
   ลด threshold ลงให้ทำงานได้ ก็เริ่มคืนคำตอบผิด: ที่ 0.90 hit 7.7% แต่ผิด 15.2%
   ไฟล์นี้ **ไม่ได้เขียนมาเพื่อลบล้างผลนั้น** แต่เพื่อยืนยันมันแบบ end-to-end
   semantic_cache_test.py วัดแบบ "คู่ข้อความ" (pairwise) -> ไฟล์นี้วัดแบบ "ระบบที่วิ่งจริง"

ทำไมยังคุ้มที่จะเขียน: ผลลบที่ได้จาก *ระบบที่ทำงานได้จริง* หนักแน่นกว่าผลลบจากการวิเคราะห์เฉยๆ
รายงานได้ว่า "เราสร้างมันขึ้นมา รันจริง แล้ววัดได้ว่าไม่เวิร์ค เพราะเหตุผลเชิงภาษาศาสตร์ข้อนี้"
ซึ่งแข็งกว่า "เราไม่ได้ลอง"

สาเหตุเชิงงาน (ไม่ใช่เชิงวิศวกรรม -- แก้ด้วยโค้ดไม่ได้):
  แคชเชิงความหมายตั้งอยู่บนสมมติฐาน "ข้อความคล้ายกัน = คำตอบเหมือนกัน"
  แต่ประชดนิยามด้วย *เจตนาที่สวนกับเนื้อความผิวเผิน* ซึ่งเนื้อความผิวเผินคือสิ่งเดียวที่ embedding วัด
  -> รีวิวร้านอาหาร 2 อันคล้ายกันเพราะ "หัวข้อเดียวกัน" ไม่ใช่ "ท่าทีเดียวกัน" -> แคชคืนคำตอบผิด

backend:
  Redis ถ้าต่อได้ (--redis-url) มิฉะนั้น fallback เป็น dict ในหน่วยความจำ -> รันได้เสมอไม่ต้องลง server
  โปรดักชันจริงควรใช้ RediSearch (HNSW vector index) แทนการสแกนทั้งหมดด้วย numpy
  ที่นี่สแกนตรงๆ เพราะ n=127 -- และเพราะคอขวดของงานนี้ไม่ใช่ความเร็ว แต่คือความถูกต้อง

รัน:
  python semantic_cache.py                       (in-memory, ฟรี)
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
ENC = "intfloat/multilingual-e5-large"      # encoder เดียวกับ semantic_cache_test.py
COST_PER_CALL = 391 / 1e6 * 2.50 + 7 / 1e6 * 10.0


class RedisBackend:
    """เก็บ vector + คำตอบไว้ใน Redis. คีย์: cache:<n> -> hash{vec(bytes), answer, text}"""

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
        """คืน (answer, sim) ถ้าเจอของคล้ายพอ มิฉะนั้น (None, best_sim)"""
        M, answers = self.b.all_vectors()
        if M.size == 0:
            self.misses += 1
            return None, 0.0
        sims = M @ vec                       # vector ถูก normalize มาแล้ว -> dot = cosine
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
    """เล่น gold ทีละข้อเหมือน traffic จริง: ถามแคชก่อน ไม่เจอค่อย 'ยิง LLM' แล้วเก็บลงแคช
    วัด: hit rate / คำตอบที่แคชคืนมาผิดกี่ % / ประหยัดได้เท่าไหร่"""
    cache = SemanticCache(backend_factory(), threshold)
    served_wrong = 0
    hit_sims = []
    for i in range(len(y)):
        ans, sim = cache.get(E[i])
        if ans is not None:
            hit_sims.append(sim)
            if str(ans) != str(y[i]):
                served_wrong += 1          # แคช hit แต่คืนคำตอบที่ไม่ตรงกับ label จริงของข้อนี้
        else:
            cache.put(E[i], str(y[i]), "")  # miss -> "ยิง LLM" (จำลองว่าได้คำตอบถูก) แล้วเก็บ
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
                print(f"ต่อ Redis ไม่ได้ ({type(e).__name__}) -> ใช้ in-memory แทน")
        return MemoryBackend()

    probe = factory()
    print(f"semantic cache | gold {n} ข้อ | encoder {ENC.split('/')[-1]} | backend {probe.name()}")
    print("กำลัง encode ...")
    E = embed(texts)
    S = E @ E.T
    np.fill_diagonal(S, -1)
    print(f"ความคล้ายสูงสุดระหว่างข้อความคนละข้อในชุดนี้: {S.max():.4f}\n")

    print(f"{'threshold':>10} {'hit':>5} {'hit rate':>9} {'คืนผิด':>7} {'% ของ hit':>10} "
          f"{'call ที่ประหยัด':>14} {'$ ประหยัด':>10}")
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
            print("ที่ threshold 0.95 (ค่าที่มักแนะนำกัน): hit 0 ครั้ง -> **แคชไม่เคยทำงานเลย**")
            print(f"เพราะไม่มีข้อความคู่ไหนในชุดนี้คล้ายกันถึง 0.95 (สูงสุด {S.max():.4f})")
        else:
            print(f"ที่ threshold 0.95: hit {at95['hits']} ครั้ง "
                  f"({100*at95['hit_rate']:.1f}%) คืนผิด {at95['wrong_served']}")
    worst = [r for r in rows if r["hits"] > 0 and (r["wrong_rate_of_hits"] or 0) > 0]
    if worst:
        w = worst[0]
        print(f"threshold ต่ำสุดที่ยังไม่คืนคำตอบผิดเลย -> ดูตารางข้างบน")
        print(f"พอลด threshold จนแคชเริ่มทำงาน มันก็เริ่มคืนคำตอบผิด "
              f"(ที่ {w['threshold']:.2f}: ผิด {100*w['wrong_rate_of_hits']:.1f}% ของ hit)")
    print("\nสรุป: นี่คือระบบที่ทำงานได้จริง แต่ผลบอกว่า *ไม่ควรเปิดใช้* กับงานนี้")
    print("สาเหตุอยู่ที่ตัวงาน ไม่ใช่ตัวโค้ด -- ประชด = เจตนาสวนกับเนื้อความผิวเผิน")
    print("ซึ่งเนื้อความผิวเผินคือสิ่งเดียวที่ embedding วัดได้ (ดู finding 14)")
    print(f"บันทึก -> {os.path.basename(OUT_JSON)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
