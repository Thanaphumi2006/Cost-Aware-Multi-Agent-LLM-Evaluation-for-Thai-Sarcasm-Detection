# -*- coding: utf-8 -*-
"""ระบบ ⑦ — Cascade: WangchanBERTa (ฟรี) คัดกรอง -> GPT verifier ปัดตก

แนวคิด: เอาสถาปัตยกรรมที่ชนะ (คัดกรองแบบ recall สูง -> ผู้ตรวจที่ "ปัดตกได้อย่างเดียว")
มาเปลี่ยนแค่ "ตัวคัดกรอง" จาก GPT (จ่ายทุกข้อ) เป็น WangchanBERTa (ฟรี ออฟไลน์ 26ms)

  v2 (ของเดิม):  GPT screener ทุกข้อ (127 calls) + GPT verifier เฉพาะที่ว่าประชด (56) = 183 calls
  cascade:       WCB screener ทุกข้อ (0 calls, ฟรี)  + GPT verifier เฉพาะที่ WCB ว่าประชด = N calls
  -> พื้นราคาที่เคยจ่ายทุกข้อ "หายไปทั้งชั้น"

ทำไม F1 0.620 ของ WCB ไม่ได้แปลว่ามันเป็นด่านคัดกรองไม่ได้:
  ด่านคัดกรองไม่ต้องแม่น -- ต้องแค่ "ไม่ปล่อยประชดหลุด" (recall สูง) ส่วนที่เหวี่ยงเกิน verifier ตัดทิ้งให้
  argmax @0.5 ให้ recall แค่ 0.700 -> ต่ำเกินจะเป็นด่านแรก
  แต่ถ้า "ลด threshold" ลง recall จะขึ้นไปได้ โดยแลกกับ precision (ซึ่ง verifier ซ่อมให้ได้)

เลือก threshold ยังไงไม่ให้โกง (สำคัญมาก):
  ห้ามเลือก threshold จาก gold ทั้งชุดแล้วไปวัดผลบน gold ชุดเดิม -- นั่นคือ leak
  ที่นี่ใช้ leave-fold-out: ข้อที่อยู่ fold k จะใช้ threshold ที่เลือกจาก "อีก 4 folds เท่านั้น"
  -> ไม่มีข้อไหนมีส่วนกำหนด threshold ที่ตัดสินตัวมันเอง

เพดานที่หนีไม่พ้น: recall ของ cascade <= recall ของ WCB screener (verifier ปัดตกได้อย่างเดียว
เพิ่มประชดใหม่ไม่ได้) -- ประชดข้อไหนที่ WCB ปล่อยหลุดตั้งแต่ด่านแรก คือหลุดถาวร

รัน:
  python cascade.py --dry-run          ดูว่าแต่ละ threshold จะ flag กี่ข้อ / คาดว่าจ่ายเท่าไหร่ (ฟรี ไม่ยิง API)
  python cascade.py --target-recall 0.95   รันจริง (ต้องมี OPENAI_API_KEY)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

import multiagent
from baseline import PRICE_PER_MTOK, metrics

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OOF_CSV = os.path.join(HERE, "wcb_oof_probs.csv")
# ตั้งชื่อให้ compare_systems.py (ซึ่ง glob multiagent_preds_gpt*.csv) เก็บไปเทียบให้อัตโนมัติ
OUT_CSV = os.path.join(HERE, "multiagent_preds_gpt_cascade.csv")
IN_P, OUT_P = PRICE_PER_MTOK["gpt"]
SEEDS_DEFAULT = [42, 7, 2024]


def cost(i, o):
    return i / 1e6 * IN_P + o / 1e6 * OUT_P


def pick_threshold(probs, y, target_recall):
    """threshold ที่ "สูงที่สุด" ที่ยัง recall >= target บนข้อมูลที่ให้มา
    สูงที่สุด = flag น้อยที่สุด = ถูกที่สุด ภายใต้เงื่อนไข recall ที่ต้องการ
    (recall เป็นฟังก์ชันไม่เพิ่มตาม threshold -> ไล่ดูเฉพาะค่า prob ของ "ข้อที่เป็นประชดจริง" ก็พอ)"""
    pos = np.sort(probs[y == 1])[::-1]          # prob ของประชดจริง เรียงมาก->น้อย
    if len(pos) == 0:
        return 0.0
    k = int(np.ceil(target_recall * len(pos)))  # ต้องจับประชดจริงให้ได้อย่างน้อย k ข้อ
    k = min(max(k, 1), len(pos))
    return float(pos[k - 1])                    # ตั้ง threshold ที่ข้อที่ k -> จับได้ k ข้อพอดี


def screen(df, seed, target_recall):
    """คืน (flag[], tau_ต่อ fold) -- flag=1 แปลว่าส่งต่อให้ verifier
    threshold ของ fold k เลือกจาก "fold อื่น" เท่านั้น (leave-fold-out) ไม่ใช่จาก fold k เอง"""
    probs = df[f"prob_seed{seed}"].values
    folds = df[f"fold_seed{seed}"].values
    y = df["label"].astype(int).values

    flag = np.zeros(len(df), dtype=int)
    taus = {}
    for k in sorted(set(folds)):
        held = folds == k
        rest = ~held
        tau = pick_threshold(probs[rest], y[rest], target_recall)   # เลือกจากอีก 4 folds
        flag[held] = (probs[held] >= tau).astype(int)               # ใช้กับ fold ที่กันไว้
        taus[int(k)] = tau
    return flag, taus


def screener_stats(df, seed, target_recall):
    flag, taus = screen(df, seed, target_recall)
    y = df["label"].astype(int).values
    tp = int(((flag == 1) & (y == 1)).sum())
    fp = int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return dict(seed=seed, flagged=int(flag.sum()), tp=tp, fp=fp, fn=fn,
                recall=rec, prec=prec, taus=taus, flag=flag)


def dry_run(df, seeds, targets):
    """ไม่ยิง API เลย -- ตอบคำถามเดียว: "ถ้าเอา WCB มาเป็นด่านคัดกรอง จะเหลือกี่ข้อให้ GPT ตรวจ" """
    # ค่าเฉลี่ยจริงจาก v2: verifier 1 ครั้ง ~391 in / 7 out tokens
    per_call = cost(391, 7)
    print(f"ประมาณราคา verifier: ${per_call:.5f}/call (391 in / 7 out tokens -- ค่าจริงจาก v2)")
    print(f"เทียบเป้า: v2 = 183 calls, $0.169 | baseline = 127 calls, $0.094\n")
    print(f"{'target':>7} {'seed':>5} {'flag':>5} {'recall':>7} {'prec':>6} {'FN':>3} "
          f"{'calls':>6} {'~cost':>8} {'vs v2':>7}")
    print("-" * 62)
    for t in targets:
        rows = [screener_stats(df, s, t) for s in seeds]
        for r in rows:
            c = r["flagged"] * per_call
            print(f"{t:>7.2f} {r['seed']:>5} {r['flagged']:>5} {r['recall']:>7.3f} {r['prec']:>6.3f} "
                  f"{r['fn']:>3} {r['flagged']:>6} {'$%.3f' % c:>8} {'%.2fx' % (c/0.169):>7}")
        mr = np.mean([r["recall"] for r in rows])
        mf = np.mean([r["flagged"] for r in rows])
        print(f"{'':>7} {'mean':>5} {mf:>5.1f} {mr:>7.3f} {'':>6} {'':>3} {mf:>6.0f} "
              f"{'$%.3f' % (mf*per_call):>8} {'%.2fx' % (mf*per_call/0.169):>7}\n")
    print("อ่านยังไง: recall ตรงนี้คือ 'เพดาน' ของ cascade -- verifier ปัดตกได้อย่างเดียว")
    print("           ดันขึ้นไม่ได้อีก ส่วน precision ต่ำไม่เป็นไร verifier มีหน้าที่ซ่อมให้")
    print("           FN = ประชดที่ด่านแรกปล่อยหลุด = เสียถาวร")


def verify(client, texts, cache):
    """GPT verifier -- prompt ตัวเดียวกับ v2 เป๊ะ (multiagent.VERIFY_SYS) ตัวแปรเดียวที่เปลี่ยนคือ 'ใครคัดกรอง'
    cache กันจ่ายซ้ำข้อความเดิม (3 seeds flag ข้อเดียวกันเยอะ -- ยิงครั้งเดียวพอ)"""
    n_in = n_out = calls = 0
    for t in texts:
        if t in cache:
            continue
        v, i, o = multiagent._ask(client, multiagent.VERIFY_SYS, multiagent.VERIFY_SCHEMA, "verdict", t)
        cache[t] = v if v in ("0", "1") else "1"     # ตอบเพี้ยน -> คงไว้ (ตรงกับกฎ "ไม่ชัด = คงเป็นประชด")
        n_in += i; n_out += o; calls += 1
        print(f"    verifier {calls}/{len(texts)}  -> {cache[t]}", end="\r", flush=True)
    return n_in, n_out, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="ไม่ยิง API -- แค่ดูว่าจะ flag กี่ข้อ")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    a = ap.parse_args()

    if not os.path.exists(OOF_CSV):
        sys.exit(f"ไม่พบ {OOF_CSV} -- รัน wcb_oof_probs.py ก่อน (ฟรี)")
    df = pd.read_csv(OOF_CSV, dtype={"text": str, "label": str}).fillna("")
    df["label"] = df["label"].str.strip()

    print(f"cascade: WangchanBERTa (ฟรี) -> GPT verifier | gold {len(df)} ข้อ\n")
    if a.dry_run:
        dry_run(df, a.seeds, [0.85, 0.90, 0.95, 1.00])
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ต้องมี OPENAI_API_KEY (export OPENAI_API_KEY=sk-...)")
    client = multiagent._make_client()

    y = df["label"].tolist()
    cache = {}
    t0 = time.time()
    per_seed, preds_by_seed = [], {}

    for seed in a.seeds:
        st = screener_stats(df, seed, a.target_recall)
        flag = st["flag"]
        idx = np.where(flag == 1)[0]
        print(f"seed {seed}: WCB flag {len(idx)} ข้อ (recall {st['recall']:.3f}) -> ส่งให้ verifier")

        n_in, n_out, ncalls = verify(client, [df["text"].iloc[i] for i in idx], cache)

        # verifier ปัดตกได้อย่างเดียว: pred=1 ก็ต่อเมื่อ (WCB flag) และ (verifier ยืนยัน)
        pred = ["0"] * len(df)
        for i in idx:
            pred[i] = cache[df["text"].iloc[i]]
        preds_by_seed[seed] = pred

        _, prec, rec, f1, (tn, fp, fn, tp) = metrics(y, pred)
        killed = sum(1 for i in idx if cache[df["text"].iloc[i]] == "0")
        per_seed.append(dict(seed=seed, f1=f1, prec=prec, rec=rec, tp=tp, fp=fp, fn=fn,
                             calls=int(flag.sum()), killed=killed,
                             cost=flag.sum() * cost(391, 7)))
        print(f"  -> F1 {f1:.3f} | prec {prec:.3f} | recall {rec:.3f} | TP {tp} FP {fp} FN {fn} "
              f"| verifier ปัดตก {killed} ข้อ | calls {int(flag.sum())}\n")

    f1s = [r["f1"] for r in per_seed]
    med_i = int(np.argsort(f1s)[len(f1s) // 2])          # seed ตัวแทน = median ไม่ใช่ตัวดีที่สุด (cherry-pick)
    rep = per_seed[med_i]["seed"]

    out = df[["text", "label"]].copy()
    out["pred"] = preds_by_seed[rep]
    for s in a.seeds:
        out[f"pred_seed{s}"] = preds_by_seed[s]
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    mean_calls = float(np.mean([r["calls"] for r in per_seed]))
    mean_cost = float(np.mean([r["cost"] for r in per_seed]))
    print("=" * 66)
    print(f"target recall (ตั้งไว้)  : {a.target_recall}")
    print(f"F1 ต่อ seed              : {', '.join(f'{f:.3f}' for f in f1s)}")
    print(f"F1 เฉลี่ย                : {np.mean(f1s):.3f}  (SD {np.std(f1s):.3f})")
    print(f"seed ตัวแทน              : {rep} (median)")
    print(f"LLM calls เฉลี่ย         : {mean_calls:.0f}   (v2 = 183, baseline = 127)")
    print(f"ค่าใช้จ่ายเฉลี่ย            : ${mean_cost:.3f}  (v2 = $0.169, baseline = $0.094)")
    print(f"ยิง API จริงรอบนี้        : {len(cache)} ครั้ง (cache กันจ่ายซ้ำข้ามseed)")
    print(f"เวลารวม                  : {(time.time()-t0)/60:.1f} นาที")
    print(f"บันทึก -> {OUT_CSV}")
    print("=" * 66)
    print("ต่อไป: python compare_systems.py  (bootstrap + McNemar เทียบกับระบบอื่น)")


if __name__ == "__main__":
    main()
