# -*- coding: utf-8 -*-
"""
Multi-agent: detector -> verifier(กรอง false positive ตาม rubric)
เทียบกับ baseline.py เอเจนต์เดี่ยว บน harness เดียวกัน (metric ตัวเดียวกันเป๊ะ)

ทำไมโครงนี้ (อิงผล baseline จริง):
  baseline GPT-4o ได้ recall 1.000 (จับประชดครบ) แต่ precision แค่ 0.526
  -> ปัญหาคือ false positive (ทายว่าประชด 27 ข้อทั้งที่ไม่ใช่ ส่วนใหญ่คือ "รีวิวสมดุล")
  -> verifier จึงเป็น "ด่านกรองทิ้ง" ไม่ใช่ "ด่านหาเพิ่ม"

ไปป์ไลน์ต่อ 1 ข้อ:
  ด่าน 1 detector : ตัดสินประชด/ไม่ (prompt เรียบๆ เหมือน baseline -> รักษา recall)
  ด่าน 2 verifier : รันเฉพาะข้อที่ด่าน 1 ว่า "ประชด" เอา decision tree จาก rubric มากรอง
                    ถ้าไม่เข้าเงื่อนไข "เสแสร้ง" -> พลิกเป็น 0
  ข้อที่ด่าน 1 ว่า "ไม่ประชด" -> ผ่านเลย ไม่เรียก verifier (ประหยัด + พลิกกลับไม่ได้อยู่แล้ว)

วัด 4 มิติเหมือน baseline + จำนวน LLM call ต่อข้อ (ไว้ตอบ "คุ้มไหม")

รัน: ตั้ง PROVIDER แล้ว  python multiagent.py
"""

import json
import os
import statistics
import sys
import time

import pandas as pd

# ใช้ metric + ราคา + ชื่อโมเดล ชุดเดียวกับ baseline (สำคัญ: ห้าม divergent)
from baseline import metrics, MODELS, PRICE_PER_MTOK

# ================== ปรับได้ ==================
PROVIDER = "gpt"           # "claude" หรือ "gpt" -- ต้องตรงกับ baseline ที่จะเทียบ
VARIANT = "conservative"   # tag ในชื่อไฟล์ผล -- เปลี่ยน prompt verifier แล้วเปลี่ยน tag ด้วย กันทับผลเก่า
LIMIT = None
SLEEP_SEC = 0.2
SAVE_EVERY = 10
MAX_RETRY = 4
# opt-in: ใช้โมเดล "ถูกกว่า" เป็นด่านคัดกรอง (detector) ส่วน verifier ยังเป็นตัวหลัก
#   ค่าว่าง = ใช้ MODELS[PROVIDER] ทั้งสองด่านเหมือนเดิม (ผลเก่า reproduce ได้)
#   ตั้งทดลอง:  SCREENER_MODEL=gpt-4o-mini python multiagent.py
# หมายเหตุต้นทุน: PRICE_PER_MTOK เป็นราคาของ verifier -- ถ้า screener ถูกกว่า ต้นทุนจริงจะ "ต่ำกว่า" ที่พิมพ์
SCREENER_MODEL = os.getenv("SCREENER_MODEL") or None
# =============================================

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.environ.get("EVAL_DIR", HERE)
os.makedirs(EVAL_DIR, exist_ok=True)
GOLD_CSV = os.environ.get("GOLD_CSV", os.path.join(HERE, "gold.csv"))
# ถ้าใช้ screener คนละตัว ต่อ tag ในชื่อไฟล์ กัน overwrite ผล conservative ของเดิม
_SCREEN_TAG = f"_screen-{SCREENER_MODEL}" if SCREENER_MODEL else ""
PRED_CSV = os.path.join(EVAL_DIR, f"multiagent_preds_{PROVIDER}_{VARIANT}{_SCREEN_TAG}.csv")
BASE_PRED = os.path.join(EVAL_DIR, f"baseline_preds_{PROVIDER}.csv")  # ไว้เทียบ

# ---- ด่าน 1: detector -- prompt เรียบๆ ตัวเดียวกับ baseline (คุมให้ recall เท่าเดิม) ----
DETECT_SYS = """ตัดสินว่าข้อความภาษาไทยนี้ "ประชด/เสียดสี" หรือไม่
ประชด = เจตนาจริงตรงข้ามกับความหมายผิวเผิน เพื่อเหน็บหรือแสดงความไม่พอใจ

ตอบเป็น JSON เท่านั้น: {"label": "1" หรือ "0"}
1 = ประชด, 0 = ไม่ประชด"""

# ---- ด่าน 2: verifier -- decision tree จาก labeling_rubric.md ----
# หน้าที่: ได้ข้อที่ detector ว่า "ประชด" มาแล้ว ตรวจว่ามี "การเสแสร้ง" จริงไหม
# v2 conservative: default = คงไว้ พลิกทิ้งเฉพาะเมื่อ "ชัดเจน" ว่าไม่ใช่ประชด
#   (v1 เดิมพลิกทุกอย่างที่ก้ำกึ่ง -> เสีย recall 10 ข้อ ; ให้ประโยชน์แห่งความสงสัยกับ detector แทน)
VERIFY_SYS = """มีคนตัดสินว่าข้อความไทยนี้ "ประชด" มาแล้ว หน้าที่ของคุณคือ "ตรวจจับความผิดพลาดชัดๆ" เท่านั้น
ไม่ใช่ตัดสินใหม่ตั้งแต่ต้น -- คนก่อนหน้าจับประชดเก่งมาก (แทบไม่พลาด) ให้เชื่อเขาไว้ก่อน

พลิกเป็น "ไม่ประชด (0)" เฉพาะเมื่อ **มั่นใจชัดเจน** ว่าเข้าข้อใดข้อหนึ่งนี้:
  - บ่น/ตำหนิตรงๆ ล้วนๆ ไม่มีการแกล้งชมหรือแกล้งขอบคุณเลย  ["ลบตรงๆ ≠ ประชด"]
  - ชมจริงใจล้วนๆ ไม่มีนัยเหน็บ
  - รีวิวสมดุลตรงไปตรงมา: ชมจริงบางจุด ติจริงบางจุด อยู่ด้วยกันตามจริง ไม่ได้เสแสร้ง
  - แค่เล่าเหตุการณ์/อ้างคำพูดคนอื่น ไม่มีโทนเหน็บของผู้เขียนเอง

ถ้า **ไม่ชัด** ว่าเข้าข้อไหนข้างบน -- แม้จะก้ำกึ่ง หรืออ่านได้สองแง่ -- ให้ **คงไว้เป็นประชด (1)**
เหตุผล: ประชดไทยมักแนบเนียน อ่านได้สองแง่เป็นเรื่องปกติของประชด ถ้าลังเลแปลว่าน่าจะประชด

ตอบ JSON เท่านั้น: {"verdict": "1" หรือ "0"}  (1 = คงเป็นประชด, 0 = พลิกทิ้งเพราะชัดว่าไม่ใช่)"""

DETECT_SCHEMA = {"type": "object",
                 "properties": {"label": {"type": "string", "enum": ["0", "1"]}},
                 "required": ["label"], "additionalProperties": False}
VERIFY_SCHEMA = {"type": "object",
                 "properties": {"verdict": {"type": "string", "enum": ["0", "1"]}},
                 "required": ["verdict"], "additionalProperties": False}


# ---------------- LLM plumbing (รับ system + schema ได้ ต่างจาก baseline ที่ fix ไว้) ----------------

def _make_client():
    if PROVIDER not in MODELS:
        sys.exit(f"PROVIDER ต้องเป็น 'claude' หรือ 'gpt' ไม่ใช่ {PROVIDER!r}")
    pkg, key = {"claude": ("anthropic", "ANTHROPIC_API_KEY"),
                "gpt": ("openai", "OPENAI_API_KEY")}[PROVIDER]
    try:
        if PROVIDER == "claude":
            import anthropic
            return anthropic.Anthropic(max_retries=MAX_RETRY)
        from openai import OpenAI
        return OpenAI(max_retries=MAX_RETRY)
    except ImportError:
        sys.exit(f"ยังไม่ได้ติดตั้ง {pkg}  ->  pip install {pkg}")
    except Exception as e:
        hint = f"\n(ตั้งคีย์ยัง? {key})" if not os.getenv(key) else ""
        sys.exit(f"สร้าง client ไม่ได้: {type(e).__name__}: {e}{hint}")


def _ask(client, system, schema, key, text, model=None):
    """เรียก LLM 1 ครั้ง คืน (value, in_tok, out_tok) ; value=None ถ้าพัง
    model=None -> ใช้ MODELS[PROVIDER] (เดิม) ; ใส่ชื่อโมเดลเพื่อทับเฉพาะ call นี้ (เช่น screener ถูกๆ)"""
    mdl = model or MODELS[PROVIDER]
    if PROVIDER == "claude":
        r = client.messages.create(
            model=mdl, max_tokens=256, system=system,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": f"ข้อความ: {text}"}],
        )
        raw = next((b.text for b in r.content if b.type == "text"), "")
        in_tok, out_tok = r.usage.input_tokens, r.usage.output_tokens
    else:
        r = client.chat.completions.create(
            model=mdl, max_tokens=20,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": f"ข้อความ: {text}"}],
        )
        raw = (r.choices[0].message.content or "").strip()
        in_tok, out_tok = r.usage.prompt_tokens, r.usage.completion_tokens
    try:
        v = str(json.loads(raw).get(key, "")).strip()
    except json.JSONDecodeError:
        return None, in_tok, out_tok
    return (v if v in ("0", "1") else None), in_tok, out_tok


def run_pipeline(client, text):
    """detector -> (ถ้าประชด) verifier. คืน dict ครบทุกมิติ"""
    t0 = time.perf_counter()
    in_tok = out_tok = calls = 0
    try:
        det, i, o = _ask(client, DETECT_SYS, DETECT_SCHEMA, "label", text, model=SCREENER_MODEL)
        in_tok += i; out_tok += o; calls += 1
        if det is None:
            raise ValueError("detector ตอบเพี้ยน")

        verdict = ""
        if det == "1":
            # กรองซ้ำเฉพาะข้อที่ว่าประชด
            ver, i, o = _ask(client, VERIFY_SYS, VERIFY_SCHEMA, "verdict", text)
            in_tok += i; out_tok += o; calls += 1
            if ver is None:
                raise ValueError("verifier ตอบเพี้ยน")
            verdict = ver
            final = ver          # verifier พลิกได้: 1->คงไว้, 0->พลิกทิ้ง
        else:
            final = "0"          # detector ว่าไม่ประชด -> ผ่านเลย

        return {"pred": final, "detect": det, "verdict": verdict,
                "in_tok": in_tok, "out_tok": out_tok, "calls": calls,
                "latency_ms": round((time.perf_counter() - t0) * 1000), "err": ""}
    except Exception as e:
        return {"pred": "err", "detect": "", "verdict": "",
                "in_tok": in_tok, "out_tok": out_tok, "calls": calls,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "err": f"{type(e).__name__}: {e}"[:200]}


# ---------------- โหลด/เซฟ (resume ที่เริ่มจาก gold เสมอ เหมือน baseline) ----------------

COLS = ["pred", "detect", "verdict", "in_tok", "out_tok", "calls", "latency_ms", "err"]


def load():
    if not os.path.exists(GOLD_CSV):
        sys.exit(f"หาไฟล์ไม่เจอ: {GOLD_CSV}")
    g = pd.read_csv(GOLD_CSV, dtype=str).fillna("")
    g["label"] = g["label"].str.strip()
    g = g[g["label"].isin(["0", "1"])].reset_index(drop=True)
    for c in COLS:
        g[c] = ""
    if os.path.exists(PRED_CSV):
        old = pd.read_csv(PRED_CSV, dtype=str).fillna("")
        prev = {r["text"]: r for _, r in old.iterrows() if r.get("pred", "") in ("0", "1")}
        hit = 0
        for i in g.index:
            r = prev.get(g.at[i, "text"])
            if r is not None:
                for c in COLS:
                    g.at[i, c] = r.get(c, "")
                hit += 1
        print(f"ทำต่อจากเดิม: ใช้ผลเก่าได้ {hit} ข้อ")
    return g


def save(df):
    df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")


# ---------------- main ----------------

def _sum_int(series):
    return sum(int(x) for x in series if str(x).lstrip("-").isdigit())


def main():
    price = PRICE_PER_MTOK[PROVIDER]
    df = load()
    todo = [i for i in df.index if df.at[i, "pred"] not in ("0", "1")]
    if LIMIT:
        todo = todo[:LIMIT]

    print(f"\nMULTI-AGENT {PROVIDER} ({MODELS[PROVIDER]}) | gold {len(df)} | ต้องทำ {len(todo)} ข้อ")
    print("ไปป์ไลน์: detector -> verifier(เฉพาะข้อที่ว่าประชด)")
    if todo:
        client = _make_client()
        t0 = time.perf_counter()
        for n, idx in enumerate(todo, 1):
            out = run_pipeline(client, str(df.at[idx, "text"]))
            for c in COLS:
                df.at[idx, c] = str(out[c])
            if out["err"]:
                print(f"  [{n}/{len(todo)}] พัง: {out['err'][:70]}")
            if n % SAVE_EVERY == 0 or n == len(todo):
                save(df); print(f"  ...{n}/{len(todo)}")
            time.sleep(SLEEP_SEC)
        print(f"เวลารวมที่ยิง API: {time.perf_counter() - t0:.1f} วิ")
    save(df)

    # ---- วัดผล ----
    done = df[df["pred"].isin(["0", "1"])]
    bad = df[df["pred"] == "err"]
    acc, prec, rec, f1, (tn, fp, fn, tp) = metrics(done["label"].tolist(), done["pred"].tolist())
    in_tok, out_tok = _sum_int(df["in_tok"]), _sum_int(df["out_tok"])
    n_calls = _sum_int(df["calls"])
    lat = [int(x) for x in df["latency_ms"] if str(x).isdigit()]
    n_verified = int((df["verdict"] != "").sum())
    n_flipped = int(((df["detect"] == "1") & (df["verdict"] == "0")).sum())

    print("\n" + "=" * 58)
    print(f"MULTI-AGENT (detector->verifier) — {PROVIDER} / {MODELS[PROVIDER]}")
    print("=" * 58)
    print(f"\n[1] คุณภาพ (n = {len(done)})")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(f"  F1       : {f1:.3f}")
    print("\n  Confusion matrix (แถว=จริง, คอลัมน์=ทำนาย)")
    print("              pred:0   pred:1")
    print(f"    true:0     {tn:>5}    {fp:>5}")
    print(f"    true:1     {fn:>5}    {tp:>5}")
    print(f"\n  verifier ทำงาน {n_verified} ข้อ | พลิก 1->0 ไป {n_flipped} ข้อ")

    print(f"\n[2] ค่าใช้จ่าย")
    print(f"  LLM calls    : {n_calls}  (baseline = {len(done)} calls)")
    print(f"  input tokens : {in_tok:,}")
    print(f"  output tokens: {out_tok:,}")
    if price:
        cost = in_tok / 1e6 * price[0] + out_tok / 1e6 * price[1]
        print(f"  ค่าใช้จ่าย   : ${cost:.4f}")

    print(f"\n[3] เวลา")
    if lat:
        p95 = sorted(lat)[max(0, int(len(lat) * 0.95) - 1)]
        print(f"  latency/ข้อ: p50 {statistics.median(lat):.0f} ms | p95 {p95} ms | รวม {sum(lat)/1000:.1f} วิ")

    print(f"\n[4] ความน่าเชื่อ: err {len(bad)}/{len(df)}")
    for e in bad["err"].head(3):
        print(f"     {e[:70]}")

    # ---- เทียบ baseline ตรงๆ (ถ้ามีไฟล์) ----
    if os.path.exists(BASE_PRED):
        b = pd.read_csv(BASE_PRED, dtype=str).fillna("")
        b = b[b["pred"].isin(["0", "1"])]
        bacc, bprec, brec, bf1, _ = metrics(b["label"].tolist(), b["pred"].tolist())
        b_in, b_out = _sum_int(b["in_tok"]), _sum_int(b["out_tok"])
        b_cost = (b_in / 1e6 * price[0] + b_out / 1e6 * price[1]) if price else 0
        print("\n" + "=" * 58)
        print("เทียบกับ baseline (คุ้มไหม)")
        print("=" * 58)
        print(f"  {'':<12}{'baseline':>12}{'multi-agent':>14}{'ต่าง':>10}")
        print(f"  {'F1':<12}{bf1:>12.3f}{f1:>14.3f}{f1-bf1:>+10.3f}")
        print(f"  {'precision':<12}{bprec:>12.3f}{prec:>14.3f}{prec-bprec:>+10.3f}")
        print(f"  {'recall':<12}{brec:>12.3f}{rec:>14.3f}{rec-brec:>+10.3f}")
        if price:
            print(f"  {'ค่าใช้จ่าย $':<12}{b_cost:>12.4f}{cost:>14.4f}{f'{cost/b_cost:.2f}x':>10}")
        print(f"  {'LLM calls':<12}{len(b):>12}{n_calls:>14}{f'{n_calls/len(b):.2f}x':>10}")
        print("\n  → F1 ต้องเกิน 0.793 (CI บนของ baseline) ถึงจะอ้างว่าชนะจริง ไม่ใช่บังเอิญ")
        if f1 > 0.793:
            print(f"     F1 = {f1:.3f} > 0.793  ✓ ชนะเกินช่วงความบังเอิญ")
        else:
            print(f"     F1 = {f1:.3f} ยังไม่เกิน 0.793 -> ยังอ้างว่าดีกว่าไม่ได้")

    print(f"\nบันทึกที่: {os.path.basename(PRED_CSV)}")


if __name__ == "__main__":
    main()
