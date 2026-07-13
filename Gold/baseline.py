# -*- coding: utf-8 -*-
"""
Baseline: เอเจนต์เดี่ยว LLM zero-shot classifier -> วัด 4 มิติ

  1. คุณภาพ  : accuracy / precision / recall / F1 (ฝั่งประชด) + confusion matrix
  2. ค่าใช้จ่าย: input/output tokens (+ ประมาณค่าเงิน ถ้าใส่ราคาไว้)
  3. เวลา    : latency ต่อข้อ (p50 / p95) + เวลารวม
  4. ความน่าเชื่อ: นับข้อที่ LLM ตอบไม่ได้/ตอบเพี้ยน แยกออกจากคำทำนายจริง

รันได้สองเจ้า (สลับที่ PROVIDER) เพื่อดูว่า bias จาก gold มีผลแค่ไหน:
  - "claude" : anthropic SDK, claude-opus-4-8
  - "gpt"    : openai SDK, gpt-4o   <- เจ้าเดียวกับที่ใช้ขุด gold (ดู PROVENANCE.md)

อินพุต : gold.csv (text, label)
เอาต์พุต: baseline_preds_<provider>.csv  (ทุกข้อ + pred + tokens + latency)

ติดตั้ง: pip install pandas anthropic openai
ตั้งคีย์: ANTHROPIC_API_KEY หรือ OPENAI_API_KEY
รัน    : python baseline.py
"""

import json
import os
import statistics
import sys
import time

import pandas as pd

# ================== ปรับได้ ==================
PROVIDER = "gpt"           # "claude" หรือ "gpt"
LIMIT = None               # จำกัดจำนวนข้อ (ไว้ทดลองก่อนรันเต็ม) เช่น 10 ; None = ทั้งหมด
SLEEP_SEC = 0.2
SAVE_EVERY = 10            # เซฟทุกกี่ข้อ (กันงานหายกลางคัน)
MAX_RETRY = 4
# =============================================

MODELS = {"claude": "claude-opus-4-8", "gpt": "gpt-4o"}

# ราคา USD ต่อ 1 ล้าน token (input, output)
# claude-opus-4-8 = 5 / 25  (จากเอกสาร Anthropic)
# gpt-4o = 2.50 / 10.00  (เช็ค OpenAI pricing 2026-07 -- standard/real-time tier)
PRICE_PER_MTOK = {"claude": (5.0, 25.0), "gpt": (2.50, 10.0)}

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_CSV = os.path.join(HERE, "gold.csv")
PRED_CSV = os.path.join(HERE, f"baseline_preds_{PROVIDER}.csv")

# prompt เรียบๆ ตั้งใจ -- เส้นฐานที่ยุติธรรม บอกแค่นิยาม ไม่ยัดกฎย่อยเหมือนตอน pre-label
SYSTEM = """ตัดสินว่าข้อความภาษาไทยนี้ "ประชด/เสียดสี" หรือไม่
ประชด = เจตนาจริงตรงข้ามกับความหมายผิวเผิน เพื่อเหน็บหรือแสดงความไม่พอใจ

ตอบเป็น JSON เท่านั้น: {"label": "1" หรือ "0"}
1 = ประชด, 0 = ไม่ประชด"""

LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["0", "1"]}},
    "required": ["label"],
    "additionalProperties": False,
}


# ---------------- ตัวเรียก LLM: คืน (label, in_tok, out_tok) ----------------

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


def _call_claude(client, text):
    # ไม่เปิด thinking: baseline ต้องเป็น LLM เรียกครั้งเดียวเรียบๆ ไว้เทียบกับ multi-agent
    # ไม่ใส่ temperature: claude-opus-4-8 ไม่รับ (400)
    # output_config บังคับรูปแบบ JSON -> ตัดปัญหา parse เพี้ยนออกไปเกือบหมด
    r = client.messages.create(
        model=MODELS["claude"],
        max_tokens=256,
        system=SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
        messages=[{"role": "user", "content": f"ข้อความ: {text}"}],
    )
    raw = next((b.text for b in r.content if b.type == "text"), "")
    return raw, r.usage.input_tokens, r.usage.output_tokens


def _call_gpt(client, text):
    r = client.chat.completions.create(
        model=MODELS["gpt"],
        max_tokens=20,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"ข้อความ: {text}"},
        ],
    )
    raw = (r.choices[0].message.content or "").strip()
    return raw, r.usage.prompt_tokens, r.usage.completion_tokens


def predict_one(client, text):
    """คืน dict: pred ("0"/"1"/"err"), in_tok, out_tok, latency_ms, err
    หมายเหตุสำคัญ: ถ้าพัง -> pred="err" ไม่ใช่ "0"
    การเดา "0" เวลาพัง จะถูกโดยบังเอิญบ่อย (gold เป็น 0 ถึง 76%) -> ตัวเลขสวยเกินจริง"""
    t0 = time.perf_counter()
    try:
        call = _call_claude if PROVIDER == "claude" else _call_gpt
        raw, in_tok, out_tok = call(client, text)
    except Exception as e:  # SDK retry หมดแล้วยังพัง
        return {"pred": "err", "in_tok": 0, "out_tok": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "err": f"{type(e).__name__}: {e}"[:200]}

    latency_ms = round((time.perf_counter() - t0) * 1000)
    try:
        lab = str(json.loads(raw).get("label", "")).strip()
    except json.JSONDecodeError:
        return {"pred": "err", "in_tok": in_tok, "out_tok": out_tok,
                "latency_ms": latency_ms, "err": f"bad json: {raw[:80]}"}
    if lab not in ("0", "1"):
        return {"pred": "err", "in_tok": in_tok, "out_tok": out_tok,
                "latency_ms": latency_ms, "err": f"bad label: {lab[:40]}"}
    return {"pred": lab, "in_tok": in_tok, "out_tok": out_tok,
            "latency_ms": latency_ms, "err": ""}


# ---------------- metric (เขียนเอง ไม่ต้องพึ่ง sklearn) ----------------

def metrics(y_true, y_pred, positive="1"):
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))
    tn = sum(t != positive and p != positive for t, p in zip(y_true, y_pred))
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, prec, rec, f1, (tn, fp, fn, tp)


# ---------------- โหลด / เซฟ (resume ที่ไม่ลืมข้อใหม่ใน gold) ----------------

COLS = ["pred", "in_tok", "out_tok", "latency_ms", "err"]


def load():
    if not os.path.exists(GOLD_CSV):
        sys.exit(f"หาไฟล์ไม่เจอ: {GOLD_CSV}")
    gold = pd.read_csv(GOLD_CSV, dtype=str).fillna("")
    gold["label"] = gold["label"].str.strip()
    gold = gold[gold["label"].isin(["0", "1"])].reset_index(drop=True)

    # เริ่มจาก gold เสมอ แล้วค่อยเอาคำทำนายเก่ามาแปะตาม text
    # (ของเดิมอ่านจากไฟล์ preds อย่างเดียว -> ข้อที่เพิ่งเติมเข้า gold จะไม่เคยถูกทำนาย)
    for c in COLS:
        gold[c] = ""
    if os.path.exists(PRED_CSV):
        old = pd.read_csv(PRED_CSV, dtype=str).fillna("")
        prev = {r["text"]: r for _, r in old.iterrows() if r.get("pred", "") in ("0", "1")}
        hit = 0
        for i in gold.index:
            r = prev.get(gold.at[i, "text"])
            if r is not None:
                for c in COLS:
                    gold.at[i, c] = r.get(c, "")
                hit += 1
        print(f"ทำต่อจากเดิม: ใช้คำทำนายเก่าได้ {hit} ข้อ (จาก {len(old)} แถวในไฟล์เก่า)")
    return gold


def save(df):
    df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")


# ---------------- main ----------------

def main():
    price = PRICE_PER_MTOK[PROVIDER]
    df = load()
    todo = [i for i in df.index if df.at[i, "pred"] not in ("0", "1")]
    if LIMIT:
        todo = todo[:LIMIT]

    print(f"\nBASELINE {PROVIDER} ({MODELS[PROVIDER]}) | gold {len(df)} ข้อ | ต้องทำนาย {len(todo)} ข้อ")
    if not todo:
        print("ทำนายครบแล้ว ข้ามไปวัดผล")
    else:
        client = _make_client()
        t_start = time.perf_counter()
        for n, idx in enumerate(todo, 1):
            out = predict_one(client, str(df.at[idx, "text"]))
            for c in COLS:
                df.at[idx, c] = str(out[c])
            if out["err"]:
                print(f"  [{n}/{len(todo)}] พัง: {out['err'][:70]}")
            if n % SAVE_EVERY == 0 or n == len(todo):
                save(df)
                print(f"  ...{n}/{len(todo)}")
            time.sleep(SLEEP_SEC)
        print(f"เวลารวมที่ยิง API: {time.perf_counter() - t_start:.1f} วิ")
    save(df)

    # ---- วัดผล: ตัดข้อที่พัง (err) ออกจากการคิด metric แล้วรายงานแยก ----
    done = df[df["pred"].isin(["0", "1"])]
    bad = df[df["pred"] == "err"]
    acc, prec, rec, f1, (tn, fp, fn, tp) = metrics(done["label"].tolist(), done["pred"].tolist())

    lat = [int(x) for x in df["latency_ms"] if str(x).isdigit()]
    in_tok = sum(int(x) for x in df["in_tok"] if str(x).isdigit())
    out_tok = sum(int(x) for x in df["out_tok"] if str(x).isdigit())

    print("\n" + "=" * 58)
    print(f"BASELINE เอเจนต์เดี่ยว — {PROVIDER} / {MODELS[PROVIDER]}")
    print("=" * 58)
    print(f"\n[1] คุณภาพ  (n = {len(done)} ข้อที่ทำนายสำเร็จ)")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}   (คลาสบวก = ประชด)")
    print(f"  Recall   : {rec:.3f}")
    print(f"  F1       : {f1:.3f}")
    print("\n  Confusion matrix (แถว=จริง, คอลัมน์=ทำนาย)")
    print("              pred:0   pred:1")
    print(f"    true:0     {tn:>5}    {fp:>5}")
    print(f"    true:1     {fn:>5}    {tp:>5}")

    print(f"\n[2] ค่าใช้จ่าย")
    print(f"  input tokens : {in_tok:,}")
    print(f"  output tokens: {out_tok:,}")
    if price:
        cost = in_tok / 1e6 * price[0] + out_tok / 1e6 * price[1]
        print(f"  ค่าใช้จ่ายประมาณ: ${cost:.4f}  (${price[0]}/${price[1]} ต่อ 1M token)")
    else:
        print(f"  (ยังไม่ได้ใส่ราคาของ {PROVIDER} ใน PRICE_PER_MTOK -> ข้ามการคิดเงิน)")

    print(f"\n[3] เวลา")
    if lat:
        p95 = sorted(lat)[max(0, int(len(lat) * 0.95) - 1)]
        print(f"  latency/ข้อ: p50 {statistics.median(lat):.0f} ms | p95 {p95} ms | รวม {sum(lat)/1000:.1f} วิ")

    print(f"\n[4] ความน่าเชื่อ")
    print(f"  ข้อที่ทำนายไม่สำเร็จ (err): {len(bad)}/{len(df)}")
    if len(bad):
        print("  -> ข้อพวกนี้ถูกตัดออกจาก metric แล้ว (ไม่ได้เดาเป็น 0)")
        for e in bad["err"].head(3):
            print(f"     {e[:70]}")

    print(f"\nบันทึกที่: {os.path.basename(PRED_CSV)}")
    print("→ เก็บ F1 + token + latency ชุดนี้ไว้เป็นเส้นฐาน เทียบกับ multi-agent")
    print("→ อ่าน PROVENANCE.md ก่อนตีความ recall: gold ฝั่งประชดถูกขุดโดย GPT-4o + Claude")


if __name__ == "__main__":
    main()
