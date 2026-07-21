# -*- coding: utf-8 -*-
"""UI ติดป้ายแบบเร็ว (คีย์บอร์ดล้วน) — ขยาย gold set ให้ positive ถึง ~60-80 ข้อ

ทำไมต้องมี: human_review.py ใช้ได้แต่ช้า (พิมพ์ทีละข้อในเทอร์มินัล) เหลืองานอีก
754 ข้อ (harvest 470 + batch400 อีก 284) — หน้าเว็บนี้ให้กด 1/0/X ได้ทันที
เซฟทุกครั้งที่กด ปิดแล้วเปิดใหม่ทำต่อได้เลย

หลักการเดียวกับ human_review.py โหมด blind: **ไม่โชว์คำตอบ/ความมั่นใจของ LLM**
จนกว่าจะตัดสินเสร็จ เพื่อไม่ให้ป้ายของเราเอนตาม LLM (ปัญหา over-flag ที่เจอใน STEP 6)

รัน:   python Gold/label_ui.py            แล้วเปิด http://127.0.0.1:5001
รวม:   python Gold/label_ui.py --merge    สร้าง Gold/gold_v2.csv (ไม่แตะ gold.csv เดิม)
"""
import argparse
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# สองกองที่ค้างอยู่: ชื่อกอง -> (ไฟล์, คอลัมน์ป้าย)
QUEUES = {
    "harvest": (os.path.join(HERE, "harvest_to_review.csv"), "label"),
    "batch400": (os.path.join(ROOT, "to_label_reviewed.csv"), "human_label"),
    # กองสุ่มจริงจาก raw pool (ไม่ผ่านตัวกรองใดๆ) -- ไว้ทำ absolute number ที่อ่านได้
    "random": (os.path.join(HERE, "random_to_label.csv"), "label"),
}
GOLD = os.path.join(HERE, "gold.csv")
GOLD_V2 = os.path.join(HERE, "gold_v2.csv")


def load(queue):
    path, col = QUEUES[queue]
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={col: "object"})
    return df, path, col


def save_atomic(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def norm_text(t):
    return re.sub(r"\s+", " ", str(t)).strip()


def is_labeled(v):
    if pd.isna(v):
        return False
    return str(v).strip() not in ("", "nan")


def canon(v):
    """ป้ายในไฟล์มีทั้ง 1.0 / '1' / 'X' -> แปลงเป็น '1' | '0' | 'X'"""
    s = str(v).strip()
    if s in ("1", "1.0"):
        return "1"
    if s in ("0", "0.0"):
        return "0"
    return "X"


def progress(queue):
    df, _, col = load(queue)
    done = df[col].map(is_labeled)
    pos = df[col][done].map(canon).eq("1").sum()
    return {"total": len(df), "done": int(done.sum()), "pos": int(pos)}


def gold_pos():
    g = pd.read_csv(GOLD, encoding="utf-8-sig")
    return int(g["label"].sum())


# ----------------------------------------------------------------------
# merge: gold.csv + ป้ายใหม่จากทั้งสองกอง -> gold_v2.csv (dedupe ด้วยข้อความ)
# ----------------------------------------------------------------------
def merge():
    g = pd.read_csv(GOLD, encoding="utf-8-sig")
    seen = set(g["text"].map(norm_text))
    rows, skipped_dup, skipped_x = [], 0, 0

    for queue in QUEUES:
        df, _, col = load(queue)
        # batch400: ถ้ามี final_label (ผ่าน adjudicate แล้ว) ให้เชื่อ final_label ก่อน
        eff = df[col]
        if "final_label" in df.columns:
            eff = df["final_label"].where(df["final_label"].map(is_labeled), df[col])
        for _, r in df[eff.map(is_labeled)].iterrows():
            lab = canon(eff.loc[r.name])
            if lab == "X":
                skipped_x += 1
                continue
            key = norm_text(r["text"])
            if key in seen:
                skipped_dup += 1
                continue
            seen.add(key)
            rows.append({
                "text": r["text"], "label": int(lab),
                "source": r.get("source", queue),
                "suspect_score": r.get("suspect_score", ""),
                "signals": r.get("signals", ""),
            })

    out = pd.concat([g, pd.DataFrame(rows, columns=g.columns)], ignore_index=True) if rows else g
    save_atomic(out, GOLD_V2)
    n, pos = len(out), int(out["label"].sum())
    print(f"gold.csv เดิม: {len(g)} ข้อ (ประชด {gold_pos()})")
    print(f"เพิ่มใหม่: {len(rows)} ข้อ | ข้าม X: {skipped_x} | ข้ามซ้ำ: {skipped_dup}")
    print(f"-> {GOLD_V2}: {n} ข้อ (ประชด {pos} / ไม่ประชด {n - pos})")
    if pos < 60:
        print(f"   เป้า ~60-80 positive: ยังขาดอีก {60 - pos}")


# ----------------------------------------------------------------------
# หน้าเว็บ
# ----------------------------------------------------------------------
PAGE = """<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ติดป้ายประชด — label_ui</title>
<style>
:root{--bg:#FBF8F3;--card:#fff;--ink:#241E15;--mut:#8A7E6C;--gold:#AE9569;
--green:#4C8F5D;--red:#B4574F;--lav:#7B72B9;--line:#E8E1D4}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,"Segoe UI",Roboto,"Noto Sans Thai",sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:20px;display:grid;grid-template-columns:1fr 300px;gap:18px}
@media(max-width:860px){.wrap{grid-template-columns:1fr}}
h1{font-size:17px;margin:0 0 4px}
.top{grid-column:1/-1;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.tabs{display:flex;gap:6px}
.tab{border:1.5px solid var(--line);background:var(--card);border-radius:20px;
padding:6px 14px;font-size:13px;cursor:pointer}
.tab.on{border-color:var(--gold);background:var(--gold);color:#fff;box-shadow:0 0 0 3px rgba(174,149,105,.35);font-weight:700}
.curq{grid-column:1/-1;font-size:13px;color:var(--gold);font-weight:700;margin:-4px 0 0}
.bar{flex:1;min-width:180px;height:10px;background:var(--line);border-radius:6px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--gold)}
.stat{font-size:12.5px;color:var(--mut);white-space:nowrap}
.stat b{color:var(--ink)}
.card{background:var(--card);border:1.5px solid var(--line);border-radius:14px;padding:22px}
.text{font-size:21px;line-height:1.75;min-height:150px;white-space:pre-wrap;word-break:break-word}
.meta{font-size:12px;color:var(--mut);margin-top:10px}
.btns{display:flex;gap:10px;margin-top:18px;flex-wrap:wrap}
button{font-family:inherit;font-size:15px;border:0;border-radius:10px;padding:12px 18px;
cursor:pointer;color:#fff}
button:focus-visible{outline:3px solid var(--lav);outline-offset:2px}
.b1{background:var(--red)}.b0{background:var(--green)}.bx{background:var(--mut)}
.ghost{background:transparent;color:var(--mut);border:1.5px solid var(--line)}
kbd{background:rgba(255,255,255,.25);border-radius:4px;padding:0 6px;font-family:inherit}
.ghost kbd{background:var(--line);color:var(--ink)}
.done-tag{display:inline-block;border-radius:14px;padding:3px 12px;font-size:13px;color:#fff;margin-bottom:10px}
.note{width:100%;margin-top:14px;border:1.5px solid var(--line);border-radius:8px;
padding:8px 10px;font:inherit;font-size:13px}
.side{font-size:13px;line-height:1.6}
.side .card{padding:16px 18px;margin-bottom:14px}
.side h2{font-size:13.5px;margin:0 0 8px;color:var(--gold)}
.side ol{margin:0;padding-left:20px}
.side li{margin-bottom:5px}
.flash{position:fixed;top:14px;left:50%;transform:translateX(-50%);background:var(--ink);
color:#fff;border-radius:20px;padding:8px 20px;font-size:13px;opacity:0;transition:opacity .25s}
.flash.show{opacity:.95}
.reveal{color:var(--lav);cursor:pointer;font-size:12.5px;background:none;border:none;padding:0}
.fin{text-align:center;padding:60px 10px;font-size:19px}
</style></head><body>
<div class="wrap">
  <div class="top">
    <h1>ติดป้ายประชด</h1>
    <div class="tabs">
      <button class="tab" id="tab-harvest" onclick="setQueue('harvest')">harvest (470)</button>
      <button class="tab" id="tab-batch400" onclick="setQueue('batch400')">batch400 (284)</button>
      <button class="tab" id="tab-random" onclick="setQueue('random')">random (250)</button>
    </div>
    <div class="bar"><i id="bar" style="width:0%"></i></div>
    <span class="stat" id="stat"></span>
    <button class="ghost" id="quitbtn" onclick="quitApp()" title="ทุกอย่างเซฟอยู่แล้วทุกครั้งที่กด ปุ่มนี้ปิดเซิร์ฟเวอร์ให้เรียบร้อย">บันทึก &amp; ปิด</button>
  </div>

  <div class="card" id="main">
    <span class="done-tag" id="donetag" style="display:none"></span>
    <div class="text" id="text">กำลังโหลด…</div>
    <div class="meta" id="meta"></div>
    <button class="reveal" id="reveal" style="display:none" onclick="doReveal()">เผยความเห็น LLM (หลังตัดสินแล้วเท่านั้น)</button>
    <div class="btns">
      <button class="b1" onclick="label('1')">ประชด <kbd>1</kbd></button>
      <button class="b0" onclick="label('0')">ไม่ประชด <kbd>0</kbd></button>
      <button class="bx" onclick="label('X')">บอกไม่ได้ <kbd>X</kbd></button>
      <button class="ghost" onclick="undo()">ย้อนกลับ <kbd>U</kbd></button>
      <button class="ghost" onclick="nav(-1)">ก่อนหน้า <kbd>←</kbd></button>
      <button class="ghost" onclick="nav(1)">ถัดไป <kbd>→</kbd></button>
    </div>
    <input class="note" id="note" placeholder="โน้ต (ไม่บังคับ) — Enter เพื่อบันทึกโน้ตอย่างเดียว">
  </div>

  <div class="side">
    <div class="card">
      <h2>นิยาม: ประชด (1) = แกล้งชม/แกล้งขอบคุณ ทั้งที่หมายตรงข้าม</h2>
      ต้องมี <b>การเสแสร้ง</b> (พูดบวก-หมายลบ) เท่านั้น
    </div>
    <div class="card">
      <h2>ต้นไม้ตัดสิน</h2>
      <ol>
        <li>บ่น/ด่า<b>ตรงๆ</b> ไม่แกล้งชม → <b>0</b></li>
        <li>ชมจริง ไม่มีนัยแฝง → <b>0</b></li>
        <li>ชมจริง+ติจริง (รีวิวสมดุล) → <b>0</b></li>
        <li>เล่าเหตุการณ์/อ้างคำคนอื่น → <b>0</b></li>
        <li>พูดบวกแต่หมายลบเพื่อเหน็บ → <b>1</b></li>
        <li>ต้องใช้บริบทนอกข้อความ → <b>X</b></li>
      </ol>
    </div>
    <div class="card">
      <h2>กันพลาด</h2>
      <ol>
        <li>ลบตรงๆ ≠ ประชด</li>
        <li>อย่าเดานัยแฝงเกินข้อความ</li>
        <li>ชมเว่อร์ ≠ ประชด</li>
        <li>เสแสร้งไม่ชัด → เอน 0</li>
      </ol>
    </div>
  </div>
</div>
<div class="flash" id="flash"></div>
<script>
let Q=localStorage.getItem('lastQueue')||'random', IDX=null, REVEALED=false;
const $=id=>document.getElementById(id);
// เก็บโครงการ์ดตั้งต้นไว้ -- ตอนกองไหนจบเราเขียนทับ #main ทิ้ง ถ้าไม่คืนกลับ
// การสลับไปกองที่ยังไม่จบจะพัง (element หาย -> JS error -> หน้าค้าง)
const MAIN_HTML=$('main').innerHTML;

function flash(m){const f=$('flash');f.textContent=m;f.classList.add('show');
  clearTimeout(f._t);f._t=setTimeout(()=>f.classList.remove('show'),1200);}

async function api(path,body){
  const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)}:{});
  return r.json();
}

function render(s){
  IDX=s.index;REVEALED=false;
  $('bar').style.width=(100*s.done/s.total)+'%';
  $('stat').innerHTML=`ทำแล้ว <b>${s.done}</b>/${s.total} · เจอประชดกองนี้ <b>${s.pos}</b>`
    +` · gold รวม <b>${s.gold_pos}+${s.new_pos}</b>/60 เป้า`;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  $('tab-'+Q).classList.add('on');
  if(s.queues){for(const [name,left] of Object.entries(s.queues)){
    const t=$('tab-'+name); if(t) t.textContent=name+' ('+(left?'เหลือ '+left:'ครบแล้ว')+')';}}
  if(s.index===null){
    $('main').innerHTML='<div class="fin">กองนี้ครบแล้ว · สลับไปกองอื่นด้านบน หรือรัน <code>python Gold/label_ui.py --merge</code></div>';
    return;
  }
  if(!$('text')) $('main').innerHTML=MAIN_HTML;   // คืนโครงการ์ดหลังหน้าจบกอง
  $('text').textContent=s.text;
  $('meta').textContent=`ข้อ ${s.index+1} จาก ${s.total}`;
  $('note').value=s.note||'';
  const t=$('donetag');
  if(s.current){t.style.display='';t.textContent='ป้ายปัจจุบัน: '+s.current;
    t.style.background=s.current==='1'?'var(--red)':s.current==='0'?'var(--green)':'var(--mut)';
    $('reveal').style.display='';}
  else{t.style.display='none';$('reveal').style.display='none';}
  $('reveal').textContent='เผยความเห็น LLM (หลังตัดสินแล้วเท่านั้น)';
}

async function setQueue(q){Q=q;localStorage.setItem('lastQueue',q);render(await api(`/api/state?queue=${Q}`));}
async function label(v){if(IDX===null)return;
  render(await api('/api/label',{queue:Q,index:IDX,value:v,note:$('note').value}));
  flash({'1':'✓ ประชด','0':'✓ ไม่ประชด','X':'✓ บอกไม่ได้'}[v]);}
async function undo(){const s=await api('/api/undo',{queue:Q});
  if(s.error){flash(s.error);return;}render(s);flash('ย้อนแล้ว');}
async function nav(d){render(await api('/api/state?queue='+Q+'&index='+(IDX+d)));}
async function doReveal(){if(REVEALED)return;const s=await api(`/api/hint?queue=${Q}&index=${IDX}`);
  $('reveal').textContent='LLM: '+s.hint;REVEALED=true;}
async function quitApp(){
  const s=await api('/api/quit',{});
  document.body.innerHTML=`<div class="fin" style="padding-top:120px">✅ ${s.msg}<br>
    <span style="font-size:14px;color:var(--mut)">ปิดแท็บนี้ได้เลย — เปิดใหม่เมื่อไหร่ก็ทำต่อจากเดิม</span></div>`;
}

document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'){
    if(e.key==='Enter'){api('/api/note',{queue:Q,index:IDX,note:$('note').value});
      flash('บันทึกโน้ต');e.target.blur();}
    return;}
  if(e.key==='1')label('1');else if(e.key==='0')label('0');
  else if(e.key==='x'||e.key==='X')label('X');
  else if(e.key==='u'||e.key==='U')undo();
  else if(e.key==='ArrowLeft')nav(-1);else if(e.key==='ArrowRight')nav(1);
});
setQueue(Q);
</script></body></html>"""


def run_server(port):
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    history = []  # (queue, index, ป้ายเดิม, โน้ตเดิม) สำหรับ undo

    def state(queue, index=None):
        """สถานะที่หน้าเว็บใช้วาด: ข้อความปัจจุบัน + ความคืบหน้า
        index=None -> ไปข้อที่ยังไม่มีป้ายข้อแรก"""
        df, _, col = load(queue)
        done = df[col].map(is_labeled)
        if index is None:
            todo = df.index[~done]
            index = int(todo[0]) if len(todo) else None
        else:
            index = max(0, min(int(index), len(df) - 1))
        p = progress(queue)
        all_p = {q: progress(q) for q in QUEUES}
        new_pos = sum(v["pos"] for v in all_p.values())
        out = {**p, "index": index, "gold_pos": gold_pos(), "new_pos": new_pos,
               "queues": {q: v["total"] - v["done"] for q, v in all_p.items()}}
        if index is not None:
            r = df.loc[index]
            out["text"] = str(r["text"])
            out["note"] = "" if pd.isna(r.get("note", "")) else str(r.get("note", ""))
            out["current"] = canon(r[col]) if is_labeled(r[col]) else None
        return out

    @app.get("/")
    def page():
        return PAGE

    @app.get("/api/state")
    def api_state():
        q = request.args.get("queue", "harvest")
        idx = request.args.get("index")
        return jsonify(state(q, None if idx is None else int(idx)))

    @app.post("/api/label")
    def api_label():
        d = request.json
        q, idx, val = d["queue"], int(d["index"]), str(d["value"])
        df, path, col = load(q)
        history.append((q, idx, df.at[idx, col], df.at[idx, "note"] if "note" in df.columns else ""))
        df.at[idx, col] = val
        if "note" in df.columns and d.get("note"):
            df.at[idx, "note"] = d["note"]
        save_atomic(df, path)
        return jsonify(state(q))

    @app.post("/api/note")
    def api_note():
        d = request.json
        df, path, _ = load(d["queue"])
        if "note" in df.columns:
            df.at[int(d["index"]), "note"] = d["note"]
            save_atomic(df, path)
        return jsonify({"ok": True})

    @app.post("/api/undo")
    def api_undo():
        if not history:
            return jsonify({"error": "ไม่มีอะไรให้ย้อน"})
        q, idx, old_label, old_note = history.pop()
        df, path, col = load(q)
        df.at[idx, col] = old_label
        if "note" in df.columns:
            df.at[idx, "note"] = old_note
        save_atomic(df, path)
        return jsonify(state(q, idx))

    @app.post("/api/quit")
    def api_quit():
        """ปิดเซิร์ฟเวอร์จากหน้าเว็บ — ข้อมูลเซฟไปแล้วทุกครั้งที่กด จึงแค่สรุปแล้วดับเครื่อง"""
        import threading

        parts = []
        for q in QUEUES:
            p = progress(q)
            parts.append(f"{q} {p['done']}/{p['total']} (ประชด {p['pos']})")
        threading.Timer(0.6, os._exit, args=[0]).start()
        return jsonify({"msg": "เซฟครบแล้ว · " + " · ".join(parts)})

    @app.get("/api/hint")
    def api_hint():
        """ความเห็น LLM — ให้ดูได้เฉพาะข้อที่ติดป้ายแล้ว (กัน anchor bias)"""
        q, idx = request.args["queue"], int(request.args["index"])
        df, _, col = load(q)
        if not is_labeled(df.at[idx, col]):
            return jsonify({"hint": "ยังไม่ได้ตัดสิน — ตัดสินก่อนแล้วค่อยดู"})
        r = df.loc[idx]
        bits = []
        if "llm_conf" in df.columns and not pd.isna(r.get("llm_conf")):
            bits.append(f"มั่นใจว่าประชด {float(r['llm_conf']):.2f}")
        if "llm_label" in df.columns and not pd.isna(r.get("llm_label")):
            bits.append(f"ป้าย {r['llm_label']}")
        if "llm_reason" in df.columns and not pd.isna(r.get("llm_reason")):
            bits.append(str(r["llm_reason"])[:160])
        return jsonify({"hint": " · ".join(bits) or "ไม่มีข้อมูล LLM สำหรับข้อนี้"})

    print(f"เปิด http://127.0.0.1:{port} — กด 1/0/X, U=ย้อน, ←/→=เลื่อนดู")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true", help="รวมป้ายใหม่ -> gold_v2.csv")
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()
    if args.merge:
        merge()
    else:
        run_server(args.port)
