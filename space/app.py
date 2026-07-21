# -*- coding: utf-8 -*-
"""Thai sarcasm detection, hosted version, 100% free, every answer explainable

## Why this version has "no neural model"

The original plan was WangchanBERTa fine-tuned on the 127-item gold, but **measurement showed it doesn't work in practice**:

| | on gold (its training data) | on unseen new sentences |
|---|---|---|
| mean prob when *sarcastic* | 0.801 | 0.838 |
| mean prob when *not sarcastic* | 0.238 | 0.810 |
| **gap** | **+0.563** | **+0.028** |

On its training data it separates well, but on new sentences it **answers "sarcastic" for almost everything**
Tested on 10 clear new sentences (5 clearly sarcastic / 5 clearly normal):
  - WangchanBERTa: **5/10** (= random guessing -- because it answers "sarcastic" for all)
  - lexical cues alone: **8/10**
No threshold helps (best 7/10 at 0.85, which is unstable)

Cause: 127 sentences are too few to truly learn "sarcasm", so it memorizes the training set instead
Matches finding 12, which already measured precision dropping 0.68 -> 0.40 across domains

=> **so the hosted version uses purely lexical cues**, which (a) are genuinely more accurate on new text
   (b) explain every answer (c) need no 405 MB model download -> the page loads in 2 seconds
   (d) match finding 14: the single regex `555` gets F1 0.590, beating every 7-8B open model

## Three answers, not two
If no cue is found, the system answers **"can't tell"** rather than guessing "not sarcastic"
Because sarcasm with no surface signal is real (2 of 10 in testing), so guessing would be lying to the user
This structure mirrors the research's router: confident -> answer / unsure -> admit it doesn't know
"""
import math
import re

import gradio as gr

# ---- cue: (name, regex, lift, description) · lift measured from the 127-item gold (finding 14) ----
# lift > 1 = finding it raises the sarcasm odds above average · < 1 = the signal points the other way
CUES = [
    ("555",        r"555",       2.46, "“555” = เสียงหัวเราะ มักมาคู่กับการเหน็บ"),
    ("??",         r"[?]{2,}",   2.54, "เครื่องหมายคำถามซ้ำ = ตั้งคำถามเชิงประชด"),
    ("ตัวอักษรยืด", r"(.)\1{2,}", 1.69, "เช่น “ดีมากกกก” = เน้นเกินจริง"),
    ("จ้า",         r"จ้า",       1.32, "น้ำเสียงกันเอง มักใช้เหน็บ"),
    ("ค่ะ",         r"ค่ะ",       0.40, "คำสุภาพ พบในข้อความ*ไม่*ประชดมากกว่า"),
    ("นะคะ",        r"นะคะ",      0.22, "คำสุภาพ สัญญาณค่อนไปทาง*ไม่*ประชด"),
    ("ครับ",        r"ครับ",      0.05, "ในชุดข้อมูลนี้ “ครับ” ไม่เคยอยู่ในข้อความประชดเลย"),
]


def find_cues(text):
    return [(n, lift, why) for n, pat, lift, why in CUES if re.search(pat, text)]


def cue_score(hits):
    """combine signals as a sum of log(lift) -- positive = leans sarcastic, negative = leans normal
    not a trained model, just a straightforward combination of measured lifts (hand-verifiable)"""
    return sum(math.log(max(l, 0.05)) for _, l, _ in hits)


def analyse(text):
    text = (text or "").strip()
    if not text:
        return "### ⬅️ พิมพ์ข้อความภาษาไทยทางซ้าย แล้วกด **ตรวจสอบ**"

    hits = find_cues(text)
    s = cue_score(hits)

    if not hits:
        icon, label, note = "🤔", "บอกไม่ได้", (
            "ไม่พบสัญญาณผิวๆ ที่ระบบรู้จักเลย ระบบนี้อ่านจาก cue เชิงคำเท่านั้น "
            "จึงตอบไม่ได้ว่าประโยคนี้ประชดหรือไม่ **และจะไม่เดาให้**"
        )
    elif s > 0:
        icon, label, note = "🌀", "น่าจะประชด", "พบสัญญาณที่มักปรากฏในข้อความประชด"
    else:
        icon, label, note = "💬", "น่าจะไม่ประชด", "พบสัญญาณที่มักปรากฏในข้อความปกติ (ไม่ประชด)"

    strength = min(abs(s) / 3.0, 1.0)
    filled = int(round(strength * 18))
    bar = "█" * filled + "░" * (18 - filled)

    out = [f"## {icon} {label}", "", note, ""]
    if hits:
        out.append(f"`{bar}` ความแรงของสัญญาณ · คะแนนรวม `{s:+.2f}`")
        out.append("")
        out.append("### สัญญาณที่พบ")
        out.append("| cue | ทิศทาง | ทำไม |")
        out.append("|---|---|---|")
        for name, lift, why in sorted(hits, key=lambda h: -h[1]):
            if lift >= 1.5:
                arrow = f"↑↑ ประชด `{lift:.2f}×`"
            elif lift > 1.0:
                arrow = f"↑ ประชด `{lift:.2f}×`"
            else:
                arrow = f"↓ ไม่ประชด `{lift:.2f}×`"
            out.append(f"| `{name}` | {arrow} | {why} |")
        out.append("")
        out.append("<sub>“2.46×” = พบ cue นี้แล้วโอกาสเป็นประชดสูงกว่าค่าเฉลี่ย 2.46 เท่า "
                   "(วัดจากชุดข้อมูล 127 ประโยค)</sub>")
    return "\n".join(out)


EXAMPLES = [
    "บริการดีมากกก รอแค่ชั่วโมงเดียวเอง 555",
    "อาหารอร่อยมากค่ะ พนักงานน่ารัก จะกลับมาอีกแน่นอน",
    "ขอบคุณที่ยกเลิกออเดอร์ตอนรอมา 2 ชั่วโมง จ้า",
    "ราคาสมเหตุสมผล ของสดใหม่ แนะนำครับ",
    "ดีจังเลย ฝนตกตอนลืมร่มพอดี",
]

CSS = """
.gradio-container { max-width: 1080px !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="ตรวจจับประชดภาษาไทย", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown(
        """
        # 🌀 ตรวจจับประชดภาษาไทย · Thai Sarcasm Detector

        พิมพ์ข้อความภาษาไทย ระบบจะบอกว่า **ประชด / ไม่ประชด / บอกไม่ได้**
        พร้อมแสดง **เหตุผล** ว่าตัดสินจากสัญญาณอะไรบ้าง

        ฟรี 100% · ไม่เรียก OpenAI · ไม่ต้องใช้ API key · ทุกคำตอบตรวจสอบย้อนได้
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(
                label="ข้อความภาษาไทย",
                placeholder="เช่น: บริการดีมากกก รอแค่ชั่วโมงเดียวเอง 555",
                lines=5,
            )
            btn = gr.Button("ตรวจสอบ", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=inp, label="ลองตัวอย่างเหล่านี้")
        with gr.Column(scale=1):
            out = gr.Markdown("### ⬅️ พิมพ์ข้อความภาษาไทยทางซ้าย แล้วกด **ตรวจสอบ**")

    btn.click(analyse, inputs=inp, outputs=out)
    inp.submit(analyse, inputs=inp, outputs=out)

    gr.Markdown(
        """
        ---
        ### ⚠️ อ่านก่อนเชื่อผลลัพธ์

        - **ระบบนี้อ่านแค่ “สัญญาณผิวๆ” ไม่ได้เข้าใจภาษา** ประชดที่เขียนแนบเนียนโดยไม่มี
          `555` / สระยืด / `??` ระบบจะตอบว่า **บอกไม่ได้** ไม่ใช่ตอบว่าไม่ประชด
        - **ความแม่นยำราว F1 0.59** วัดบนชุดข้อมูล 127 ประโยค
          เทียบกับ GPT-4o ที่ได้ ~0.73 งานนี้ยากจริงสำหรับทุกวิธี
        - **สัญญาณพวกนี้มาจากภาษาโซเชียลไทย** (รีวิว Wongnai + ทวีต Wisesight)
          กับข้อความทางการหรือคนละโดเมน จะแม่นน้อยลงมาก
        - ชุดข้อมูลมีแค่ **127 ประโยค (ประชด 30)** เล็กมาก ตัวเลขทั้งหมดจึงมีความไม่แน่นอนสูง

        ### ทำไมไม่ใช้โมเดล AI?

        เราลองแล้ว fine-tune WangchanBERTa บนข้อมูลชุดนี้ แล้ว **วัดว่ามันแย่กว่า**:
        บนประโยคใหม่ที่ไม่เคยเห็น โมเดลตอบ “ประชด” เกือบทุกอย่าง (ถูก 5/10 = เดาสุ่ม)
        ส่วน cue เชิงคำได้ **8/10** · 127 ประโยคน้อยเกินกว่าโมเดลจะเรียนรู้ประชดได้จริง
        มันจึงจำชุดเทรนแทน จึงเลือกวิธีที่ **วัดแล้วว่าดีกว่าและอธิบายได้**

        ### ระบบเต็มของงานวิจัย
        เวอร์ชันเต็มจะส่งเฉพาะประโยคที่ *ไม่แน่ใจ* ไปให้ GPT ตัดสิน → จ่ายแค่ ~40% ของราคาเต็ม
        แต่ได้คุณภาพ ~95% เวอร์ชันบนเว็บนี้ตัดส่วนที่เสียเงินออกเพื่อเปิดใช้ฟรีได้อย่างปลอดภัย

        📄 [โค้ดและผลการทดลองทั้งหมดบน GitHub](https://github.com/Thanaphumi2006/Cost-Aware-Multi-Agent-LLM-Evaluation-for-Thai-Sarcasm-Detection)
        """
    )

if __name__ == "__main__":
    demo.launch()
