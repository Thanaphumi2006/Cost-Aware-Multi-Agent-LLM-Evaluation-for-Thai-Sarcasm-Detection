# Sarcasm/irony labeling rubric (the sharp version), for Thai

Goal: make the human (and a second reviewer) agree as much as possible.
The real problem: both humans and LLMs confuse the "boundary" between sarcasm vs. direct criticism vs. balanced review.
This rubric pins that boundary with real examples from our data.

---

## Definition (the main decider)

**Sarcasm (1) = the author "feigns praise/thanks" while genuinely meaning the opposite, to jab.**

The heart of it = there must be **pretense** (saying positive, meaning negative), deliberately so the reader catches it.

> If there is no pretense = not sarcasm (0), always, even if the content is negative.

---

## Decision tree (follow in order)

1. Does the author **state something negative directly** (complains/criticizes, no fake praise)? → **0** (direct criticism)
2. Does the author **genuinely praise** with no hidden meaning? → **0** (sincere praise)
3. Are there **both real pros and real cons** (a balanced review)? → **0**
4. Is it just **recounting events / quoting someone else**? → **0** (unless the author adds a clear jabbing tone)
5. **Saying positive but really meaning negative, to jab?** → **1** (sarcasm)
6. Genuinely undecidable without outside context → **X**

---

## 4 groups to separate cleanly (with real examples)

### ✅ Sarcasm = 1
Feigned praise/thanks that actually conveys something bad; usually has a "self-contradicting" point.

- *"...ไม่มีส่วนผสมให้ระคายเคืองผิว **ให้ผิวแอบงอนโมโหจนขึ้นสิว**"*
  → praises "no irritation" but says the skin broke out = self-contradiction = jab → **1**
- *"ลบแอปโลด แถมไม่ต้องเจอกันอีกเลยนะ แอปนี้ 555"*
  → sarcastic/annoyed tone ("glad I never have to see it again") → **1**

**Check:** praise/thanks + a self-contradicting point + a jabbing tone → all 3 present = 1

### ❌ Direct criticism = 0
Complains/insults directly, no feigned praise, even if very negative.

- *"รสชาติหวานไปนิด...กระดูกอ่อนไม่ใช่กระดูกอ่อนจริงๆ คือแข็งมากกัดไม่เข้า"*
  → direct criticism, no fake praise → **0** (do not label 1)

**Trap:** "negative = sarcasm" is wrong! Direct negative = 0.

### ❌ Balanced review = 0
Some genuine praise + some genuine criticism, side by side, as-is.

- *"MG รถมันคุ้มจริงๆ แต่...รถมี defect เยอะ...ช่างไม่ค่อยเก่ง เบิกอะไหล่นาน"*
  → "คุ้มจริงๆ" is real praise + real criticism = balanced review → **0** (not pretense)

### ❌ Sincere praise = 0
Praise that means praise, even if worded over the top.

- *"บรรยากาศ...ลมโกรกดีมาก พัดลมไม่ต้องเปิดเลย"*
  → genuinely praising the cool breeze (don't assume it secretly means "it's hot") → **0**

### ❓ Undecidable = X
- Need outside context (who's speaking / the situation) to know whether it's a jab.
- Sentence too short/ambiguous to guess the intent.

---

## Anti-mistake rules (read before every keypress)

1. **Direct negative ≠ sarcasm** — there must be "feigned praise" to count as 1.
2. **Don't read hidden meaning beyond the text** — if you have to imagine "maybe it secretly means..." then it's unclear → lean 0 or X.
3. **Over the top ≠ sarcasm** — exaggerated praise may be genuine excitement.
4. **Recounting events / quoting others** is not itself sarcasm, unless the author adds a jabbing tone.
5. Torn between 1 and 0 → if the "pretense" isn't clear, give 0.

---

## Gold-quality notes
- Should have at least ~30-40 sarcastic (1) items, or F1 swings.
- If possible, have a second person re-review a portion and measure inter-annotator agreement.
