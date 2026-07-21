# Report outline: is multi-agent LLM worth more than a single agent? A case study in Thai sarcasm detection

> Every number in this outline comes from a real result file in this folder. No made-up numbers.
> Sources: `baseline_preds_gpt.csv`, `multiagent_preds_gpt_conservative.csv`,
> `multiagent_preds_gpt_v1aggressive.csv`, `compare_systems.py`, `PROVENANCE.md`

---

## 1. Introduction / research question

**Question:** is a multi-agent system (splitting work across several LLM stages) worth more than a single LLM call?
**Worth it = measure 4 dimensions at once:** quality (F1) · cost · time · number of LLM calls.

**Why this task:** Thai sarcasm is a task where "the surface meaning is opposite to the intent,"
which should benefit from having the LLM reason in multiple stages. If multi-agent
doesn't help even on a task like this, that's strong evidence it's not a cure-all.

**Definition used (from `labeling_rubric.md`):**
sarcasm = there must be **"pretense"** (feigned praise/thanks to jab)
- direct complaints ≠ sarcasm
- balanced reviews (real praise + real criticism) = not sarcasm (0)
This definition matters a lot, because it's the root of all the main findings (see section 5).

---

## 2. Dataset (gold set)

| | |
|---|---|
| source | Wongnai (restaurant reviews, long) + Wisesight (tweets, short) |
| size | **127 items** |
| sarcastic (1) | **30 items** (23.6%) |
| not sarcastic (0) | **97 items** (76.4%) |

**How it was built (must be stated plainly, see `PROVENANCE.md`):**
raw data → keyword filtering → **GPT-4o assisted mining (800 → 495 items)**
→ Claude ranked (130 → 37) → **human decides the final label per the rubric, blind**
(doesn't see the LLM's confidence while labeling, so the LLM doesn't steer the human).

**A limitation to declare — self-selection bias:**
the "sarcastic" side is not random but was pre-filtered by GPT-4o
→ **GPT-4o's recall in this report is inflated** (because we scooped the items it already finds)
→ but this bias **hits every system equally** (all 3 systems use GPT-4o),
   so **system-to-system comparison stays fair.** What's unreliable is the "raw numbers," not the "differences."

**Numerical evidence of this bias (from `analyze_baseline.py`):**
it shows up in the **false positive rate, not recall**:
- items from keywords: FPR = 0.205
- items mined by the LLM: FPR = **1.000 (9/9)**
- recall = 1.000 in both groups
→ interpretation: the LLM-mined pile is "items the LLM thinks are sarcastic" that the human says aren't, so it misses them all.

---

## 3. Method: 3 systems on the same harness

Every system uses the same **GPT-4o**, the same gold set, and **the same measurement code**
(`multiagent.py` `import metrics` comes straight from `baseline.py`, proving there's no double standard in measurement).

**① Single agent (baseline)** — `baseline.py`
one LLM call → JSON answer `{"label": "1"/"0"}`

**② Multi-agent: detector → verifier** — `multiagent.py`
- **detector** = exactly the same prompt as baseline (the control variable)
- **verifier** = a stage that **filters out** false positives per the rubric, **not** a stage that finds more sarcasm
- the verifier runs **only on items the detector answered 1**
  reason: the verifier can only overturn one way (1→0); running it on items already answered 0 just wastes tokens
  → this is why calls = 183, not 254 (saving 28%)

**③ WangchanBERTa (not done yet, future work)**
a self-trained small model that answers "do you really need the LLM?"

**Error handling (important for credibility):**
if the LLM answers malformed → record it as `"err"`, **do not guess `"0"`**,
because gold is 76% zeros, so guessing 0 would be right by chance often → inflated numbers.
(Actual result: **0 errors across all 127 items in every system**; no item was dropped.)

---

## 4. Results

### 4.1 Main table, all 4 dimensions

| System | F1 | precision | recall | LLM calls | cost | latency p50 |
|---|---|---|---|---|---|---|
| **baseline** (single) | 0.690 | 0.526 | **1.000** | 127 | **$0.094** | **751 ms** |
| multi-agent **v1** (aggressive verifier) | 0.714 | **0.769** | 0.667 | 180 | $0.157 | 721 ms |
| multi-agent **v2** (careful verifier) | **0.744** | 0.604 | 0.967 | 183 | $0.169 | 967 ms |

**The price multi-agent v2 pays vs. baseline:**
- cost **1.80×** (tokens 34,535 → 62,859 in)
- LLM calls **1.44×**
- latency **1.29× at p50** (751 → 967 ms) / whole set 105s → 131s

### 4.2 baseline confusion matrix (the origin of everything)

|  | predicted sarcastic | predicted not |
|---|---|---|
| **actually sarcastic** | TP = **30** | FN = **0** |
| **actually not** | FP = **27** | TN = 70 |

→ **catches every item (FN = 0) but over-flags 27**

### 4.3 Significance test (paired bootstrap, 5000 rounds)

**Why paired:** both systems are measured on **the same gold set, item by item.**
Comparing via "F1 must exceed the baseline's upper CI" treats them as separate sets → **wrongly too strict.**
The correct way is to resample the same items for both and look at the **difference** in F1 per round.

**Result (v2 vs baseline):**
- ΔF1 = **+0.054**
- 95% CI of the difference = **[−0.002, +0.113]**
- chance v2 is **not** better = **3%**
- **McNemar: v2-right/baseline-wrong = 8 | baseline-right/v2-wrong = 1**

→ **"genuinely better with ~97% confidence, but not passing the strict two-sided 95% bar."**
   The lower CI bound crosses 0 by only −0.002 — extremely close.

---

## 5. Findings (the heart of the report)

### Finding 1: the starting hypothesis was wrong — the problem is precision, not recall

We started by thinking the LLM would "fail to find sarcasm" (subtle sarcasm).
**The truth is the opposite: recall = 1.000, catching every item.**
The bottleneck is **precision 0.526** — more than half of what it calls sarcastic isn't.

**The 27 FPs are mostly "balanced reviews"** (some real praise, some real criticism),
which the rubric clearly labels 0 because there's **no pretense**.
→ the LLM confuses **"both positive and negative in one text"** with **"feigned praise to jab."**
   This is a clear, explainable semantic error, not just noise.

### Finding 2: **the verifier's design matters more than "having" a verifier** ⭐

This is the most valuable finding, and one you can test with a real ablation.
**Same verifier, same architecture, differing only in the rule when borderline:**

| | v1 "unsure → overturn" | v2 "unsure → keep" |
|---|---|---|
| total overturned | 27 items | 8 items |
| FP killed correctly | 17 | 7 |
| TP killed wrongly (damage) | **10** | **1** |
| recall | 0.667 | 0.967 |
| F1 | 0.714 | **0.744** |

**v1 trades recall for precision almost 1:1 → F1 barely moves** (just +0.024).
**v2 overturns few but accurately → F1 +0.054.**

**Lesson:** saying "add a verifier and it improves" isn't enough. A poorly designed verifier
**burns TP faster than it kills FP**, at the same token cost.
The principled reason: **subtle sarcasm reads two ways by nature**,
so "when unsure, discard" is a rule that directly conflicts with the nature of this task.
→ **the verifier's prior must match the nature of the task, not be set by intuition.**

### Finding 3: multi-agent genuinely helps, but marginally and not for free

F1 +0.054 (~97% confidence) in exchange for **cost 1.80× · calls 1.44× · latency 1.29×.**
"Worth it?" has no universal answer; it depends on how expensive FP is for the task.
- if FP is expensive (e.g. a moderator-alert system): **worth it**
- if it's just general categorization: **not worth it — paying nearly double for 5 F1 points**

### Finding 4: **don't report accuracy** (a methodological caveat)

baseline accuracy = **0.787**
guessing "not sarcastic" every time = **0.764**
→ 2 points apart, even though one system uses the LLM and the other does nothing.
**Accuracy hides all the failure** because the data is skewed 76/24.

---

## 6. Limitations (write them before being asked)

1. **Self-selection bias** — the sarcastic side of gold was pre-filtered by GPT-4o → inflated recall
   (mitigation: the bias hits every system equally → the comparison still holds / numerical evidence in section 2).
2. **n sarcastic = 30, too few to be conclusive** — wide CI, and v2 misses the 95% bar by only 0.002.
   **This is a statistical ceiling, not a technical one.** No amount of prompt tuning narrows the CI.
3. **A single labeler** — no inter-annotator agreement (κ) → can't measure "human agreement."
4. **A single model (GPT-4o)** — unknown whether this finding generalizes across models.
5. **WangchanBERTa not done** → can't yet answer "do you really need the LLM?"

---

## 7. Future work (ordered by payoff)

1. **Expand the sarcastic side of gold to 50-60 items** ← **highest payoff**
   The only way to narrow the CI enough to be conclusive (`harvest_to_review.csv` has ~470 unreviewed items left).
   *And also sample-check items the LLM calls "not sarcastic," to reduce self-selection bias.*
2. **WangchanBERTa** — answers whether a small model is enough.
3. **Fix precision at the source** — put "balanced review = 0" examples into the detector's prompt
   (cheaper than a whole verifier stage, since it adds no calls).
4. A second labeler → report κ.

---

## 8. Conclusion

For Thai sarcasm detection, **multi-agent genuinely gives better F1 than a single agent (+0.054, ~97% confidence),
but costs 1.80× more and is 1.29× slower** — a trade worth it only when false positives are expensive.

And the lesson more important than the numbers: **having a verifier doesn't guarantee improvement.**
The same verifier with a different borderline rule gives recall as different as 0.667 vs. 0.967.
**The multi-agent architecture is not the important variable; the prompt design of each stage is.**
