# Presentation slides: 12 slides (~12 minutes)

> Every number comes from a real result file · use `project_overview.svg` on slide 4 directly
> Full detail: `RESULTS.md` (findings 1-9) · `REPORT.md`

---

## Slide 1: the question

# Is multi-agent worth more than a single agent?

**Case study:** Thai sarcasm/irony detection

Measure 4 dimensions at once, not just F1
> **quality · cost · time · number of LLM calls**

*(say: people usually show only F1 and claim multi-agent is better, but never say how much it costs,
and never compare against a single agent squeezed to its limit. That's the twist you'll see.)*

---

## Slide 2: why sarcasm

Sarcasm = **the genuine intent is opposite to the surface meaning**

**The key rule of this task: there must be "pretense"**
- ✅ "ขอบคุณมากนะคะที่ให้รอ 2 ชั่วโมง" → sarcasm (feigned thanks)
- ❌ "ร้านนี้ห่วยมาก รอนาน" → **direct complaint = not sarcasm**
- ❌ "อาหารอร่อย แต่ที่จอดรถแย่" → **balanced review = not sarcasm**

*(the bottom two are the heart of it; you'll see the LLM break here)*

---

## Slide 3: dataset + a limitation I declare myself

**gold = 127 items** (30 sarcastic / 97 not)
Wongnai (long reviews) + Wisesight (short tweets) · human decides the final label, blind

### ⚠️ Self-selection bias — I say it before being asked
The 30 sarcastic items were **mined with GPT-4o's help** (filtered 800 → 495)
→ **GPT-4o's recall on this task is inflated**
→ **but the bias hits every system equally → the *comparison* stays fair**

**Evidence:** the LLM-mined items have FPR = **1.000 (9/9)** vs keyword items 0.205

### ⚠️ n(sarcastic) = 30 → wide CI (±0.10). Remember this number; it's the hero of the story.

---

## Slide 4: the systems tested · same harness

**[paste `project_overview.svg` here]**

- **① baseline** single LLM call
- **② multi-agent (v1/v2)** detector → **verifier (an FP-filter stage, not a find-more stage)**
- **④ debate / ⑤ hybrid** 3-4 agents debate then decide
- **⑥ WangchanBERTa** a free small model (5-fold CV)

**Tight controls:**
- the detector uses **exactly** the baseline prompt
- `multiagent.py` **imports the measurement functions straight from `baseline.py`** → no double standard possible even if wanted
- malformed LLM answer → recorded as `err`, **not guessed 0**. Actual result: **0 errors**

---

## Slide 5: results across 4 dimensions (before the twist)

| System | F1 | precision | recall | cost | agents |
|---|---|---|---|---|---|
| baseline | 0.690 | 0.526 | **1.000** | **$0.094** | 1 |
| multi-agent v1 | 0.714 | **0.769** | 0.667 | $0.157 | 2 |
| **multi-agent v2** | **0.744** | 0.604 | 0.967 | $0.169 | 2 |
| debate | 0.694 | 0.595 | 0.833 | $0.695 | 3 |
| hybrid | 0.700 | 0.560 | 0.933 | $0.407 | 4 |

**At a glance: v2 wins F1 · more agents (debate/hybrid) don't help · but v2 pays 1.80×**
*(this is the "old story"; slide 8 flips it all)*

---

## Slide 6: Finding 1 — my hypothesis was **wrong**

### I thought the LLM would "fail to find sarcasm"

**Reality: recall = 1.000, catches every item, misses none**

|  | pred sarcastic | pred not |
|---|---|---|
| actually sarcastic | TP **30** | FN **0** |
| actually not | FP **27** | TN 70 |

### The real bottleneck is **precision = 0.526**
The 27 FPs are mostly **"balanced reviews"** (real praise + real criticism)
→ the LLM can't tell **"has both positive and negative"** apart from **"feigned praise to jab"**

---

## Slide 7: Finding 2 ⭐ **the verifier's design matters more than having a verifier**

**Same verifier, same architecture, differing only in the borderline rule**

| | v1 "unsure → **discard**" | v2 "unsure → **keep**" |
|---|---|---|
| overturned | 27 items | 8 items |
| FP killed correctly | 17 | 7 |
| **TP killed wrongly** | **10** ❌ | **1** ✅ |
| recall | 0.667 | **0.967** |
| F1 | 0.714 | **0.744** |

**Because subtle sarcasm reads two ways by nature → "when unsure, discard" directly conflicts with this task**

---

## Slide 8: the twist 🔴 the baseline was handicapped all along

### GPT doesn't answer "0/1" — it answers a **token with a logprob**

We threw away the logprob on every item and compared raw 0/1 = **making multi-agent fight a handicapped opponent**

**Read P("sarcastic") from the logprob and tune the threshold** (leave-fold-out, no leak):

| | F1 | cost | added calls |
|---|---|---|---|
| baseline @argmax (original) | 0.690 | $0.094 | — |
| **baseline + threshold** | **0.725** | **$0.094** | **0 (free!)** |

**+0.035 F1 for free, not one extra call** (the score comes from a call already paid for)

---

## Slide 9: Finding 3 ⭐⭐ re-compare fairly → **no one wins**

**Compare every system to baseline+threshold (0.725) instead of the raw baseline · paired bootstrap, 5,000 rounds**

| System | ΔF1 | 95% CI | P(not better) | cost |
|---|---|---|---|---|
| v2 (best multi-agent) | +0.019 | **[−0.073, +0.113]** | 36% | 1.80× |
| hybrid | −0.025 | [−0.117, +0.071] | 70% | 4.3× |
| debate | −0.030 | [−0.158, +0.096] | 69% | 7.4× |

> **Every CI crosses 0. No multi-agent system separates from the tuned single agent.**
> v2's old advantage "+0.054 vs baseline" → shrinks to **+0.019** → **half of it was the baseline being handicapped**

---

## Slide 10: Finding 4 ⭐⭐⭐ the axis that actually moves cost is the **model**, not the architecture

**Single agent + threshold across 5 models (25× price range):**

| model | F1 | $/run |
|---|---|---|
| gpt-4.1-nano | 0.706 | $0.004 |
| **gpt-4.1-mini** ⭐ | **0.727** | $0.015 |
| gpt-4o | 0.725 | $0.094 |

**F1 stays within a 0.05 band across the 25× price range (smaller than the CI) · the top-F1 model is 6× cheaper than gpt-4o**

### The knockout (paired):
**gpt-4.1-mini alone (0.727, $0.015)  ≈  gpt-4o v2 multi-agent (0.744, $0.169)**
Δ +0.016 · CI [−0.094,+0.135] crosses 0 · **v2 is 11× more expensive but indistinguishable**

---

## Slide 11: conclusion (revised after the fair comparison)

### The old story I almost believed:
~~"multi-agent is better +0.054, worth it if FP is expensive"~~

### The real story after squeezing the baseline dry:
1. ⭐ **before asking "add an agent?" → first ask "have I squeezed out everything the single agent already gives?"**
   the baseline throws away the logprob it already paid for, on every item
2. **On this task, no evidence any multi-agent is worth the extra money** (every CI crosses 0)
3. **The real "cost-aware" move = pick a cheap model + read the logprob**, not pick the architecture

### Caveat (multi-agent is not worthless):
error profiles differ — v2 recall 0.967 / threshold 0.833 → for "must not miss" tasks, v2 is still worth it.
"Indistinguishable" ≠ "definitely equal." **n=30 is the ceiling.**

---

## Slide 12: future work

### Ordered by payoff
1. ⭐ **Expand gold to 50-60 sarcastic** ← the only way to narrow every CI enough to be conclusive
   Use the logprob to pick "borderline items" (P 0.2-0.8) to label — items that actually separate systems, not obvious ones.
2. **Fix precision at the source** — put "balanced review = 0" examples into the detector prompt (free, no added calls)
3. **cascade (WCB screening) tried, lost** (finding 6): WCB "can't rank," so it can't screen

---

## Likely questions (prepared)

**Q: If multi-agent doesn't win, why do it?**
> Because "did it and found it not worth it" is a valuable result, and I got a stronger lesson:
> you must tune the baseline to its limit first, or the comparison is biased. Most work in this area skips this step.

**Q: 127 items is too few — can you trust it?**
> Yes, which is why I attach a CI to every number and say myself that n=30 is the ceiling. All my conclusions are
> "indistinguishable," not "definitely equal." The top future-work item is expanding gold.

**Q: You mined data with GPT-4o and then measured on it — isn't that cheating?**
> Cheating for the **raw numbers** (especially recall), but not for the **system comparison**,
> because every system eats the same bias · evidence: the LLM-mined pile has FPR = 1.000.

**Q: Isn't threshold tuning overfitting?**
> No — the threshold is chosen leave-fold-out (fold k uses values from the other 4 folds) for every reported number;
> no item helps set the threshold that judges it.

**Q: How much can you trust the model prices?**
> Token counts are measured exactly from the API · the $/1M figures should be checked against current pricing (in frontier.py).
