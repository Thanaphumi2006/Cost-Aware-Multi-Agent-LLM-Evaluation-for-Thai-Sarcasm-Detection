---
title: Thai Sarcasm Detector
emoji: 🌀
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Thai sarcasm detection with reasons, free, no API key
---

# 🌀 Thai Sarcasm Detector

Detects sarcasm (ประชด) in Thai text, shows **why**, and says **"can't tell"** when it doesn't know.

No OpenAI calls, no API key, no per-request cost, no model download, starts in seconds on free CPU.

## Why there is no neural model in here

The original plan was a fine-tuned WangchanBERTa. It was built, then **measured and rejected**:

| | on gold (its own training data) | on unseen sentences |
|---|---|---|
| mean probability when **sarcastic** | 0.801 | 0.838 |
| mean probability when **not sarcastic** | 0.238 | 0.810 |
| **separation** | **+0.563** | **+0.028** |

It separates well on data it memorised and collapses on anything new, calling almost everything
sarcastic. Tested on 10 clear unseen sentences (5 obviously sarcastic, 5 obviously genuine):

- fine-tuned WangchanBERTa, **5/10** (coin-flip; it answered "sarcastic" to everything)
- lexical cues alone, **8/10**

No threshold fixed it; the best was 7/10 at 0.85 and unstable. 127 training sentences is simply
too few to learn sarcasm, so the model memorised the set instead. This matches finding 12, where
cross-domain precision was measured falling 0.68 → 0.40.

So the hosted app uses **lexical cues only**, measurably better on new text, fully explainable,
and consistent with finding 14, where searching for `555` alone scored F1 0.590 and beat every
open 7–8B model tested.

## Three answers, not two

When no known cue is present the app answers **"can't tell"** rather than guessing "not sarcastic."
Sarcasm without surface markers is real, it accounted for 2 of the 10 test sentences, so guessing
would be lying to the user. This mirrors the research system's router: confident → answer,
unsure → admit it.

## Honest limitations

- Reads **surface cues only**, it does not understand Thai. Subtle sarcasm returns "can't tell".
- Accuracy around **F1 0.59** on a 127-sentence set. GPT-4o reaches only ~0.73; this task is hard.
- Cues come from Thai social media (Wongnai reviews, Wisesight tweets). Formal or out-of-domain
  text will do noticeably worse.
- **127 sentences, 30 sarcastic**, a small sample, so all figures carry real uncertainty.

## The full research system

The complete system escalates only *uncertain* sentences to GPT, reaching **95% of full-price
quality for ~40% of the cost**. This hosted version drops the paid path so it can run publicly
for free with no key to abuse.

📄 [Full code, experiments and results on GitHub](https://github.com/Thanaphumi2006/Cost-Aware-Multi-Agent-LLM-Evaluation-for-Thai-Sarcasm-Detection)
