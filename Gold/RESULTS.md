# Results: single agent vs. multi-agent (Thai sarcasm)

Measured on the 127-item gold set (30 sarcastic / 97 not). Every system uses GPT-4o and the same harness.
For the origin of the gold set and its self-selection bias, see `PROVENANCE.md`.

## Summary of the systems

| System | F1 | precision | recall | LLM calls | cost | latency p50 |
|---|---|---|---|---|---|---|
| **baseline** (single) | 0.690 | 0.526 | **1.000** | 127 | $0.094 | 751 ms |
| **baseline + threshold** (single, threshold from logprob) ⭐ | 0.725 | 0.641 | 0.833 | 127 | **$0.094** | 751 ms |
| multi-agent **v1** (shallow verifier) | 0.714 | **0.769** | 0.667 | 180 | $0.157 | 721 ms |
| multi-agent **v2** (recall-preserving verifier) | **0.744** | 0.604 | 0.967 | 183 | $0.169 | 967 ms |
| **debate** (prosecutor + defender + judge) | 0.694 | 0.595 | 0.833 | 381 | $0.695 | 4557 ms |
| **cascade** (WCB screener → GPT verifier) ✗ | 0.628 ± 0.006 | 0.500 | 0.844 | 119 | $0.124 | — |
| **WangchanBERTa** (5-fold CV × 3 seeds) | 0.620 ± 0.005 | 0.553 | 0.700 | **0** | **$0.00** | **26 ms** |

WangchanBERTa is measured with 5-fold stratified CV (every item is predicted by a model that never saw it)
→ full 127-item predictions, so it pairs directly against the LLMs. Trained on CPU in 44 min. Runs offline.

## Main findings (answering the thesis "is multi-agent worth it?")

**1. The baseline has perfect recall (1.000) but low precision (0.526)**
GPT-4o catches every sarcastic item, but misreads 27 "balanced reviews" (genuine praise + genuine criticism) as
sarcastic. The problem is not "failing to find sarcasm" but "casting too wide a net."

**2. The verifier is a false-positive filter; prompt wording changes the result a lot**
- v1 (overturn when unsure): overturned 27 items, 17 right and 10 wrong → recall fell to 0.667, F1 barely moved.
- v2 (overturn only when clearly not sarcastic): overturned 8 items, lost only 1 real positive → best F1, 0.744.

**3. v2 beats the baseline "almost significantly," but at nearly double the cost**
paired bootstrap (resample the same gold set 5000 times):
- ΔF1 = +0.054, 95% CI [-0.002, +0.113]
- The CI barely crosses 0 (lower bound -0.002) → chance v2 is not better = only 3%
- McNemar: v2-right-baseline-wrong 8 / baseline-right-v2-wrong 1 (clearly favors v2)
=> "probably genuinely better (~97% confidence)" but not passing the strict 95% bar.
   At n=30 positives, statistical power is too low to be conclusive.

**4. More agents != better; debate loses to the simpler pipeline while costing 4× more**
Debate architecture (prosecutor + defender → judge, 3 calls/item, can re-decide in either direction):
- F1 **0.694** — barely different from the single-agent baseline (+0.005, chance not better = **46%** = coin flip)
- Loses to pipeline v2 (0.744): Δ **-0.049**, chance debate is not better than v2 = **83%**
- **But costs 4.1× more than v2 ($0.695 vs $0.169) and is 4.7× slower (4557 vs 967 ms)**

Why debate loses — **it returns the freedom to re-decide both ways and loses recall for free**:
recall drops from 1.000 (baseline) to **0.833** — the defender successfully argues away 5 real positives (FN=5),
whereas v2's verifier is **structurally constrained** to overturn one way only (1→0) with the rule "when unsure, keep."
=> v2 therefore preserves almost all of the recall 1.000 the detector got for free (0.967), then buys precision slowly.

**Architectural lesson (more important than the numbers):**
**"the agent's constraints" matter more than "the number of agents."**
A verifier that can only reject (heavily constrained) beats a 3-agent panel that can decide freely in every direction,
because it guards the good thing it already has (full recall) instead of re-gambling every item.

**5. The free small model "can't be dismissed yet" — and this is the most unexpected result**
WangchanBERTa (F1 0.620) looks like it loses to the baseline (0.690) at a glance, but under paired testing:
- ΔF1 = -0.072, 95% CI **[-0.206, +0.053]** → **crosses 0**
- McNemar: WCB-right-baseline-wrong **15** | baseline-right-WCB-wrong **14** → **essentially a tie**
=> at n=30, **we cannot conclude GPT-4o beats a free model trained on just 127 items.**

But **don't over-read this**: the point estimate still says WCB is worse (chance it is worse = 87%).
"Can't conclude it's worse" != "equally good" — it just means the gold set is too small to decide.
And the **error profiles differ clearly**: WCB lets 9 sarcastic items slip (recall 0.700) while the LLM misses none
→ if the task is "screening where nothing may be missed," the LLM still wins clearly, even if the F1 gap isn't significant.

WCB's measurable advantages: **$0.00 · 0 API calls · 26 ms (29× faster than the LLM) · runs offline.**

**6. Using WangchanBERTa (free) as the screener instead of GPT "does not work," and the reason it fails is itself a finding**

The idea (exactly per lesson 4): the winning architecture is *a high-recall screener → a reject-only verifier*.
But v2 uses **GPT as the screener = paying on every item (127 calls)**, a "price floor" that cannot be cut.
Swap the first stage to WangchanBERTa (free · offline · 26 ms) and that whole price floor should disappear.

**WCB's F1 of 0.620 is not the reason it can't be a screener.** A screener doesn't need to be accurate; it just needs
to "not let sarcasm slip" (high recall), and the verifier trims the over-flagging later → so measure **recall as the
threshold is lowered**, not F1 at 0.5.

Method: produce out-of-fold probabilities (`wcb_oof_probs.py`, 5-fold × 3 seeds, same protocol as finding 5),
then choose the threshold **leave-fold-out** (items in fold k use a threshold chosen from the other 4 folds only)
→ no item helps set the threshold that judges it (prevents threshold leak).

| target recall | flagged (of 127) | actual recall | first-stage precision | GPT calls | cost | vs v2 |
|---|---|---|---|---|---|---|
| 0.85 | 81 | 0.822 | 0.31 | 81 | $0.085 | 0.50× |
| 0.90 | 92 | 0.889 | 0.29 | 92 | $0.096 | 0.57× |
| 0.95 | 105 | 0.933 | 0.27 | 105 | $0.110 | 0.65× |
| 1.00 | **119** | 0.967 | **0.24** | 119 | $0.124 | 0.74× |

**The number that kills this idea: to catch 29/30 sarcastic items, the screener must flag 119 of 127 items = 94% of the set.**

- Send every item straight to the verifier without screening = 127 calls, recall 1.000
- Let WCB screen first = 119 calls, recall 0.967
- → **this free screener saves 8 calls and, in exchange, lets 1 real positive slip.**
  It "filters" nothing; it just adds a step.

Compare to GPT doing the same job: recall 1.000 by flagging only **57 items** (precision 0.526).
WCB has to flag **2× more** to get a **lower** recall at half the precision (0.24), and v2's verifier is designed to be
conservative ("unsure → keep"), overturning only 8 of 56 items it sees (14%) → hand it 89 false positives instead of 27
and **it cannot recover the precision.**

**Lesson: "free" does not mean "put it in front of the pipeline and save."**
A usable screener must **be able to rank** — to tell which items are "clearly not sarcastic" so it can drop them for free.
WCB trained on 127 items **cannot**: its scores carry almost no information in the tail that the decision needs.
→ confirms finding 5 from another angle: the free small model "can't be dismissed yet" not just as a standalone system
but also as a **component**.

**Confirmed by an actual run (not just an estimate):** ran the real verifier on the items WCB flagged (target recall 1.00)
→ **F1 0.628 ± 0.006** (3 seeds) · 119 calls · $0.124
- **Worse than the single-agent baseline (0.690) and *more expensive* ($0.124 vs $0.094)** — loses on every axis.
- paired bootstrap vs baseline: Δ **-0.062**, 95% CI [-0.152, +0.025] · McNemar: base-right-cascade-wrong 14 / cascade-right-base-wrong 9
- (A dry-run earlier guessed F1 would drop to ~0.42 — too pessimistic; the verifier overturns more than expected.
  But **the conclusion doesn't change**: pay more than baseline to get lower F1.)

*(Reproducibility note: this run's WCB @0.5 gave F1 0.606 / 0.576 / 0.588 → mean **0.590** (SD 0.012),
lower than reported in finding 5 (0.620 ± 0.005) and more variable, likely a different torch/transformers version.
So the WCB numbers in the table may be "optimistic," and the ±0.005 SD may be understated.)*

**7. Single agent + threshold ≈ multi-agent v2 at half the price (the finding that shakes the thesis most)**

Every system above answers a hard 0/1 "at a single operating point," but GPT doesn't really answer 0/1 — it answers a
*token* with a logprob. Extract P("1") from the label token's logprob (`gpt_threshold.py`) **without touching the prompt
and without a single extra call** → this score comes "for free" from the calls already paid for in the baseline.

GPT **can rank** (completely unlike WCB in finding 6): moving the threshold genuinely buys precision:

| tau | precision | recall | F1 |
|---|---|---|---|
| 0.50 (argmax = original baseline) | 0.526 | **1.000** | 0.690 |
| 0.75 | 0.558 | 0.967 | 0.707 |
| 0.95 | 0.596 | 0.933 | 0.727 |
| 0.99 | 0.667 | 0.800 | 0.727 |

Choosing the threshold **leave-fold-out** (tau ≈ 0.984 every fold) → **F1 0.725 · prec 0.641 · recall 0.833 · 127 calls · $0.094**

**Compared to v2 (2 agents), paired, this is the pure money question of "is the 2nd agent worth it":**
- v2 0.744 vs threshold 0.725 → Δ **+0.019**, 95% CI **[-0.076, +0.121]** → **crosses 0, widely**
- Chance v2 is **not** better = **34%**
- McNemar: v2-right-threshold-wrong **7** | threshold-right-v2-wrong **8** → **a dead tie**
- **But v2 costs 1.80× more (+$0.075, +56 calls)**

=> **the 2nd agent buys an F1 gain indistinguishable from noise, at nearly double the cost.**
   What v2 actually does (buy precision while losing the least recall) **a single threshold knob does almost as well, for free.**

**Caveat (don't rush to switch to threshold):** the error profiles differ clearly.
- v2: recall **0.967** (misses 1 positive) — keeps the good thing from the detector.
- threshold: recall **0.833** (misses 5 positives) — buys precision by *giving up recall*.
→ if the task is "screening where nothing may be missed," **v2 still wins** and the $0.075 is worth it.
→ if judged on pure F1, **the free knob wins decisively on price.**

**Lesson (continuing from 4):** before asking *"add an agent?"*, first ask
***"have I squeezed out everything the single agent already gives me?"*** Here the answer is **no**.
The baseline throws the logprob (information already paid for) in the trash on every item, and we bought a 2nd agent to make up for it.
**Comparing multi-agent against a not-fully-tuned baseline = a comparison that credits multi-agent too much.**

**8. The fair comparison: compare every system to the tuned baseline → no system wins significantly**

Findings 1-7 compare everything to the baseline @argmax (0.690), which "throws away the logprob" already paid for
= making multi-agent fight a handicapped opponent. The correct comparison point is **baseline+threshold (0.725, same cost)**.
Re-ran everything (`compare_fair.py`, paired bootstrap, no API calls, reuses existing preds):

| System | F1 | Δ vs baseline+threshold | 95% CI | P(not better) | McNemar | cost |
|---|---|---|---|---|---|---|
| conservative (v2) | 0.744 | +0.019 | [-0.073, +0.113] | 36% | 7-8 | $0.169 |
| v1aggressive | 0.714 | -0.010 | [-0.159, +0.135] | 57% | 11-8 | $0.157 |
| hybrid | 0.700 | -0.025 | [-0.117, +0.071] | 70% | 6-11 | $0.407 |
| debate | 0.694 | -0.030 | [-0.158, +0.096] | 69% | 9-12 | $0.695 |
| cascade | 0.628 | -0.097 | [-0.203, +0.012] | 96% | 7-20 | $0.124 |
| wangchanberta | 0.618 | -0.107 | [-0.235, +0.018] | 95% | 7-14 | $0.00 |

**Every CI crosses 0. No system separates from baseline+threshold significantly.**
The best multi-agent system (v2) still has a 36% chance of *not* being better than a single threshold knob, at 1.80× the cost.
Debate/hybrid, which cost 4-7× more, actually sit *below* the free knob.

=> **The multi-agent v2 advantage once reported as "+0.054 vs baseline" shrinks to +0.019 vs the tuned baseline
and vanishes into the noise.** Half of the original +0.054 was "the baseline being handicapped," not "multi-agent being good."

**9. Cost-quality frontier: changing the *model* moves cost 25× with F1 barely moving, and a cheap model gets the top F1**

Findings 1-8 are all on gpt-4o alone. A project called "Cost-Aware" should measure the *model* axis too, not just architecture.
Take the best + cheapest architecture (single agent + threshold, finding 7) and run it on 5 models spanning 25× in price
(`frontier.py`, threshold chosen leave-fold-out as before):

| model | F1 | prec | recall | $/run (127 items) | $/1M in |
|---|---|---|---|---|---|
| gpt-4.1-nano | 0.706 | 0.632 | 0.800 | $0.0038 | $0.10 |
| gpt-4o-mini | 0.676 | 0.568 | 0.833 | $0.0056 | $0.15 |
| **gpt-4.1-mini** ⭐ | **0.727** | 0.667 | 0.800 | $0.0150 | $0.40 |
| gpt-4.1 | 0.716 | 0.649 | 0.800 | $0.0752 | $2.00 |
| gpt-4o | 0.725 | 0.641 | 0.833 | $0.0940 | $2.50 |

- **F1 stays in 0.68-0.73 across the entire 25× price range.** This 0.05 band is smaller than the CI at n=30 (±0.10) → almost all noise.
- **The model with the top F1 is gpt-4.1-mini ($0.015), not gpt-4o ($0.094)** — the model the whole project was built on isn't even the best.
- gpt-4o-mini is the outlier (lowest F1, 0.676) → F1 does not track price linearly; pick the model by actually testing, not by guessing from price.

**The knockout, cheap single agent vs. flagship multi-agent (paired, n=127):**

| | F1 | cost | agents / calls |
|---|---|---|---|
| gpt-4.1-mini + threshold | 0.727 | **$0.015** | 1 agent / 127 |
| gpt-4o v2 (flagship multi-agent) | 0.744 | $0.169 | 2 agents / 183 |

Δ +0.016, 95% CI **[-0.094, +0.135]** (crosses 0) · P(v2 not better) = **39%** · **v2 is 11× more expensive.**

=> The project's best multi-agent system is **indistinguishable from a single agent on a cheap model, but 11× more expensive.**
   The real "cost-aware" move here is not "pick the architecture" but **"pick a cheap model + read its logprob."**

*(Prices are estimates in `frontier.py`/PRICE; check against current pricing. Token counts measured exactly from the API:
34,535 in / 762 out, identical for every model.)*

**10. Fixing precision at the source (few-shot in the detector) cuts FP but F1 doesn't move significantly**

Finding 1 shows the bottleneck is precision: 27 FPs, mostly "balanced reviews" (genuine praise + criticism) the LLM reads as sarcasm.
Every earlier system cleans up these FPs *after* the detector fires; this finding tries fixing **the detector itself**:
add an explicit "feigned praise" rule + 4 **synthetic** few-shot examples (not gold, to avoid leak) teaching the "balanced review = 0" boundary.
Run on gpt-4o + threshold leave-fold-out (differs from finding 7 only in the prompt):

| | F1 | precision | recall | FP | FN |
|---|---|---|---|---|---|
| baseline @argmax (original prompt) | 0.690 | 0.526 | 1.000 | **27** | 0 |
| + threshold (original prompt) | 0.725 | 0.641 | 0.833 | 19 | 5 |
| **+ threshold + balanced few-shot** | 0.738 | **0.686** | 0.800 | **11** | 6 |

- **The mechanism works: FP falls 27 → 11 · precision 0.526 → 0.686**, few-shot removes "balanced reviews" as intended.
- **But F1 moves only 0.725 → 0.738 = not significant** (paired vs original prompt: +0.014, 95% CI [-0.075, +0.100], P(not better) 37%)
  because cutting FP trades against lower recall (0.833 → 0.800, FN 5 → 6) — a wash.
- **vs v2 multi-agent (paired): 0.738 vs 0.744 → Δ +0.005, P(v2 not better) 46% = coin flip.**
  → **single detector + good prompt = 2-agent multi-agent** (another line of evidence that adding an agent isn't needed).

**Cost caveat (correcting my own misconception):** few-shot makes the prompt longer → input tokens ~2× → **$0.19, not $0.094**.
"Free" applies only to the *number of calls* (still 127), not the *dollars*; at this price it's about the same as v2 ($0.169), not cheaper.
(The truly cheap option is still gpt-4.1-mini with the original prompt, $0.015, from finding 9.)

**Summary of this finding:** everything — baseline, threshold, few-shot, v2, debate — sits in the F1 0.69-0.74 band, indistinguishable at n=30.
Even fixing the bottleneck (precision) directly still hits the same band → reinforcing that **the ceiling is the data, not the method.**

**11. Expand gold to 45 positives (from 30) and re-compare: the "indistinguishable" conclusion holds (and strengthens)**

Findings 6-10 all hit the same ceiling: n(positives)=30 → CIs too wide to decide. So try "raising the ceiling" by labeling more.
A human labeled the top 37 items of `to_label_next.csv` (chosen by logprob P>0.2) → 15 positive + 15 negative + 7 unsure
→ **gold expands 127→157 items, positives 30→45 (+50%)**, stored as `gold_expanded.csv` (canonical gold.csv untouched).

Compare the main pair (baseline+threshold vs v2 multi-agent) on gpt-4o, threshold leave-fold-out, on the 157-item set:

| | n=127 (30 pos) | n=157 (45 pos) |
|---|---|---|
| baseline+threshold F1 | 0.725 | 0.745 |
| v2 multi-agent F1 | 0.744 | 0.724 |
| Δ (v2−thr) | +0.019 | **−0.021** |
| 95% CI | [−0.073, +0.113] | [−0.112, +0.072] |
| P(v2 not better) | 36% | **68%** |
| McNemar (v2 : thr) | 7 : 8 | 11 : 17 |

- **Still crosses 0 — adding 50% more positives still can't separate multi-agent from single agent.** Finding 8's conclusion holds.
- The point estimate **flips toward threshold** (−0.021) but **don't read it as "threshold wins,"** because...

**Bias to declare (important):** the 30 new items were **not random** — chosen by the model's logprob score (P>0.2)
→ biased toward the logprob-threshold system directly · so absolute F1 on 157 **can't be compared to the original 127.**
Read only the direction: "multi-agent still hasn't emerged as better, even with more data."

**Statistical-power lesson:** the CI barely narrowed (0.186 → 0.184) despite +50% positives,
because the chosen items are "hard/borderline" (P 0.2-0.8 + model got wrong) — these are noisy and don't add power as hoped.
→ **To actually narrow the CI, you must sample positives randomly (not just the hard ones), and far more of them.**

**12. Real cross-domain test (Pantip): precision collapses because genuine praise is read as sarcasm**

Findings 1-11 all measure on a single domain (Wongnai reviews + Wisesight tweets), so test whether it transfers across domains.
A human hand-labeled 55 Pantip comments (human-decided, not model) from political + food/review threads.
Measured with gpt-4.1-mini (balanced) via `validate_link.py` + `eval_domain.py`:

| | original gold (reviews/tweets) | Pantip (n=55) |
|---|---|---|
| precision | 0.68 | **0.40** |
| recall | 0.83 | 0.86 |
| F1 | 0.75 | **0.545** |

TP 6 · FP 9 · FN 1 · TN 39 · F1 95% CI [0.25, 0.77]

**The core: precision falls 0.68 → 0.40** because the 9 FPs are "praise/normal text the model reads as sarcasm."
Recall stays high (0.86, caught 6/7 real positives), meaning the model "cries wolf" out-of-domain: it doesn't miss sarcasm
but flags a lot of genuine praise as sarcasm, confirming with real numbers what the YouTube trials showed (over-flagging praise).

**A "small samples deceive" lesson:** the first pilot of 14 items (all political threads) gave F1 0.80, looked like it passed.
But there was little genuine praise; adding 41 praise items from food threads dropped precision to 0.40 → small samples can mislead.

Robustness: the precision finding is solid (FP 9 of ~48 negatives is a clear pattern, not noise).
The recall/F1 side is still underpowered (only 7 positives, wide CI), so read F1 as direction, but "over-flags praise" is conclusive.

**Conclusion: the model does not transfer well across domains.** Use only on content similar to gold (reviews/short posts).
Other domains require re-labeling + re-tuning (the web app's few-shot correction button helps reduce these FPs). This is the biggest risk at deployment.

**13. open-model frontier (revised): open models genuinely lose, but the gap is ~40% smaller than first reported, and "Thai-specialized" models have no advantage**

Extending the cost axis directly: *"what F1 can $0 of API buy?"* Run open models locally (RTX 3060 Ti, 4-bit)
on the same 127-item gold via lm-evaluation-harness (task `thai_sarcasm`, same prompt as DETECT_SYS).

**What was wrong in the first version of this finding (2 fixes):**

1. **Wrong GPT baseline.** The first version wrote GPT bot = F1 0.696, but finding 9 (the `frontier.py` table)
   lists gpt-4.1-mini = **0.727** (prec 0.667 / recall 0.800). Re-running from `frontier_probs_gpt-4.1-mini.csv`
   gives 0.727/0.667/0.800, matching finding 9 exactly, and no file in the project produces 0.696 → 0.696 was a wrong number.
2. **Unfair protocol.** The first version compared *threshold-tuned GPT* against *raw-argmax open models*,
   the same mistake finding 8 caught ("the baseline was handicapped").
   → this time both sides use the same leave-fold-out (StratifiedKFold 5, seed 42).

**A third factor: prompt format.** All open models are instruct/chat models, but the original harness measured
loglikelihood on a bare prompt without the model's chat template → severe calibration loss
(Typhoon flagged 114/127, OpenThaiGPT 126/127 when the truth is 30). With `--apply_chat_template`, AUC rises clearly.

| System | prompt | F1 (leave-fold-out) | prec | recall | AUC | cost |
|---|---|---|---|---|---|---|
| **GPT bot (gpt-4.1-mini + threshold)** | — | **0.727** | 0.667 | 0.800 | 0.890 | ~$0.015/run |
| Qwen2.5-7B (general) | chat | 0.576 | 0.528 | 0.633 | 0.774 | **$0** |
| SeaLLM-7B-v2.5 (Thai/SEA) | chat | 0.576 | 0.528 | 0.633 | 0.812 | **$0** |
| Typhoon-8B (Thai) | chat | 0.523 | 0.397 | 0.767 | 0.797 | **$0** |
| OpenThaiGPT-7B (Thai) | chat | 0.453 | 0.378 | 0.567 | 0.758 | **$0** |
| WangchanBERTa (fine-tuned) | — | ~0.62 | 0.55 | 0.70 | | $0 (needs training) |

**paired bootstrap (5000, same 127 items, same protocol both sides):**

| open model | ΔF1 (GPT wins) | 95% CI | P(GPT better) | McNemar |
|---|---|---|---|---|
| Qwen2.5-7B | +0.152 | [+0.040, +0.275] | 99.6% | 13/3 |
| SeaLLM-7B | +0.152 | [+0.034, +0.282] | 99.4% | 14/4 |
| Typhoon-8B | +0.205 | [+0.064, +0.341] | 99.8% | 30/6 |
| OpenThaiGPT-7B | +0.274 | [+0.118, +0.433] | 100.0% | 30/7 |

**The original conclusion stands, but the effect size changes:** GPT still beats every open model significantly (P ≥ 99.4%),
but the real gap is **+0.152, not +0.251**. About 40% of the previously-reported gap came from
the wrong baseline + unfair protocol + prompt format mismatch, not from model capability.

**A question the old finding left open, now answered:** the first version guessed *Thai-specialized* models
"had the best shot at catching GPT." **Not true**: SeaLLM (0.576) ties general-purpose Qwen (0.576) exactly,
while Typhoon (0.523) and OpenThaiGPT (0.453) lose to Qwen.
→ **Thai-specific training doesn't help this task**: the problem is "reading sarcastic intent," not "reading Thai."
*(Evidence scope: 4 models, 7-8B, 4-bit, zero-shot only. Not a claim that Thai LLMs are weak in general.)*

**An interesting side effect: chat template fixes calibration but doesn't always add ranking:**
- Typhoon: AUC 0.569 → 0.797, flags 114 → 27 items (F1 0.448 → 0.523)
- SeaLLM: AUC 0.653 → 0.812 (F1 0.494 → 0.576)
- Qwen: argmax F1 0.455 → 0.632 (much better calibration) but AUC *falls* 0.809 → 0.774
  → F1 after leave-fold-out barely moves (0.571 → 0.576) because the threshold already compensates for calibration.
- OpenThaiGPT has no `chat_template` in its tokenizer (predates the standard); had to add Llama-2 `[INST]` manually.

**Reproducibility confirmed:** this Qwen run compared against `open_qwen2.5-7b_preds.csv` from the earlier 127-item run:
mean |ΔP| = 0.0087 · Pearson r = 0.9942 · differ on only 1 item at threshold 0.5.
And re-running `frontier.py` reproduces the finding-9 table for all 5 models exactly → protocol correct.
*(Qwen2.5-1.5B on M1, which earlier got F1 0.356, was not re-run under the new protocol and is dropped from the table.)*

*(Files: `<model>_pred.csv` = bare prompt, `<model>_chat_pred.csv` = chat template · machine RTX 3060 Ti 8GB,
4-bit, batch 1 · must pin `transformers<5` or `load_in_4bit` breaks, see HANDOFF.md)*

**14. The rock bottom of the cost axis: a 3-character regex (`555`) beats every 7-8B open model**

Finding 13 asked "what F1 does $0 buy?" and answered with 7-8B open models. But there's something cheaper still — **no model at all.**
Measure surface cues on the 127-item gold (free, no API):

| cue | n | P(sarcastic\|cue) | lift |
|---|---|---|---|
| `??` | 5 | 0.600 | 2.54x |
| **`555`** | **31** | **0.581** | **2.46x** |
| **elongated vowels/letters (3+ repeats)** | **60** | **0.400** | **1.69x** |
| `จ้า` | 16 | 0.312 | 1.32x |
| `ค่ะ` | 42 | 0.095 | 0.40x |
| `นะคะ` | 19 | 0.053 | 0.22x |
| **`ครับ`** | 18 | **0.000** | **0.00x** |

(base rate = 0.236 · cues guessed to signal sarcasm but **absent from the data**: `เหอะ` appears in 1 item · `จ๋า`/`แหม`/`ดีจัง` in 0)

**Deciding on `555` alone (cue chosen leave-fold-out, seed 42) → F1 0.590 (prec 0.581 / rec 0.600)**
All 5 folds pick `555` identically → this number is not overfit to the test set; it can be reported.

| System | F1 | cost | latency | needs GPU |
|---|---|---|---|---|
| GPT bot (gpt-4.1-mini + threshold) | 0.727 | $0.015/run | 751 ms | no |
| **regex `555`** | **0.590** | **$0** | **~0 ms** | **no** |
| Qwen2.5-7B / SeaLLM-7B | 0.576 | $0 | seconds | yes |
| Typhoon-8B | 0.523 | $0 | seconds | yes |
| OpenThaiGPT-7B | 0.453 | $0 | seconds | yes |

=> **7-8B open models needing an 8GB card + 4-bit quantization still lose to a 3-character regex** on this domain.
The true "$0 floor" of this task is not a local LLM but substring matching.

**2 counterintuitive side results:**

1. **Normalizing away elongated vowels (e.g. "ดีม๊ากกกก" → "ดีมาก") *destroys* signal, it doesn't clean.**
   Elongation has lift 1.69x and appears in 47% of items · the actual savings would be 1.33% of characters
   = ~$0.0002 per run → **trading a real feature for pennies. Not worth it.**
2. **Polite words are an *inverse* signal:** `ครับ` appears in 18 items, sarcastic in **0** · `นะคะ` 0.22x · `ค่ะ` 0.40x

**But feeding the cues to the LLM = no help (measured, free, no re-firing the API):**
leave-fold-out logistic regression on [GPT's prob] vs [GPT's prob + 14 cue flags]:

| features | F1 | prec | recall |
|---|---|---|---|
| GPT prob only | **0.727** | 0.667 | 0.800 |
| GPT prob + cue flags | 0.667 | 0.667 | 0.667 |

**ΔF1 = -0.061 · 95% CI [-0.172, +0.040] · P(cues help) = 11.9%** · McNemar 3/5
→ GPT already reads these cues; adding them just adds parameters that overfit at n=127.
(coefficients: `gpt_prob` +1.86 · `ครับ` -1.25 · `elong` +1.02 · `??` +1.00 → cues are real but redundant)

**Confirmed a second way: embedding kNN also gets ~0.59 (free, no API)**

If a single regex getting 0.590 were a fluke, another cheap method shouldn't match it — but it does.
Encode the 127 gold items with multilingual-e5-large and do leave-fold-out kNN (neighbors pulled from other folds only, no leak):

| System | F1 | cost | needs GPU |
|---|---|---|---|
| GPT bot | 0.727 | $0.015/run | no |
| **regex `555`** | **0.590** | $0 | no |
| **kNN on e5-large (k=1)** | **0.588** | $0 | at encode time |
| Qwen2.5-7B / SeaLLM-7B | 0.576 | $0 | yes |
| Typhoon-8B | 0.523 | $0 | yes |
| OpenThaiGPT-7B | 0.453 | $0 | yes |

=> **Two unrelated cheap methods (lexical regex and embedding kNN) both land at ~0.59.**
Meanwhile the 7-8B open LLMs sit below at 0.45-0.58 → "the $0 floor" is not a point but a *layer* at ~0.59.
This conclusion is stronger than "one regex got lucky."

**Side result: dynamic few-shot (RAG) is unlikely to work here; measured, free**

The idea "retrieve the 2 most similar examples into the prompt instead of fixed few-shot" relies on the assumption
that *semantically similar text = same label.* Measure that assumption directly on 3 encoders:

| encoder | top-2 neighbors' labels match query | kNN F1 | **gap** (sarc↔sarc minus sarc↔non-sarc) |
|---|---|---|---|
| WangchanBERTa | 0.787 | 0.481 | **-0.058** |
| BGE-M3 | 0.772 | 0.528 | **-0.001** |
| multilingual-e5-large | 0.787 | 0.588 | **-0.003** |

(chance = 0.639 · gap > 0 would mean sarcastic items cluster together)

**The gap is negative for all 3.** Sarcastic items are **not** closer to each other than to non-sarcastic items.
A stronger encoder pushes the gap from -0.058 toward ~0 (i.e. "no information"), not positive.
→ retrieving the top-2 most similar gives examples whose labels are **barely correlated** with the item being judged.
For *sarcastic* items (the minority class, already weak on recall), this risks pulling in "non-sarcastic" examples that mislead.

Together with finding 10 (hand-built balanced synthetic few-shot gave +0.014, not significant, at 2× cost)
→ **not worth the effort/money on RAG few-shot for this dataset** (logged as "tested, not merely untried").
*(An open note: the whole-set gap is ~0 but kNN k=1 still gets 0.588 → signal exists in very local neighborhoods,
not at the class-structure level; not yet dug into why.)*

**Side result 2: semantic cache (caching by semantic similarity) doesn't work here; measured, free**

The idea "if a new sentence is 95% similar to an old one, return the old answer without calling the LLM" relies on the same RAG assumption:
*similar text = same answer.* Measure directly across all 8,001 text pairs (encoder: multilingual-e5-large):

| similarity threshold | pairs meeting it | % of all pairs | wrong answers returned |
|---|---|---|---|
| 0.95 - 0.99 | **0** | 0.0% | (cache never fires) |
| 0.93 | 27 | 0.3% | 3.7% |
| 0.90 | 618 | 7.7% | **15.2%** |

**The max similarity in the dataset is 0.942** — no pair reaches 0.95.
→ set the threshold at 95% = cache **hit rate 0%** (writing the code passes nothing through).
→ lower the threshold to make it work, and it starts returning wrong answers: at 0.90, 7.7% hit but **15.2% wrong**.
(Two random items have different labels 36.4% of the time → better than random but far from "safe.")

**The top 5 most-similar-but-different-label pairs are all restaurant reviews that happen to resemble one sarcastic review.**
They're similar because of *the same topic*, not *the same stance* — the same mechanism as the gap result above.
The "similar = same answer" assumption fails specifically for sarcasm, because sarcasm is defined by *intent opposite to the surface content*,
and the surface content is the only thing the embedding measures · (and the saving is only ~$0.0001 per cacheable call).

**Important limitation (untested):** `555` is Thai social-media language; this gold set is Wongnai reviews + Wisesight tweets.
Finding 12 already showed cross-domain precision collapses 0.68 → 0.40, and word-level cues **should transfer worse than models, not better.**
→ this conclusion is limited to *this domain* only. **Not yet measured on Pantip** (the `domain_*_labeled.csv` sets are in .gitignore,
not on this machine). To close the question, run `eval_domain.py` on the 55 Pantip items on the machine that has those files.

*(Analysis scripts are in the session scratchpad, not committed; recomputable from `gold.csv` +
`frontier_probs_gpt-4.1-mini.csv` with StratifiedKFold(5, shuffle, seed 42).)*

**15. 3-way micro-router: pay 40% of full price, get 95% of the GPT bot's F1**

Finding 14 puts the "$0 floor" at ~0.59; the GPT bot is at 0.727. The remaining question is
**how much you must pay to climb between them** — not "pay everything or pay nothing."

`router.py` uses **two** thresholds (cascade.py uses one), splitting into 3 paths:
prob < lo → decide "not sarcastic" itself (free) · prob ≥ hi → decide "sarcastic" itself (free) ·
in between → **unsure** → send to GPT (pay only for these).
Sweep the escalation budget b = fraction of items sent onward (uncertainty sampling around tau), leave-fold-out,
3 seeds (42/7/2024), items in fold k use thresholds from the other 4 folds only.

| budget | escalate | F1 | prec | recall | calls | cost | % of GPT bot |
|---|---|---|---|---|---|---|---|
| 0.00 | 0 (0%) | 0.590 | 0.537 | 0.656 | 0 | **$0** | 81% |
| 0.10 | 15 (12%) | 0.629 | 0.579 | 0.689 | 15 | $0.016 | 87% |
| 0.20 | 27 (21%) | 0.650 | 0.591 | 0.722 | 27 | $0.028 | 89% |
| 0.30 | 40 (31%) | 0.666 | 0.619 | 0.722 | 40 | $0.042 | 92% |
| **0.40** | **51 (40%)** | **0.690** | 0.644 | 0.744 | 51 | **$0.053** | **95%** |
| 0.70 | 89 (70%) | 0.694 | 0.650 | 0.744 | 89 | $0.094 | 95% |
| 1.00 | 124 (98%) | 0.721 | 0.663 | 0.789 | 124 | $0.130 | 99% |

=> **Pay 40% of the price, get F1 0.690 = 95% of the GPT bot.** Past 0.40 the curve is flat
(0.40→0.70 nearly doubles the cost for F1 +0.004) → **the sweet spot is ~40%, not 100%.**
b=1.00 gives 0.721 vs the reported GPT bot 0.727 → endpoints agree = the leave-fold-out pipeline is correct.

**A side result more important than the router itself: tuning WCB's threshold leave-fold-out *makes it worse***

| threshold method | F1 (mean, 3 seeds) | reportable? |
|---|---|---|
| tune on all of gold, measure on the same gold | **0.627** | **no — leak** |
| tune leave-fold-out (from the other 4 folds) | 0.556 | yes |
| **no tuning, use 0.5 directly** | **0.590** | yes |

At n=127 (30 positives), tuning the threshold on 4 folds and applying it to the 5th **loses to no tuning at all by 0.034.**
The gap between 0.627 (leak) and 0.556 (honest) = **0.071 is pure illusion.**
→ `router.py` defaults to `--tau-mode fixed` (0.5) · `--tau-mode tuned` is there to reproduce this result.
Same mechanism as finding 13 (40% of the old gap was measurement error) — **a tuning knob at small n is a debt, not an asset.**

And WCB at 0.5 gets 0.590, which **matches regex `555` (0.590) and e5 kNN (0.588) exactly**
→ three unrelated free methods (lexical / embedding / neural) all land at 0.59.
Confirms that finding 14's "$0 layer at ~0.59" is a real ceiling for this task, not a coincidence of any one method.

**16. semantic cache: actually built, actually run, confirmed unusable (end-to-end)**

Finding 14 measured semantic cache *pairwise*; `semantic_cache.py` builds it as a **working system**
(Redis backend + in-memory fallback, e5-large) and replays gold item by item like real traffic:
ask the cache first, on a miss call the LLM, then store the result.

| threshold | hits | hit rate | wrong answers | % of hits |
|---|---|---|---|---|
| 0.99 / 0.97 / **0.95** | **0** | **0.0%** | 0 | (cache never fires) |
| 0.93 | 12 | 9.4% | 1 | 8.3% |
| 0.90 | 37 | 29.1% | 2 | 5.4% |
| 0.85 | 97 | 76.4% | 13 | **13.4%** |

The max similarity between distinct items = **0.9422** (matches the pairwise value in finding 14 exactly)
→ at the commonly-recommended threshold of 0.95, **hit rate 0% — writing the code passes nothing through.**
→ lower the threshold until the cache works, and it immediately returns wrong answers, all to save ~$0.001/call.
(Streaming hit rate is higher than pairwise because it compares against *every item seen so far*, not pair by pair — more realistic.)

**The cause is the task, not the code.** A semantic cache assumes "similar = same answer,"
but sarcasm is defined by *intent opposite to the surface content*, and the surface content is the only thing the embedding measures
→ two restaurant reviews are similar because of **the same topic**, not **the same stance.**
Logged as "built, measured, doesn't work, for this reason," which is stronger than "not tried."

**17. async debate: cut latency 54% with cost unchanged (measured)**

`multiagent_debate.py` runs prosecutor → defender → judge in sequence, but **the prosecutor and defender never read each other's statements**
(see `_argue`: the user prompt is just `"ข้อความ: {text}"`) → the two are truly independent and can fire concurrently.
`async_debate.py` uses `asyncio.gather` → latency = max(prosecutor, defender) + judge instead of the sum.

**Actually run on 30 items (gpt-4o, concurrency 4, $0.165, 0 errors)** vs sequential on **the same 30 items**:

| | p50 | p95 | note |
|---|---|---|---|
| sequential | 4583 ms | 5643 ms | prosecutor → defender → judge |
| **async** | **2100 ms** | **3692 ms** | (prosecutor \|\| defender) → judge |
| **reduction** | **54%** | 35% | |

- The argue stage (prosecutor‖defender) p50 **1471 ms** = max of the two, not the sum → confirms gather works.
- **Call/token count exactly unchanged** (3 calls/item · $0.165/30 items = $0.0055/item, matches $0.695/127 items).
- throughput: wall-clock 20s per 30 items = **0.65 s/item** at concurrency 4 (sequential 4.58 s/item) = ~7× faster.
- 0 errors at concurrency 4 → no rate-limit collisions at this level.

**Reduced 54%, not the 33% expected initially,** because the initial guess assumed 3 equal-cost stages, but actually
the judge stage has `max_tokens=20` while the argue stage has `max_tokens=120` → pulling *one entire argue stage*
off the critical path saves more than 1/3. Lesson: **estimating latency requires looking at each stage's max_tokens, not counting stages.**

⚠️ **This run's F1 (0.400) cannot be used for anything.** The first 30 items of gold contain only **2 positives**
(gold.csv is not shuffled; positives cluster at the end of the file) → TP 1 / FN 1 / FP 2 is pure noise.
This run was designed to measure **latency**, a per-*item* property, so n=30 suffices · **do not report this F1.**

**And "checking that async gives the same result as sequential" cannot be done by re-running anyway.** Nowhere is temperature pinned
(default 1.0), meaning sequential can't even reproduce itself; any difference seen would be sampling noise, not architecture.
→ to verify properly you'd need `temperature=0` on both sides first, then compare (see "Reporting caveats").

The cost for a full 127-item run = **~$0.695** (gpt-4o, from `multiagent_preds_gpt_debate.csv`: 183,661 in / 23,585 out),
*not* the $0.169 that multiagent_debate.py prints for comparison — that's pipeline v2.

**18. load test: p50 = 28 ms because 83% of traffic never touches the LLM**

`loadtest.py` fires Poisson requests through the router (mock latency from measured values: WCB 26ms /
GPT call 751ms / debate 4557ms). `--mock` mode is free and lets you tune concurrency unlimited times.

At rps 8 / budget 0.20: escalate 17% · **p50 28 ms** · p95 923 ms · $0.088/min · 0 errors.
Split by path: auto n=137 p50 **26 ms** (free) vs escalated n=28 p50 **840 ms** (calls the LLM)
→ **the 32× gap between the two paths is "the price of uncertainty,"** and the whole system's p50 rides the free path.

Concurrency sweep at rps 25 (12s) finds a clear bottleneck:

| concurrency | throughput | time to drain the queue |
|---|---|---|
| 2 | 11.70 rps (target 25) | 26.4s **queue overflow** |
| 4 | 20.87 rps | 14.9s |
| 16 | 25.08 rps | 12.4s on target |

**A limitation to state plainly in the report:** gold has 127 items; this is not real production traffic.
This script can only answer "is the orchestration correct / what's p95 / what's cost per minute."
It **cannot** answer whether the system withstands 10k req/s — don't claim beyond that.
Metrics export as a Prometheus textfile (`--prom`) + `grafana_dashboard.json` (8 panels) ready to import.

**19. Expand gold to the full harvest pool (595 items, 104 positive): the thesis holds on 4.7× the data, and the "free upgrade" becomes significant for the first time**

Finding 11 ended with "to actually narrow the CI, label far more." This time we finished the pool:
labeled `harvest_to_review.csv` completely, **495/495** via `label_ui.py` (blind as before)
→ **`gold_v2.csv` 595 items, 104 positive (17.5%)**, then re-ran all 3 systems on every item → `v2_results/`.

| System (gold_v2, n=595) | F1 | prec | recall | calls | cost |
|---|---|---|---|---|---|
| baseline (argmax) | 0.337 | 0.206 | 0.933 | 595 | $0.275 |
| **baseline + threshold (leave-fold-out)** ⭐ | **0.377** | 0.238 | 0.894 | 595 | **$0.275** |
| v2 multi-agent (screener→verifier) | 0.357 | 0.222 | 0.913 | 1080 | $0.814 (2.96×) |

**19.1 The most important new result: threshold now beats the baseline significantly (first time in the project)**

paired bootstrap n=595: threshold−baseline = **+0.040, 95% CI [+0.018, +0.060]**
→ **CI does not cross 0** · McNemar **80 : 6** (heavily favors threshold).

Evolution of this comparison: n=127 → +0.035 (crosses 0) · n=302 → +0.015 CI [−0.033,+0.061] (crosses 0)
· **n=595 → +0.040 CI [+0.018,+0.060] (does not cross)**
=> finding 1 ("reading the logprob you already paid for = a free upgrade") is promoted from a *trend* to a
**data-confirmed conclusion**, at identical cost ($0.275 both · 595 calls both).

**19.2 The main thesis holds, and the point estimate tilts more clearly toward the single agent**

Compare the pair that actually answers the thesis (multi-agent vs the **fair baseline** = threshold, not argmax):

| n | multi-agent − threshold | 95% CI | CI width | P(multi not better) | McNemar (multi : thr) |
|---|---|---|---|---|---|
| 127 | +0.019 | [−0.073, +0.113] | 0.186 | 36% | 7 : 8 |
| 302 | +0.002 | [−0.044, +0.050] | 0.094 | 48% | 17 : 42 |
| **595** | **−0.019** | **[−0.047, +0.009]** | **0.056** | **91%** | **49 : 83** |

- **Still crosses 0 → can't conclude multi-agent is worse** but at n=595 the direction is much clearer:
  the chance multi-agent is *not* better = **91%** and McNemar tilts nearly 2:1 toward the single agent.
- Costs **2.96×** more (1080 calls vs 595) for a negative point estimate.
=> The thesis conclusion doesn't change but hardens: **no evidence multi-agent is worth it, and the evidence starts to tilt toward not-worth-it.**

**19.3 Hard negatives depress every architecture equally (absolute F1 keeps falling)**

Absolute F1 falls from ~0.45 (n=302) to ~0.35 (n=595) because the added items are all hard negatives:
the harvest pool was keyword+LLM-selected to "look sarcastic," but the human judged **403/495 (81%)** of them not sarcastic
→ the positive fraction dilutes (22% → 17.5%) and FP surges equally across systems (threshold: FP 297).
**Precision collapses identically; adding an agent doesn't help** — this is a property of the *data*, not the architecture.

GPT is even more "overconfident" here: extreme probs (<0.01 or >0.99) on **404/595 items (68%)**,
yet notably **even with little probability mass left, the threshold still extracts a significant gain** (19.1)
· leak-tuning on the full set gets 0.381 vs leave-fold-out 0.377 → a gap of only 0.004
(vs finding 15 where WCB's leak gap was 0.071), meaning **GPT's threshold is stable, not overfitted.**

**Bias to declare (repeated from finding 11):** gold_v2 is **not** an i.i.d. representation of real traffic;
negatives were pre-selected to be hard → **absolute F1 can't be compared across sets**, only paired within one set.
`Gold/random_to_label.csv` (250 truly random items from the 72,865 pool) is prepared for readable absolute numbers — **not yet labeled.**
A useful flip side: gold_v2 works as a **hard set** to measure any system's hard-negative robustness directly.

**20. Test on a truly random set (245 items, 4.9% sarcastic): all the project's headline numbers are artifacts of the mined set, and "reading confidence" is the one thing that makes the system usable**

Every earlier gold set (127 / 302 / 595) was **keyword+LLM pre-selected** to be sarcasm-dense (17-24%)
→ absolute F1 can't be read into the real world (declared since finding 11). This closes that gap:
250 truly random items from the 72,865 pool (seed 42), fully labeled blind → **true base rate = 4.9% (12/245)**
vs the harvest pool's ~22% → **the keyword filter inflates sarcasm density ~4.5×.**

| System (random set, n=245, 12 positive) | F1 | prec | recall | acc | flag rate |
|---|---|---|---|---|---|
| always guess "not sarcastic" | 0.000 | — | 0.000 | **0.951** | 0% |
| lexical cues (the deployed free tool) | 0.090 | 0.055 | 0.250 | 0.751 | 22% |
| GPT-4o baseline (argmax) | 0.202 | 0.115 | 0.833 | 0.678 | **36%** |
| **GPT-4o + threshold (leave-fold-out)** ⭐ | **0.462** | 0.429 | 0.500 | 0.943 | 6% |

(threshold F1 95% CI = [0.182, 0.688] — wide because there are only 12 positives; absolute numbers read as a rough level, but the direction is clear)

**20.1 "The $0 floor at F1 0.59" is an artifact of the mined set — on real data it drops to 0.09**

Finding 14 celebrated free regex/cue methods hitting F1 ~0.59 ("the bottom of the cost axis"),
but that was measured on a 24%-sarcastic set. **On the truly random 4.9% set the cue method falls to F1 0.090** (prec 0.055),
because the same cues that hit sarcasm in the mined set now fire all over "people talking normally" (52 FP).
=> **Not a property of the method but the arithmetic of a rare class**: precision dies when positives are rare.
The true "$0 floor" of this task in the real world = **barely better than always guessing "not sarcastic" (F1 0, acc 95%).**

**20.2 GPT argmax fails the same way — casts so wide a net that precision is 0.115**

GPT-4o at argmax (the same one that got F1 0.69 on the mined set) **flags sarcasm on 36% of all text**
when the truth is only 4.9% → 77 FP, precision **0.115**, F1 collapses to **0.202**.
On real data, "catching them all" (recall 0.833) is worthless if 7 of every 8 caught are false.

**20.3 The most important result: "reading the confidence already paid for" is the one thing that rescues the system**

Thresholding from the logprob (free, no extra calls) on the random set:
- flag rate falls 36% → **6%** (near the true base rate of 4.9%)
- precision jumps 0.115 → **0.429** (nearly 4×) · F1 jumps 0.202 → **0.462**
- **that's +0.26 F1 for free** — the biggest lever in the whole project, bigger than every kind of added agent combined.

GPT is still heavily "overconfident" (extreme probs 150/245 = 61%), yet the threshold still extracts a huge gain
=> finding 1 ("half the 'multi-agent win' was the baseline throwing away confidence it had already paid for")
**is proven here for the last and firmest time**: on real data, reading confidence is not a "small upgrade"
but **the line between a usable and an unusable system** (F1 0.46 vs 0.20) — and it's free.

**The whole project's overarching lesson (reinforced with real data):**
What people think must be fixed with architecture (add agents, debate, verifier) fixes nothing at all on real data —
the problem is **rare class + over-flagging**, which is fixed by one free thing you already have (the logprob).
**The best absolute F1 in the real world = 0.46, not the 0.73** once reported → the task is harder than every headline says.

**Caveat:** 12 positives is still few, so the CI is wide [0.182, 0.688] — read the absolute number as a "level"
(0.4-0.5, not 0.7), not a precise point · this set (`Gold/gold_random.csv`) is the project's first test set whose distribution
matches the real world, and can be used to measure the absolute performance of any system.

## Thesis conclusion (revised after findings 7-12)

**Before:** "multi-agent (recall-preserving verifier) genuinely helps but marginally (+0.054, cost 1.80×)"
**Now:** compared to a single agent that **squeezes the logprob dry**, that advantage is **indistinguishable from noise.**
On this task/dataset (n positives = 30), **there is no evidence any multi-agent system is worth the extra money.**

**Central lesson:** before asking *"add an agent?"*, first ask *"have I squeezed out everything the single agent already gives?"*
Comparing multi-agent to an untuned baseline is a biased comparison, because the free knob (threshold) does the same job
(buy precision, trade recall) almost as well without a single extra call.

**Caveat (multi-agent is not worthless):** the error profiles genuinely differ.
v2 keeps recall 0.967 (misses 1 positive) while threshold gets 0.833 (misses 5) → for "screening where nothing may be missed," v2 is still worth it.
What this finding says is "if judged on pure F1, multi-agent doesn't win," not "multi-agent loses in every situation."

**Original (kept as a record):**

multi-agent (recall-preserving verifier) **genuinely helps, but marginally**:
F1 +0.054 (precision +0.078, losing only -0.033 recall) in exchange for **cost 1.80× / calls 1.44×.**
"Worth it?" = depends on whether you'll pay nearly double for ~5 more F1 points.

The remaining bottleneck = **precision.** Even v2 lets 19 FPs through (because it's set to preserve recall).
There's a precision/recall tension a single verifier can't fully resolve → room for future work.

## Reporting caveats (important)

- **Don't use accuracy** — baseline acc 0.787 is nearly the same as guessing "not sarcastic" every time (0.764), because the data is skewed.
- GPT-4o's recall may be inflated by self-selection bias (see PROVENANCE.md); but this bias hits baseline and multi-agent equally → the *comparison* stays fair.
- n positives = 30 is small; every conclusion should carry a CI, not a lone point estimate.
- **Nowhere in the repo is OpenAI's `temperature` or `seed` pinned** → every call runs at default temperature = 1.0,
  which means **re-running the same system will not give the same predictions.** Consequences for reporting:
  - Every GPT system's F1 is the **result of a single run** → run-to-run noise is baked in.
  - `compare_systems.py` does paired bootstrap **across items** (item-level), which can't capture this noise
    → the reported CIs are **narrower than reality** (slightly overconfident).
  - It doesn't invalidate the *comparison* between systems, since all GPT systems face the same noise,
    but **don't interpret small differences (< ~0.03) as real** without multiple re-runs.
  - Future work: set `temperature=0` (or `seed=`) and re-run 3 times to measure the run-to-run SD
    before claiming 0.01-0.05 differences between architectures.
  (Discovered when trying to verify async debate matches sequential — **can't verify by re-running**
   because sequential can't reproduce itself, see finding 17.)
- The "F1 > 0.793 (upper CI of the baseline)" bar mentioned early on is too strict (an unpaired comparison);
  the correct test is the paired bootstrap of the difference, as above.

## Related files

- `baseline.py` / `baseline_preds_gpt.csv` — single agent
- `multiagent.py` / `multiagent_preds_gpt_conservative.csv` (v2) / `..._v1aggressive.csv` (v1)
- `wangchanberta.py` / `wangchanberta_preds.csv` — small model (5-fold CV, the reported numbers)
- `wcb_oof_probs.py` / `wcb_oof_probs.csv` — WCB out-of-fold probabilities (used to find the threshold in finding 6)
- `cascade.py` — system ⑦ cascade (WCB screener → GPT verifier) · `--dry-run` reproduces the finding-6 table for free
- `train_final_wcb.py` / `wcb_model/` — the real model for the web app (has seen all of gold → never use to measure)
- `compare_systems.py` — paired bootstrap + McNemar (covers all 3 systems)
- `gpt_threshold.py` / `gpt_screener_probs.csv` / `multiagent_preds_gpt_threshold.csv` — system ⑧ (finding 7)
  `--score` fires GPT to collect logprobs (~$0.09) · `--sweep` analyzes thresholds (free)
- `frontier.py` / `frontier_probs_*.csv` / `frontier_meta_*.csv` — cost-quality frontier, 5 models (finding 9)
  `--score` fires every model (~$0.20) · `--report` analyzes F1/cost (free)
- `compare_fair.py` — compare every system to baseline+threshold (finding 8, free, reuses existing preds)
- `analyze_baseline.py` — break results down by item origin (evidence of self-selection bias)
- `app.py` — web app to try/compare the 3 systems live (see README_WEB.md)

**Added in findings 15-18 (production / cost-aware):**
- `router.py` / `router_frontier.csv` / `multiagent_preds_gpt_router.csv` — 3-way micro-router (finding 15)
  **free, no API** (uses precomputed probs) · `--tau-mode tuned` reproduces the "tuning made it worse" result
- `semantic_cache.py` / `semantic_cache_result.json` — the real semantic cache (Redis + fallback) (finding 16)
  paired with `semantic_cache_test.py` (pairwise) · both free
- `async_debate.py` / `async_debate_metrics.json` / `multiagent_preds_gpt_debate_async.csv` — asyncio debate (finding 17)
  `--dry-run` estimates free · a real run needs a key (30 items ~$0.17 / full 127 items ~$0.70)
- `envload.py` — reads OPENAI_API_KEY from `.env` (on Windows, a User-scope env var doesn't reach an already-open process)
- `loadtest.py` / `loadtest_result.json` / `loadtest_requests.csv` — load test (finding 18)
  `--mock` (default) free · `--live` real · `--prom` emits a Prometheus textfile
- `grafana_dashboard.json` — 8-panel dashboard (escalation ratio / $ per minute / latency by path)

## Running the web app

```powershell
$env:OPENAI_API_KEY="sk-..."                       # optional, but without it only WangchanBERTa runs
C:/Users/thana/pt/Scripts/python.exe app.py        # open http://127.0.0.1:5000
```
Type Thai text and compare all 3 systems at once (answer + time + tokens + cost per item).
You can sample from gold, and if the text is in gold the app warns you that WCB has already seen it.
