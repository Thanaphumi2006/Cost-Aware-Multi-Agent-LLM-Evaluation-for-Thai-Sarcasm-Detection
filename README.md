# Cost-Aware Multi-Agent LLM Evaluation for Thai Sarcasm Detection

Is a multi-agent LLM system actually worth it over a single agent, for detecting **Thai sarcasm (ประชด/เสียดสี)**?
This project answers that on four axes at once: **quality (F1) · cost · latency · number of LLM calls**.

Every system is evaluated on the same gold set (127 items: 30 sarcastic / 97 not), with the same GPT-4o backend
and the same measurement harness. Systems are compared with a **paired bootstrap (5,000 resamples) + McNemar's test**,
which is the correct choice here because all systems run on the identical items.

## Using it for real (`Gold/predict.py`)

The findings collapse into one deployable recommendation: **one cheap model, one call, read the logprob, apply a
threshold** — no multi-agent machinery. `predict.py` packages exactly that, with operating points chosen from the
gold PR curve:

```bash
export OPENAI_API_KEY=sk-...
python Gold/predict.py "ขอบคุณที่ให้รอ 2 ชม. บริการดีจริงๆ"      # → {"label": 1, "prob": 1.0, "decision": "sarcasm"}
python Gold/predict.py --csv reviews.csv --out scored.csv         # batch a whole file
python Gold/predict.py "..." --op high_recall --review-band       # switch operating point
python Gold/eval_domain.py newdomain.csv                          # test on a NEW labeled domain
```

Repeated text is cached on disk (`.predict_cache.json`), so re-scoring the same items is free. The web app
(`python Gold/app.py`) also has a browser CSV-upload batch path built on the same predictor. **Before trusting it
outside restaurant reviews and tweets, run `eval_domain.py` on a human-labeled sample from your target domain** —
cross-domain transfer is the biggest untested risk.

| Operating point | Model | P / R / F1 | Use when |
|---|---|---|---|
| `balanced` (default) | gpt-4.1-mini (~$0.0001/item) | 0.68 / 0.83 / 0.75 | general use, cost-sensitive |
| `high_recall` | gpt-4o (~6× cost) | 0.43 / **1.00** / — | "never miss" screening → human reviews flags |
| `--review-band` | either | — | abstain in the 0.05–0.50 band, route to a human |

**Know before you ship** (measured, not guessed): **gpt-4.1-mini has a hard recall ceiling of ~0.83** — 5 of 30 gold
positives score ≈0, invisible at any threshold; use `high_recall` (gpt-4o) if misses are costly. **Precision caps
~0.68** on both models (balanced reviews are genuinely ambiguous), so a >0.80-precision setting isn't achievable.
Gold recall is inflated by self-selection bias (see `PROVENANCE.md`), the data is Wongnai reviews + Wisesight tweets,
and realistic F1 is ~0.70 — not 0.9x. Don't over-promise.

## Results

| System | F1 | Precision | Recall | LLM calls | Cost | Latency p50 |
|---|---|---|---|---|---|---|
| ① Single agent (baseline, argmax) | 0.690 | 0.526 | **1.000** | 127 | $0.094 | 751 ms |
| ① **Single agent + threshold** (reads its own logprob) ⭐ | 0.725 | 0.641 | 0.833 | 127 | **$0.094** | 751 ms |
| ② Pipeline v2 — screener → verifier | **0.744** | 0.604 | 0.967 | 183 | $0.169 | 967 ms |
| ③ Pipeline v1 (verifier flips freely) | 0.714 | **0.769** | 0.667 | 180 | $0.157 | 721 ms |
| ④ Debate (prosecutor + defender + judge) | 0.694 | 0.595 | 0.833 | 381 | $0.695 | 4,557 ms |
| ⑤ Hybrid (screener + debate panel, 4 agents) | 0.700 | 0.560 | 0.933 | 292 | $0.407 | 832 ms |
| ⑦ Cascade (WangchanBERTa screens → GPT verifier) ✗ | 0.628 | 0.500 | 0.844 | 119 | $0.124 | — |
| ⑥ WangchanBERTa (5-fold CV × 3 seeds) | 0.620 ±0.005 | 0.553 | 0.700 | **0** | **$0.00** | **26 ms** |

## Main finding

> **Before asking "should I add another agent?", ask "have I used everything the one agent already gives me?"
> On this task, once the single agent reads its own confidence, no multi-agent system beats it by a statistically
> significant margin — and several cost 2–7× more to tie.**

The single agent's output isn't really "sarcasm / not" — it's a *token* with a log-probability behind it. Reading
`P("sarcasm")` off that logprob and tuning one threshold (**leave-fold-out**, no leakage) lifts F1 from 0.690 to
**0.725 at identical cost and zero extra calls**. That score is the honest opponent every multi-agent system should
have been measured against.

When you do that (paired bootstrap, 5,000 resamples), **every system's confidence interval crosses zero**:

| System | ΔF1 vs tuned single agent | 95% CI | P(not better) | Cost |
|---|---|---|---|---|
| Pipeline v2 (best multi-agent) | +0.019 | [−0.073, +0.113] | 36% | 1.80× |
| Hybrid | −0.025 | [−0.117, +0.071] | 70% | 4.3× |
| Debate | −0.030 | [−0.158, +0.096] | 69% | 7.4× |

The multi-agent edge that earlier looked like "+0.054 over baseline" was **half handicap**: about half of it was the
baseline throwing away information it had already paid for, not the extra agents earning their keep.

**Two caveats that keep this honest, not hype:**

1. **This is "can't distinguish," not "proven equal."** At n(sarcasm) = 30 the CIs are wide — the data can't resolve
   a ~0.02 F1 difference either way. A larger gold set could still reveal a real multi-agent edge.
2. **The error profiles differ.** v2 holds recall 0.967 (misses 1 sarcastic item); the threshold trick buys precision
   by dropping recall to 0.833 (misses 5). For "screen and never miss," the second agent still earns its $0.075.
   The claim is *multi-agent doesn't win on F1*, not *multi-agent is useless*.

### The bigger lever is the model, not the architecture

Running that same tuned single agent across five models spanning a **25× price range** (leave-fold-out threshold each):

| Model | F1 | $ / run (127 items) | $ / 1M in |
|---|---|---|---|
| gpt-4.1-nano | 0.706 | $0.0038 | $0.10 |
| gpt-4o-mini | 0.676 | $0.0056 | $0.15 |
| **gpt-4.1-mini** ⭐ | **0.727** | $0.0150 | $0.40 |
| gpt-4.1 | 0.716 | $0.0752 | $2.00 |
| gpt-4o | 0.725 | $0.0940 | $2.50 |

F1 stays in a 0.05 band across the whole range — **smaller than the ±0.10 CI at n=30, i.e. mostly noise** — and the
**highest F1 belongs to a model 6× cheaper than gpt-4o**, the model this entire project was built on. The punchline,
paired and head-to-head: **a cheap single agent (gpt-4.1-mini + threshold, F1 0.727, $0.015) is statistically tied
with the flagship two-agent pipeline (gpt-4o v2, F1 0.744, $0.169) — at 1/11th the cost** (ΔF1 +0.016, 95% CI
[−0.094, +0.135]).

So the real "cost-aware" move here isn't picking an architecture — it's **picking a cheap model and reading its logprob.**
(Model prices are estimates in `Gold/frontier.py`; verify against current pricing. Token counts are measured exactly.)

Full numbers, confidence intervals, and McNemar counts → **[`Gold/RESULTS.md`](Gold/RESULTS.md)** (findings 6–9)

## Caveats (read before citing any number)

- **Do not use accuracy.** The baseline scores 0.787, which is barely above the always-predict-"not sarcasm"
  floor of 0.764 — the class distribution is skewed, so accuracy is meaningless here.
- **Self-selection bias.** Part of the gold positives were mined with GPT-4o, so GPT-4o's recall is likely inflated.
  The bias hits every system equally, so the *comparison between systems* remains fair
  (details: [`Gold/PROVENANCE.md`](Gold/PROVENANCE.md)).
- **n(sarcasm) = 30.** Every conclusion needs a confidence interval, not a bare point estimate. No amount of prompt
  tuning narrows that CI — only a larger gold set will.

## Repository layout

```
Gold/
  gold.csv                 evaluation set, 127 items (sources: Wongnai + Wisesight)
  labeling_rubric.md       annotation rubric -- sarcasm requires feigned praise ("การเสแสร้ง")
  baseline.py              (1) single agent
  multiagent.py            (2) + (3) pipeline (detector -> verifier)
  multiagent_debate.py     (4) debate
  multiagent_hybrid.py     (5) hybrid
  wangchanberta.py         (6) small model, 5-fold CV -- these are the reported numbers
  compare_systems.py       paired bootstrap + McNemar
  predict.py               deployable predictor (the research conclusion, packaged)
  app.py                   web demo: try / batch / YouTube, with correction-learning
  eval_domain.py           measure the model on any labeled domain (metrics + CI)
  label_any.py             terminal labeler for any text file
  fetch_yt_comments.py     pull Thai YouTube comments  ·  validate_yt.py (one-command)
  *_preds_*.csv            per-item predictions for every system (fully auditable)
  RESULTS.md REPORT.md SLIDES.md
```

## Web demo

```bash
pip install openai pandas numpy scikit-learn flask yt-dlp    # + torch transformers for WangchanBERTa
python Gold/app.py                                           # → http://127.0.0.1:5000
```

Open the page and paste your OpenAI key into the key box (kept in server memory only, never written to disk).
No key? WangchanBERTa still runs offline. The demo has three tabs:

- **Try one** — type a Thai sentence and compare all three systems side by side (single agent, 2-agent pipeline,
  WangchanBERTa) with per-item cost, latency, and token counts.
- **Batch** — upload a CSV (a `text` column) or paste lines; get a results table you can download. Repeated text is
  cached, so re-scoring is free.
- **YouTube** — paste a video link; it pulls the Thai comments, classifies them, and lists the sarcastic ones **5 per
  page**. Each row has a **"wrong"** button: click it and your correction is fed back as a few-shot example, so the
  model gets similar comments right on the next pass. *(This is in-context learning — it does not retrain the model —
  and the UI says so. Corrections also accumulate as a labeled set for `eval_domain.py`.)*

> **Caveat.** The model is validated only on restaurant reviews + tweets (F1 ~0.72). On other domains (YouTube, news, formal
> text) results are a guess — the demo over-flags praise on YouTube, which is exactly why the correction feature and
> the warnings are there. To measure a new domain properly, label a sample and run `eval_domain.py`.

Runs on `127.0.0.1` only — do not expose it to the internet, since the key box would let any visitor spend your key.

## Reproducing the research

```bash
cd Gold
python baseline.py               # (1) single agent
python multiagent.py             # (2) pipeline
python compare_systems.py        # compare all systems + statistics (paired bootstrap + McNemar)
```

**Not in the repo** (regenerable): `Gold/wcb_model/` (401 MB — run `train_final_wcb.py`) and the raw scrape
`raw_texts.csv` / `scored_texts.csv` (~230 MB), neither of which is needed to reproduce the results.
