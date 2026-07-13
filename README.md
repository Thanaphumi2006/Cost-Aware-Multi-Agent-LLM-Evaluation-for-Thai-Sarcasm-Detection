# Cost-Aware Multi-Agent LLM Evaluation for Thai Sarcasm Detection

Is a multi-agent LLM system actually worth it over a single agent, for detecting **Thai sarcasm (ประชด/เสียดสี)**?
This project answers that on four axes at once: **quality (F1) · cost · latency · number of LLM calls**.

Every system is evaluated on the same gold set (127 items: 30 sarcastic / 97 not), with the same GPT-4o backend
and the same measurement harness. Systems are compared with a **paired bootstrap (5,000 resamples) + McNemar's test**,
which is the correct choice here because all systems run on the identical items.

## Results

| System | F1 | Precision | Recall | LLM calls | Cost | Latency p50 |
|---|---|---|---|---|---|---|
| ① Single agent (baseline) | 0.690 | 0.526 | **1.000** | 127 | $0.094 | 751 ms |
| ② **Pipeline v2 — screener → verifier** ⭐ | **0.744** | 0.604 | 0.967 | 183 | $0.169 | 967 ms |
| ③ Pipeline v1 (verifier flips freely) | 0.714 | **0.769** | 0.667 | 180 | $0.157 | 721 ms |
| ④ Debate (prosecutor + defender + judge) | 0.694 | 0.595 | 0.833 | 381 | $0.695 | 4,557 ms |
| ⑤ Hybrid (screener + debate panel, 4 agents) | 0.700 | 0.560 | 0.933 | 292 | $0.407 | 832 ms |
| ⑥ WangchanBERTa (5-fold CV × 3 seeds) | 0.620 ±0.005 | 0.553 | 0.700 | **0** | **$0.00** | **26 ms** |

## Main finding

> **Constraining an agent's power correctly matters more than the number of agents or the depth of deliberation.**

The winning system is also the **simplest and cheapest** one. Its second agent can do exactly one thing — **reject**
(flip 1→0). It therefore preserves the recall = 1.000 that the screener already achieved for free, and buys
precision on top of it, instead of re-deciding every item from scratch and gambling that recall away.

Three independent lines of evidence point the same way:

1. **v1 vs v2** — same system, the only difference is the tie-break rule for "what to do when unsure" → recall 0.667 vs 0.967.
2. **Pipeline vs debate** — 3× the agents, 4.1× the cost, 4.7× the latency, and it still **loses** (0.694 vs 0.744).
3. **Hybrid** — allow deliberation *inside* the constrained frame, and it **still loses** (0.700): after hearing both
   sides the judge becomes more hesitant and rejects only 5 items instead of the verifier's 8.

Full numbers, confidence intervals, and McNemar counts → **[`Gold/RESULTS.md`](Gold/RESULTS.md)**

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
  app.py                   web app for testing and comparing systems live
  *_preds_*.csv            per-item predictions for every system (fully auditable)
  RESULTS.md REPORT.md SLIDES.md
```

## Running it

```powershell
pip install openai pandas numpy scikit-learn flask torch transformers sentencepiece protobuf
$env:OPENAI_API_KEY="sk-..."     # never commit a key to the repo

cd Gold
python baseline.py               # (1) single agent
python multiagent.py             # (2) pipeline
python compare_systems.py        # compare all systems + statistics
python app.py                    # web app at http://127.0.0.1:5000
```

**Not in the repo** (regenerable): `Gold/wcb_model/` (401 MB — run `train_final_wcb.py`) and the raw scrape
`raw_texts.csv` / `scored_texts.csv` (~230 MB), neither of which is needed to reproduce the results.
