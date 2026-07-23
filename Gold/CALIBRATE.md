# Calibrating the detector to your own domain

**Why this is the most important step for real use.** All of `gold.csv` is Wongnai reviews +
Wisesight tweets. Finding 12 measured what happens when you move off that: precision fell from
**0.68 to 0.40** on Pantip, because the detector reads genuine praise as sarcasm. The fix is not a
new model, it is re-picking the escalation threshold on a labelled sample of *your* target content.
`calibrate_domain.py` does that using the exact deployed scorer (`predict.SarcasmDetector`,
`operating="balanced"` = gpt-4.1-mini @ t=0.095), so the numbers reflect what the demo will do.

Everything the domain data touches (`domain_*`) is gitignored, so scraped third-party text and your
labels never get committed.

## The three steps

```bash
export OPENAI_API_KEY=sk-...        # or put it in .env
cd Gold
```

### 1. Collect a sample from your domain

```bash
python calibrate_domain.py --name myshop --fetch https://www.youtube.com/watch?v=...   # or a Pantip/Reddit link
```

Writes `domain_myshop_labeled.csv` with a `text` column and an empty `label` column. `--limit N`
controls how many comments (default 120). You can also skip this and bring your own CSV — it only
needs a `text` column.

Aim for a sample big enough to have **≥ 30 sarcastic items**. Sarcasm is rare (~5% in the wild,
finding 20), so you may need to pull several hundred comments to get there.

### 2. Label it by hand

Open `domain_myshop_labeled.csv` and set `label` = `1` (sarcastic) or `0` (not) for each row, using
the same rubric as gold — sarcasm requires **feigned praise / การเสแสร้ง**, see
[`labeling_rubric.md`](labeling_rubric.md). Label *blind* (decide before looking at any model output),
or your labels drift toward the model's over-flagging.

Prefer a keyboard UI? Point `label_ui.py` at the file:

```bash
DOMAIN_LABEL_FILE=domain_myshop_labeled.csv python label_ui.py   # then open http://127.0.0.1:5001, tab "domain"
```

### 3. Calibrate

```bash
python calibrate_domain.py --name myshop --csv domain_myshop_labeled.csv
```

It scores each item once (cached to `domain_myshop_probs.csv`, so re-runs are free) and prints:

```
=== on your domain ===
  cue-only floor (answers 74/127)    P=... R=... F1=...   LLM calls 0/127 (0%)
  LLM @ deployed t=0.095             P=... R=... F1=...   LLM calls 127/127 (100%)
  LLM @ domain-tuned (5-fold)        P=... R=... F1=...   LLM calls 127/127 (100%)

=== what to do ===
  -> deploy threshold  t = 0.NNN  for domain 'myshop'
```

- **cue-only floor** = what you get if you never escalate (the free browser tier alone, with the
  finding-21 cut-off). If this already meets your bar, you may not need the paid model at all.
- **LLM @ deployed** = today's threshold on your domain. If precision here is poor, that is the
  cross-domain collapse finding 12 warned about.
- **LLM @ domain-tuned** = the honest, leakage-free estimate (leave-fold-out) of what re-tuning buys.
  The per-fold thresholds are printed too; a tight spread means the number is trustworthy.

## Deploying the result

Set the recommended threshold in [`predict.py`](predict.py), `OPERATING["balanced"]["t"]`. That one
number flows through `app.py`'s `/api/escalate`, `predict.py` CLI, and every batch tool.

If tuning barely moved F1 (the tool says so), the default already fits your domain — don't change it.

**No labels, or a moving target?** The demo's **"correct it"** button adapts in-context from examples
without any of this — press it when a result is wrong and similar cases improve next time. Calibration
is the rigorous path; the button is the zero-setup one.

## Reading the numbers honestly

- With < ~30 positives every F1 has a wide CI. Treat results as a *level* (is precision ~0.4 or ~0.8?),
  not a precise point.
- Score once, then re-analyse for free — the probs cache means you never pay twice for the same text.
- This measures the **escalation tier** (the paid model). The cue cut-off (finding 21) is fixed in the
  browser and needs no per-domain tuning.
