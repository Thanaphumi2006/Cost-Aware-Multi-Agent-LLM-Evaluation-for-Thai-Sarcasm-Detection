# Applying the detector: a short playbook

What each part is for, the exact commands, and the honest status of the three "apply" tracks.

## Run it

One command (macOS/Linux):

```bash
./setup.sh          # .venv -> pinned deps -> build the WCB model once -> serve at http://127.0.0.1:8000/app
```

Then open **http://127.0.0.1:8000/app**. (Manual/Windows steps: see the README "Run it" section.)

## Which tool for which job

| You want to... | Use | Notes |
|---|---|---|
| Try one sentence | the demo at `/app`, type it | instant free cue verdict; click "ให้ AI ช่วย" if it's unsure |
| Triage a comment thread | the demo, paste a **YouTube / Pantip / Reddit** link | fetch → scan → review → download the confirmed CSV |
| Triage a batch you already have | the demo, paste comments (one per line) **or upload a CSV** | works on any content, not just the 3 platforms |
| Auto-label a dataset unattended | `python Gold/autolabel.py --csv in.csv --out out.csv --eval` | AND-gate, ~0.90 precision in-domain; defers the rest to review |
| Guard **your** Thai sentiment pipeline | `python Gold/sarcasm_flag.py --csv scored.csv --out flagged.csv` | you bring `text` + `sentiment`; flags where sarcasm likely inverts a non-negative label. Also `flag(text, sentiment)` |
| Sentiment **end to end** (WangchanBERTa + guard) | `python Gold/sentiment_pipeline.py --csv texts.csv --out scored.csv` | runs a WangchanBERTa sentiment model then the guard, in one step. Also `analyze(text)` |
| Check accuracy on your own domain | `python Gold/calibrate_domain.py --name X --csv labelled.csv` | needs a labelled sample; runbook in `Gold/CALIBRATE.md` |

### Sarcasm-flagging for a sentiment pipeline

Sentiment models miscode sarcasm: a positive surface ("บริการดีมาก รอแค่ 2 ชั่วโมง") hides a negative
intent, so the pipeline mislabels a complaint as POSITIVE. `Gold/sarcasm_flag.py` adds a stage that flags
exactly those items (a non-negative label on sarcastic text) for a human to re-check — it never relabels
silently. Bring your pipeline's output as a CSV with a `text` column and its `sentiment` column; the tool
adds `sarcastic`, `flag` (review/ok), and `reason`. Or call it inline:

```python
from sarcasm_flag import flag
r = flag(text, sentiment=my_pipeline_label)
if not r["trust_sentiment"]:
    route_to_human(text)          # r["flag"] == "review"
```

Cost-aware (the free cue answers most rows; only cue-unsure text costs a gpt-4.1-mini call). Same domain
caveat as everything else: calibrate first, treat "review" as "a human should look", not a verdict.

**Wired end to end** (`Gold/sentiment_pipeline.py`): if you don't already have a sentiment model, this runs
one for you (WangchanBERTa, default `phoner45/wangchan-sentiment-thai-text-model`, auto-downloaded from HF
on first run) and applies the guard in a single pass. On a quick check it caught two sarcastic reviews the
sentiment model labelled POSITIVE, while leaving a correctly-negative sarcastic one alone. Swap the model
with `--model <hf-id>` or `WCB_SENTIMENT_MODEL=...`.

## What it is good at (and not)

- **Good: recall.** It catches almost all sarcasm (recall ~0.97) on the domains tested.
- **Weak: precision off-domain.** On content unlike reviews/tweets it over-flags. Finding 22 measured
  precision ~0.20 on political YouTube comments. The demo shows a "ระวังการจับเกิน" warning when a scan's
  flagged share is implausibly high.
- **So:** use it as a **recall-triage tool** (surface candidates, a human confirms), not an unattended
  auto-labeller, unless you have calibrated on your domain.

**Example real run** (40 comments from a political YouTube video, the full cascade): 62% flagged, 5 answered
free by WangchanBERTa, 22 GPT calls (~$0.003 total). The 62% is the over-flagging in action; the review
station is how you turn it into a clean list.

## Deploy it publicly

Verified production command (full runbook, nginx + TLS, in `Gold/HOSTING.md`):

```bash
OPENAI_API_KEY=sk-... TRUST_PROXY=1 gunicorn -w 2 -b 127.0.0.1:8000 --chdir Gold serve_public:app
```

`serve_public` exposes only `/app`, `/api/fetch_comments`, `/api/escalate`, `/healthz`; the key is
environment-only; requests are rate-limited and size-capped; fetching is SSRF-safe; hardening headers on
every response. Put it behind an HTTPS reverse proxy on your own domain. **Never expose `app.py`** (the
developer page has a key box and must stay local).

## The paper

`docs/paper.tex` is current (includes finding 21) and `docs/paper.pdf` is **stale**. The source romanizes
Thai (`prachot`, not ประชด) and uses only standard packages, so it compiles with **plain pdflatex** — no
Thai font, no xelatex. It is a single self-contained file.

Rebuild in ~2 minutes:
1. Overleaf → **New Project → Upload Project** (or **Import from GitHub**), add `docs/paper.tex`.
2. Menu → Compiler → **pdfLaTeX** (the default), then **Recompile**.
3. Download the PDF and drop it in as `docs/paper.pdf`.

Or locally, on any machine with a TeX install: `pdflatex paper.tex` (twice, for the references).

## Status of the three apply tracks

- **Use it — done.** The tool runs the full cascade on real content end to end and exports a reviewer queue.
- **Deploy publicly — production path verified locally** (gunicorn, 2 workers, `TRUST_PROXY` honored).
  Actual public reach needs your own domain, TLS certificate, and a host; that part is yours.
- **Rebuild `paper.pdf` — not possible in the dev sandbox** (no LaTeX toolchain, Homebrew blocked), but the
  source is pdflatex-clean and self-contained: upload `docs/paper.tex` to Overleaf and Recompile (steps above).
