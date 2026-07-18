# Handoff — continue the open-model eval on the RTX 3060 Ti box

This file lets you move the work to the NVIDIA machine and give a fresh Claude full context.

## What this is
Cost-aware LLM evaluation for Thai sarcasm (ประชด) detection. The deployed bot is a single
gpt-4.1-mini call + a tuned logprob threshold (F1 ~0.72). Findings 1-13 are in `Gold/RESULTS.md`.

## The ONE task for the 3060 Ti box
Benchmark open Thai LLMs on the 127-item gold set at $0 API cost, to finish **finding 13**
(the open-model frontier). So far only Qwen2.5-7B ran (F1 0.444, loses to the GPT bot,
significant: +0.251 [+0.05,+0.46], P=99.3%). Still needed: the Thai-specialized models
(Typhoon, SeaLLM, OpenThaiGPT). The Mac (M1/M2, no CUDA) cannot run them; this box can.

## Step 1 — get the files (clean, no secrets travel)
On the 3060 Ti machine (Windows/Linux with the NVIDIA GPU):
```
git clone https://github.com/Thanaphumi2006/Cost-Aware-Multi-Agent-LLM-Evaluation-for-Thai-Sarcasm-Detection.git
cd Cost-Aware-Multi-Agent-LLM-Evaluation-for-Thai-Sarcasm-Detection
git checkout cost-experiments-open-models
```
You do NOT need an OpenAI key here — open models run locally.
You do NOT need `wcb_model/` or the scraped data — the open-model eval only needs `Gold/gold.csv`.

## Step 2 — install (in a fresh venv on that box)
```
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install "lm-eval==0.4.12" accelerate datasets bitsandbytes torch
```
`bitsandbytes` + `--device cuda` only works on the NVIDIA GPU (that is why it failed on the Mac).

## Step 3 — accept the gated model licenses (once)
Open each page while logged into huggingface.co and click "Agree/Access":
- https://huggingface.co/scb10x/llama-3-typhoon-v1.5-8b-instruct
- https://huggingface.co/SeaLLMs/SeaLLM-7B-v2.5
- https://huggingface.co/openthaigpt/openthaigpt-1.0.0-7b-chat
Then: `huggingface-cli login` (paste a token from huggingface.co/settings/tokens).
Qwen is open and needs none of this.

## Step 4 — run (8 GB card: 4-bit is required)
```
cd Gold
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,load_in_4bit=True ^
  --device cuda --batch_size 4 --include_path lm_eval --tasks thai_sarcasm ^
  --num_fewshot 0 --log_samples --output_path out_qwen7b
python lm_eval/score_lm_eval.py --samples out_qwen7b --out qwen7b_pred.csv
```
Repeat with `pretrained=scb10x/llama-3-typhoon-v1.5-8b-instruct`, `SeaLLMs/SeaLLM-7B-v2.5`,
`openthaigpt/openthaigpt-1.0.0-7b-chat` (change `--output_path` and `--out` each time).
(`^` = Windows line-continuation; on Linux use `\`. Or run the notebook `Gold/lm_eval/thai_sarcasm_colab.ipynb`.)
If "CUDA out of memory": drop `--batch_size` to 1.

## Step 5 — bring results back
Each run writes `<model>_pred.csv` (same format as `predict.py`: text,label,pred_prob,pred_label,pred_decision).
Commit + push them, or copy them back to the Mac, so the paired bootstrap vs the GPT bot can be run.

---

## Paste THIS into a fresh Claude on the 3060 Ti box (context handoff)

> I'm continuing a project: cost-aware LLM evaluation for Thai sarcasm detection. The repo is
> already cloned (branch `cost-experiments-open-models`) and `HANDOFF.md` at the root explains it.
> Deployed bot = single gpt-4.1-mini + tuned logprob threshold (F1 ~0.72). Findings are in
> `Gold/RESULTS.md`; the newest is finding 13, the "$0 open-model frontier": Qwen2.5-7B scored
> F1 0.444 and lost significantly to the GPT bot. My job on THIS machine (RTX 3060 Ti, 8 GB, CUDA)
> is to run the Thai-specialized open models (Typhoon, SeaLLM, OpenThaiGPT) that the Mac couldn't,
> using the lm-evaluation-harness task in `Gold/lm_eval/` (task name `thai_sarcasm`) on the
> 127-item `Gold/gold.csv`, with `load_in_4bit=True --device cuda`. Help me get those models
> running (bitsandbytes/CUDA, gated-license logins, OOM at 8 GB), then produce per-model
> prediction CSVs in the project format so I can compare them to the GPT bot. Read `HANDOFF.md`
> and `Gold/lm_eval/README.md` first, then walk me through step 4.

## Back on the Mac, when the CSVs return
A fresh Claude (or the same project) can run the paired bootstrap + McNemar of each Thai model
vs the GPT bot (`gold_pred.csv`) and update finding 13 in `Gold/RESULTS.md`.
