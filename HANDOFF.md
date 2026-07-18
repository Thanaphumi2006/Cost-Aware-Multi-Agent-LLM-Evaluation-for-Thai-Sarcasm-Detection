# Handoff — open-model eval on the RTX 3060 Ti box  ✅ DONE (2026-07-18)

The task this file described is **complete**. All four open models ran on the 127-item gold set
at $0 API cost, twice each (bare prompt + chat template). Results are in `Gold/RESULTS.md`
finding 13, which was **rewritten** — the original version had a wrong baseline and an unfair
protocol. Keep reading only if you need to re-run or extend this.

## What was found
- GPT bot (gpt-4.1-mini + threshold) **0.727** beats every open model, P ≥ 99.4%.
- Best open models: Qwen2.5-7B and SeaLLM-7B, both **0.576** (chat template + leave-fold-out).
- The gap is **+0.152**, not the +0.251 originally reported — ~40% of the old gap was
  measurement error, not capability. See finding 13 for the three causes.
- Thai-specialized models gave **no advantage** over general Qwen.

## Environment that actually works on Windows (3 fixes the old version got wrong)
```
python -m venv C:\ve                # 1. SHORT PATH — see below
C:\ve\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
C:\ve\Scripts\python -m pip install "lm-eval==0.4.12" "transformers<5" accelerate datasets bitsandbytes pandas
```
1. **Venv must live at a short path.** `lm-eval` ships deeply nested task YAMLs
   (`arabic_leaderboard_*`) whose paths exceed Windows' 260-char `MAX_PATH` when the venv sits
   inside this repo folder. pip fails with `OSError: [Errno 2] No such file or directory`.
   `C:\ve` fixes it without admin. (Enabling `LongPathsEnabled` would too, but needs admin.)
2. **`--index-url .../cu128` is required.** Plain `pip install torch` gives a CPU-only wheel on
   Windows, and then `load_in_4bit` + `--device cuda` cannot work at all.
3. **`transformers<5` is mandatory.** transformers 5.x removed the `load_in_4bit` passthrough
   (`TypeError: Qwen2ForCausalLM.__init__() got an unexpected keyword argument 'load_in_4bit'`).
   lm-eval 0.4.12 still passes it, and the CLI cannot construct a `BitsAndBytesConfig`.
   `pandas` is also needed — `score_lm_eval.py` imports it but the old install line omitted it.

Verify with: `C:\ve\Scripts\python -c "import torch;print(torch.cuda.is_available())"` → `True`

## Models and licenses
All four models were **already in the HF cache** (`~/.cache/huggingface/hub`, ~58 GB total) and a
token file already existed, so no license clicking or downloading was needed. If starting on a
fresh box, accept the gated licenses for `scb10x/llama-3-typhoon-v1.5-8b-instruct`,
`SeaLLMs/SeaLLM-7B-v2.5`, `openthaigpt/openthaigpt-1.0.0-7b-chat`, then `huggingface-cli login`.
(Note: `huggingface_hub` 0.36 uses `huggingface-cli login`; only 1.x renames it to `hf auth login`.)

## Running it (from `Gold/`)
```
C:\ve\Scripts\lm_eval --model hf --model_args pretrained=<repo>,load_in_4bit=True ^
  --device cuda --batch_size 1 --include_path lm_eval --tasks thai_sarcasm ^
  --num_fewshot 0 --apply_chat_template --log_samples --output_path out_<name>_chat
C:\ve\Scripts\python lm_eval\score_lm_eval.py --samples out_<name>_chat --out <name>_chat_pred.csv
```
- **`--batch_size 1`.** A 7B at 4-bit peaks at ~7959/8192 MiB on this card; 8B has no headroom.
- **`--apply_chat_template` matters a lot.** Without it these instruct models lose calibration
  badly (Typhoon flagged 114/127, OpenThaiGPT 126/127, vs 30 true positives).
- **OpenThaiGPT has no `chat_template`** in its tokenizer and errors out. Build a tokenizer copy
  carrying the Llama-2 `[INST]` template and pass `tokenizer=<path>` in `--model_args`.
- Runtime ~4 min/model once weights are warm in the OS cache; the first cold load took ~85 min.

## Comparing to the GPT bot — use the same protocol on BOTH sides
The original finding 13 compared *threshold-tuned GPT* against *raw-argmax open models*. Don't.
Apply the project's leave-fold-out protocol (`gpt_threshold.py:150-157`, StratifiedKFold 5,
shuffle, seed 42) to the open model's `pred_prob` too, then paired-bootstrap.

**Join prediction CSVs by POSITION, not text.** `score_lm_eval.py:80` rewrites newlines to
spaces, so a text join silently drops every multi-line item (65/127 matched). All files derive
from `gold.csv` in order; assert the label sequences match, then index positionally.

## Files produced
`Gold/<model>_pred.csv` (bare prompt) and `Gold/<model>_chat_pred.csv` (chat template) for
`qwen7b`, `typhoon`, `seallm`, `openthaigpt` — schema `text,label,pred_prob,pred_label,pred_decision`,
the same as `predict.py`, so `compare_systems.py` / bootstrap / McNemar work across all systems.
