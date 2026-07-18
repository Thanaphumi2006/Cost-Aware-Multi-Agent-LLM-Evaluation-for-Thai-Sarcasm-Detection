# Thai sarcasm as an lm-evaluation-harness task

Benchmark open Thai-capable LLMs on the **same 127-item gold set**, at $0 API cost, and
drop the results straight into the project's existing comparison pipeline.

## Why
- Extends the cost axis: "what F1 can **$0** of API buy?" (open models run locally).
- Reproducible / citable: `lm_eval --tasks thai_sarcasm ...` reruns the exact setup.
- Same per-item CSV format as `predict.py` / `batch_eval.py`, so `compare_systems.py`,
  paired bootstrap, and McNemar work across open **and** closed models.

## Scope (honest)
The harness is single-call, so it covers **single-model** baselines only (open + closed).
Keep using `multiagent.py` / `compare_systems.py` for the multi-agent pipelines. Both write
per-item prediction CSVs in the same schema, so cross-system stats still line up.

## Install
```
pip install lm-eval            # or:  pip install "lm-eval[vllm]"
```

## Run (from the Gold/ directory, so `gold.csv` resolves)
```
cd Gold
lm_eval --model hf \
        --model_args pretrained=<thai-model> \
        --include_path lm_eval \
        --tasks thai_sarcasm \
        --num_fewshot 0 \
        --log_samples --output_path lm_eval_out
```
Example open Thai / multilingual models to try (names as on the HF Hub):
`scb10x/llama-3-typhoon-...`, `SeaLLMs/SeaLLM-7B-...`, `openthaigpt/openthaigpt-...`,
`Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`.

On an M2 Mac use a small/quantized model and expect it to be slow but fine for 127 items
(`--model_args pretrained=...,dtype=float16` or an MLX/llama.cpp backend for speed).

## Score -> F1/precision/recall + project CSV
```
python lm_eval/score_lm_eval.py --samples lm_eval_out --out <model>_lmeval_pred.csv
```
This reads the harness `samples_*.jsonl`, turns the two-choice loglikelihoods into
`P(sarcasm)` (softmax), and writes `text, label, pred_prob, pred_label, pred_decision`.
Add `--threshold 0.5` to cut on the probability instead of argmax.

## How it works
- `output_type: multiple_choice`: the model scores the loglikelihood of "ใช่" vs "ไม่ใช่"
  after the same rubric prompt used by the GPT detector. No text generation, so it is fast
  and deterministic on any HF causal LM.
- The harness prints `acc`; the score script is the source of truth for F1/precision/recall
  (kept out of the YAML because the harness metric API differs across versions).

## Files
- `thai_sarcasm.yaml` — the task config (loads `gold.csv`, prompt, choices, metric).
- `utils.py` — `process_docs` (adds int `gold`) and `doc_to_text` (rubric prompt).
- `score_lm_eval.py` — turns harness output into F1 + the project's prediction CSV.
