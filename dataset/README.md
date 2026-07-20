---
license: cc-by-4.0
task_categories:
  - text-classification
language:
  - th
tags:
  - sarcasm-detection
  - thai
  - irony
pretty_name: Thai Sarcasm Gold Set
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: canonical
        path: canonical.csv
      - split: hard
        path: hard.csv
---

# Thai Sarcasm Gold Set (ประชด)

Human-labeled Thai sarcasm detection data from the study
[Cost-Aware Multi-Agent LLM Evaluation for Thai Sarcasm Detection](https://github.com/Thanaphumi2006/Cost-Aware-Multi-Agent-LLM-Evaluation-for-Thai-Sarcasm-Detection).

Every label was decided by a human annotator working **blind**: the annotator judged each text
before seeing any model's draft label. Label 1 = sarcastic (ประชด), 0 = not sarcastic.
Texts where sarcasm cannot be judged without outside context were excluded rather than guessed.

## Two splits, two purposes

| Split | Items | Sarcastic | Use it for |
|---|---|---|---|
| `canonical` | 127 | 30 (24%) | The evaluation set behind the study's findings 1 to 18. |
| `hard` | 302 | 67 (22%) | A superset with 175 added items that were pre-selected to *look* sarcastic. A stress set: absolute scores here are not comparable to `canonical` by design. |

## Definition of sarcasm used

An item is sarcastic only when the author **feigns praise or thanks while meaning the opposite**
(การเสแสร้ง). The labeling rubric draws three boundary lines that cause most disagreement:

- Direct complaints are 0: negative content without pretend praise is not sarcasm.
- Balanced reviews are 0: genuine praise plus genuine criticism side by side.
- Exuberant sincere praise is 0: enthusiasm alone is not irony.

The full rubric with real examples lives in the repository:
[`Gold/labeling_rubric.md`](https://github.com/Thanaphumi2006/Cost-Aware-Multi-Agent-LLM-Evaluation-for-Thai-Sarcasm-Detection/blob/main/Gold/labeling_rubric.md).

## Sources and provenance

Texts come from Wongnai restaurant reviews and the Wisesight social-media corpus (both public),
filtered by keyword suspicion scoring and then human-labeled. Two biases are documented and
should be considered before use:

1. **Selection bias.** Candidates were surfaced by keyword filters and LLM suspicion scores,
   not sampled uniformly. Some positives were mined with GPT-4o, which likely inflates that
   model family's measured recall. The bias affects all compared systems equally, so paired
   comparisons on this data are fair; absolute numbers should be read with care.
2. **The `hard` split is adversarial by construction.** Its added negatives look sarcastic
   (laughter markers, elongated vowels, effusive tone) but are sincere. Precision measured on
   `hard` will be much lower than on typical traffic for any system.

Details: [`Gold/PROVENANCE.md`](https://github.com/Thanaphumi2006/Cost-Aware-Multi-Agent-LLM-Evaluation-for-Thai-Sarcasm-Detection/blob/main/Gold/PROVENANCE.md).

## Fields

| Column | Meaning |
|---|---|
| `text` | The Thai text, as scraped (may contain line breaks). |
| `label` | 1 = sarcastic, 0 = not sarcastic. |
| `source` | Original corpus or mining batch the text came from. |
| `suspect_score` | Keyword suspicion score at mining time, where available. |
| `signals` | Keyword signals that fired at mining time, where available. |

## Citation

```bibtex
@misc{kunuthai2026thaisarcasm,
  author = {Kunuthai, Thanaphumi},
  title  = {Cost-Aware Multi-Agent LLM Evaluation for Thai Sarcasm Detection},
  year   = {2026},
  url    = {https://github.com/Thanaphumi2006/Cost-Aware-Multi-Agent-LLM-Evaluation-for-Thai-Sarcasm-Detection}
}
```
