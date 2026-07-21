# Description of all files in the project

Project: Thai sarcasm detection + a harness to compare multi-agent vs. single agent vs. a small model.
This document explains what each file is, when it's used, and which file is the latest version.

---

## 1. Project planning documents

**project-plan-detailed.txt** — use this as the main one
The latest complete plan: a project summary, the thesis that sets it apart (doing it in Thai + adding a small model),
the 3 systems compared, a day-by-day plan for all 4 weeks with end-of-week checkpoints, and scope-cutting criteria if time is tight.

**project-plan-multiagent-harness.md / .txt** — earlier version (shorter)
The first multi-agent + harness plan, without the day-by-day detail and without the "sets it apart" thesis.
Both .md and .txt exist with the same content. Superseded by project-plan-detailed.txt; kept for reference.

**project-brief-thai-sarcasm.md** — oldest (before the pivot)
A summary from when the project was still "just sarcasm detection," before the multi-agent angle. Superseded; kept to show the origin of the idea.

---

## 2. Rubric + labeling practice

**sarcasm-guideline-and-practice.txt** — use this as the main one
Combines the sarcasm definition guide (definition, the "direct negative != sarcasm" boundary, a 3-question decision method,
example pairs, ambiguous-case rules) + an 18-item practice set with answers, all in one file.

**sarcasm-annotation-guideline.txt** — already merged into the file above
The definition guide as a separate version (guide only, no practice set).

**sarcasm-practice-set.txt** — already merged into the file above
The 18-item practice set as a separate version (practice only, no guide).

**annotation-instructions.txt** — used when actually labeling
How to label to_label.csv step by step: the meaning of each column, the labeling steps, anti-mistake shortcut rules,
speed techniques, how to measure agreement (Cohen's kappa) with a partner, and the criteria for "done."

---

## 3. Scripts (code to run)

**step1_load_data.py** — step 1
Load the 2 raw datasets (wisesight + wongnai), keep only the text, drop the original labels.
Produces: raw_texts.csv
(note: requires installing datasets<4.0 as specified in the file)

**round1_keyword_filter.py** — step 3 (round 1)
Read raw_texts.csv and score "sarcasm suspicion" with keywords/rules, split into piles, and mix a ratio into a labeling pile.
Produces: scored_texts.csv and to_label.csv

---

## 4. Explainer files (for understanding)

**how-keyword-scoring-works.txt**
Explains how the round-1 script scores (a signal+points table, a counting example, why a high score doesn't mean sarcasm).
For understanding or to send to a mentor/friend.

---

## 5. Files "produced by running the scripts" (don't exist until you run them)

These are not in the folder now; they appear when you run the scripts yourself:

- **raw_texts.csv** — from step1_load_data.py: raw text only (no labels)
- **scored_texts.csv** — from round1_keyword_filter.py: every text + score + signals (for inspection/tuning, not for labeling)
- **to_label.csv** — from round1_keyword_filter.py: the 400 selected items with an empty label column to fill (this is the file you actually label)
- **gold.csv** (not yet) — appears after you finish labeling to_label.csv and drop the Xs = the real ground truth for measuring every model

---

## Usage order (overview)

1. Read project-plan-detailed.txt for the whole-project picture
2. Read sarcasm-guideline-and-practice.txt + pass the practice set
3. Run step1_load_data.py -> raw_texts.csv
4. Run round1_keyword_filter.py -> to_label.csv
5. Label to_label.csv per annotation-instructions.txt -> gold.csv
6. (Week 2+) build single agent -> multi-agent team -> harness -> small model -> compare

## Current status
Week 1: rubric and scripts prepared + ran suspect selection to produce to_label.csv.
Next step: label ~200-300 gold items.
