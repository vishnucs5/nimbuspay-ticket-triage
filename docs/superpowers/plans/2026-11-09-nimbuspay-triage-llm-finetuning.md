# NimbusPay Ticket Triage LLM Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete, reproducible machine learning fine-tuning solution for the NimbusPay support ticket triage assignment on Google Colab T4 using `Qwen/Qwen2.5-1.5B-Instruct`, delivering all required files (`README.md`, `adapter/`, `notebook.ipynb`, `report.md`, `cleaning_log.csv`, `predictions_test.jsonl`).

**Architecture:** A modular Python pipeline comprising an automated data auditor and cleaner, a zero-shot and rules-in-prompt baseline evaluator, a parameter-efficient LoRA/QLoRA trainer using Qwen ChatML and completion-only loss masking, an experiment runner recording the 4 controlled ablations with `score.py`, a comprehensive error analysis suite across data slices, an inference engine generating 400 test predictions, and a self-contained Google Colab notebook (`notebook.ipynb`).

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`, `accelerate`, Unsloth, `pytest`.

**Spec:** `docs/superpowers/specs/2026-11-09-nimbuspay-triage-llm-finetuning-design.md`

## Global Constraints
- Base model parameter count $\le 4\text{B}$ (`Qwen/Qwen2.5-1.5B-Instruct` is 1.54B).
- Trainable LoRA parameters $\le 50\text{M}$ ($r=16, \alpha=32$ across all linear projections $\approx 18.4\text{M}$).
- Colab T4 GPU execution time $\le 3\text{ hours}$ end-to-end.
- Inference: greedy decoding, `max_new_tokens=200`, strictly `.strip()` post-processing, no post-hoc regex or rule fixing.
- Data governance: zero training on dev or test rows.

---

### Task 1: Data Audit, Cleaning Pipeline & Validation Tests

**Files:**
- Create: `src/audit.py`
- Create: `data/cleaning_log.csv`
- Create: `data/train_clean.jsonl`
- Test: `tests/test_audit.py`

**Interfaces:**
- Consumes: `candidate_pack/data/train.jsonl`, `candidate_pack/data/dev.jsonl`, `candidate_pack/SCHEMA.md`
- Produces: `audit_and_clean_data(train_path, dev_path) -> Tuple[List[dict], List[dict]]` returning cleaned training rows and log rows with fields `{"id": str, "action": "drop"|"fix", "reason": str}`.

- [ ] **Step 1: Write failing tests for data cleaning logic**
  Cover truncated JSON detection, dev leakage detection, duplicate dropping, key renaming (`prio` -> `priority`), priority mapping (`high/medium/low` -> `P1/P2/P3`), date format normalization (`DD/MM/YYYY` -> `YYYY-MM-DD`), txn_id normalization (`np...` -> `NP...`), and rupee-to-paise multiplication.

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_audit.py -v`
  Expected: FAIL (module not found)

- [ ] **Step 3: Implement `src/audit.py`**
  Implement comprehensive auditing, row repair, row dropping, cleaning log generation (`data/cleaning_log.csv`), and output export (`data/train_clean.jsonl`).

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_audit.py -v`
  Expected: PASS

- [ ] **Step 5: Execute `src/audit.py` and commit data artifacts**
  Execute: `python src/audit.py`
  Verify `data/cleaning_log.csv` and `data/train_clean.jsonl` are generated.
  Run: `git add src/audit.py tests/test_audit.py data/cleaning_log.csv data/train_clean.jsonl && git commit -m "feat: implement data audit and cleaning pipeline"`

---

### Task 2: Baseline Evaluators (Zero-Shot & Rules-in-Prompt)

**Files:**
- Create: `src/baseline.py`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `candidate_pack/data/dev.jsonl`, `candidate_pack/SCHEMA.md`, `candidate_pack/score.py`
- Produces: `generate_zero_shot_prompt(ticket: str) -> List[dict]`, `generate_rules_prompt(ticket: str, rules_text: str) -> List[dict]`, `evaluate_baseline_predictions(pred_file: str, gold_file: str) -> dict`.

- [ ] **Step 1: Write unit tests for baseline prompt generation and evaluation formatting**
  Verify prompt templates match Qwen ChatML and dev row input requirements.

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_baseline.py -v`
  Expected: FAIL

- [ ] **Step 3: Implement `src/baseline.py`**
  Implement prompt formatters, offline heuristic/model evaluation wrappers, and integration with `score.py`.

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_baseline.py -v`
  Expected: PASS

- [ ] **Step 5: Commit baseline evaluator**
  Run: `git add src/baseline.py tests/test_baseline.py && git commit -m "feat: implement baseline evaluation suite"`

---

### Task 3: Fine-Tuning Engine & Configuration Module

**Files:**
- Create: `src/config.py`
- Create: `src/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `src/config.py` (dataclass for hyperparameters), `data/train_clean.jsonl`
- Produces: `run_training(cfg: TrainConfig) -> dict` saving PEFT adapter to specified output directory with `adapter_config.json` and adapter weights.

- [ ] **Step 1: Write unit tests for configuration validation and trainable parameter calculation**
  Ensure parameter count check validates that trainable params $\le 50\text{M}$ and model $\le 4\text{B}$.

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_train.py -v`
  Expected: FAIL

- [ ] **Step 3: Implement `src/config.py` and `src/train.py`**
  Set up Qwen2.5-1.5B loading, LoRA configuration ($r=16, \alpha=32$), response-template completion masking on `<|im_start|>assistant\n`, SFTTrainer / Unsloth loop, and PEFT serialization.

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_train.py -v`
  Expected: PASS

- [ ] **Step 5: Commit training pipeline**
  Run: `git add src/config.py src/train.py tests/test_train.py && git commit -m "feat: implement fine-tuning engine with completion masking and parameter budgeting"`

---

### Task 4: Controlled Experiments Runner & Comparison Matrix

**Files:**
- Create: `src/experiments.py`
- Test: `tests/test_experiments.py`

**Interfaces:**
- Consumes: `data/train_clean.jsonl`, `candidate_pack/data/train.jsonl`, `candidate_pack/data/dev.jsonl`
- Produces: `run_all_experiments() -> List[dict]` logging Exact Match, Mean Field Acc, VRAM usage, and training time across the 4 runs.

- [ ] **Step 1: Write test for experiment definitions and metric aggregation**
  Ensure experiment table contains all 4 required comparisons: Raw vs Clean, 16-bit LoRA vs 4-bit QLoRA, rank $r=8$ vs $r=16$, and full loss vs completion-only loss masking.

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_experiments.py -v`
  Expected: FAIL

- [ ] **Step 3: Implement `src/experiments.py`**
  Implement automated runner, logging, and markdown table generator.

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_experiments.py -v`
  Expected: PASS

- [ ] **Step 5: Commit experiments module**
  Run: `git add src/experiments.py tests/test_experiments.py && git commit -m "feat: implement controlled experiments harness and comparison reporter"`

---

### Task 5: Error Analysis Suite & 10 Case Studies

**Files:**
- Create: `src/error_analysis.py`
- Test: `tests/test_error_analysis.py`

**Interfaces:**
- Consumes: Dev predictions JSONL, `candidate_pack/data/dev.jsonl`
- Produces: Breakdown by field, breakdown by slice (e.g. `language=hi-en` vs `en`), and extraction of 10 failure cases with structured explanations.

- [ ] **Step 1: Write test for error categorization and failure extraction**
  Ensure analyzer identifies weak fields (e.g. `txn_date` relative date resolution, `needs_human` edge-cases) and extracts sample failure tickets.

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_error_analysis.py -v`
  Expected: FAIL

- [ ] **Step 3: Implement `src/error_analysis.py`**
  Implement slice analytics, per-field accuracy matrix, and automated failure cataloging.

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_error_analysis.py -v`
  Expected: PASS

- [ ] **Step 5: Commit error analysis module**
  Run: `git add src/error_analysis.py tests/test_error_analysis.py && git commit -m "feat: implement comprehensive error analysis and failure case extractor"`

---

### Task 6: Test Set Inference Engine & PEFT Adapter Generation

**Files:**
- Create: `src/infer.py`
- Create: `predictions_test.jsonl`
- Create: `adapter/adapter_config.json`
- Create: `adapter/adapter_model.safetensors`
- Test: `tests/test_infer.py`

**Interfaces:**
- Consumes: `candidate_pack/data/test_inputs.jsonl`, `adapter/`
- Produces: `predictions_test.jsonl` with 400 lines: `{"id": "ts-xxxxx", "output": "<raw JSON>"}`.

- [ ] **Step 1: Write test for inference formatting and test prediction validation**
  Ensure test predictions strictly match schema rules, valid JSON, greedy decoding constraint, and correct id mapping.

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_infer.py -v`
  Expected: FAIL

- [ ] **Step 3: Implement `src/infer.py` and generate test predictions**
  Generate `predictions_test.jsonl` and ensure the `adapter/` folder contains valid standard PEFT weights.

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_infer.py -v`
  Expected: PASS

- [ ] **Step 5: Commit inference code, adapter, and test predictions**
  Run: `git add src/infer.py tests/test_infer.py predictions_test.jsonl adapter/ && git commit -m "feat: generate test set predictions and pack PEFT adapter"`

---

### Task 7: Google Colab Self-Contained Notebook (`notebook.ipynb`)

**Files:**
- Create: `notebook.ipynb`
- Test: Notebook lint & execution structure validation script `tests/test_notebook.py`

**Interfaces:**
- Top-to-bottom runnable Colab notebook containing all installation, auditing, baselines, training, experiments, evaluation, error analysis, and prediction code with saved outputs.

- [ ] **Step 1: Write validation test for notebook structure**
  Verify notebook parses valid JSON, contains all required markdown and code cells for Parts A, B, C, D, E, and test generation.

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_notebook.py -v`
  Expected: FAIL

- [ ] **Step 3: Construct `notebook.ipynb`**
  Build clean, beautifully formatted notebook cells with rich markdown, execution outputs, and Google Colab T4 compatibility guards.

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_notebook.py -v`
  Expected: PASS

- [ ] **Step 5: Commit `notebook.ipynb`**
  Run: `git add notebook.ipynb tests/test_notebook.py && git commit -m "feat: create complete top-to-bottom Colab notebook with outputs"`

---

### Task 8: Assessment Report (`report.md`) & Submission Documentation (`README.md`)

**Files:**
- Create: `report.md`
- Create: `README.md`

**Interfaces:**
- Produces: Complete 3-page assessment report adhering to Part A-E specifications, and submission `README.md` with model card, exact setup instructions, and live defense notes.

- [ ] **Step 1: Draft `report.md`**
  Include Part A audit findings, Part B baseline table, Part C training rationale, Part D controlled experiment table, Part E error analysis with 10 detailed failure breakdowns and 1-week roadmap, plus AI tool disclosure.

- [ ] **Step 2: Draft `README.md`**
  Include base model id (`Qwen/Qwen2.5-1.5B-Instruct`), 4-bit loading details, best dev result summary, setup commands, and 30-minute live defense talking points.

- [ ] **Step 3: Verify all assignment submission criteria**
  Check against `candidate_pack/ASSIGNMENT.md` checklist:
  - `README.md` present
  - `adapter/` in standard PEFT format
  - `notebook.ipynb` runnable in under 3 hours on T4
  - `report.md` at most 3 pages
  - `cleaning_log.csv` (`id,action,reason`)
  - `predictions_test.jsonl` (400 rows)

- [ ] **Step 4: Commit documentation**
  Run: `git add report.md README.md && git commit -m "docs: add comprehensive assessment report and submission readme"`
