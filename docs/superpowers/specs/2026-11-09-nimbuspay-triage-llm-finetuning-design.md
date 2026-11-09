# Design Document: Fine-Tuning Qwen2.5-1.5B for NimbusPay Ticket Triage

- **Author:** Machine Learning Intern Candidate, VIMA3YA
- **Target Hardware:** Google Colab Free Tier (1x NVIDIA T4 16GB GPU)
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters)
- **Date:** 2026-11-09

---

## 1. Executive Summary & Problem Scope

NimbusPay requires a lightweight, production-grade language model to read incoming customer support tickets and output a structured JSON triage record conforming strictly to the `NimbusPay Triage Labelling Spec (v2)`.
At test time, the model receives solely a minimal system prompt and the raw ticket text—without business rules or schema examples. The model must internalize the schema, business rules, relative date resolution, entity normalization, and edge-case handling purely through fine-tuning.

All training and inference run within the constraints of Google Colab's free T4 GPU tier:
- Parameter budget: $\le 4\text{B}$ total parameters, $\le 50\text{M}$ trainable LoRA parameters.
- Inference constraints: Greedy decoding (`max_new_tokens=200`), no regex or post-generation heuristic fixing, no grammar-based constrained decoding.

---

## 2. Part A: Dataset Audit & Cleaning Strategy

Analysis of `candidate_pack/data/train.jsonl` (1,445 rows) against `SCHEMA.md` and `candidate_pack/data/dev.jsonl` identified 8 critical data quality failures:

1. **Truncated / Incomplete JSON (`action: drop`, 25 rows):**
   Assistant labels abruptly truncated mid-token (e.g. `tr-00060`, `tr-00125`) due to context length limits in the upstream generation pipeline. Dropped to preserve structural JSON pre-training signals.
2. **Train-Dev Contamination (`action: drop`, 30 rows):**
   30 tickets in `train.jsonl` are byte-for-byte identical to tickets in `dev.jsonl`. Dropped strictly to respect the assignment rule *"Never train on dev or test rows"* and eliminate evaluation leakage.
3. **Exact Redundant Duplicates (`action: drop`, 60 rows):**
   Duplicate user messages within the training split. Deduplicated to prevent overfitting on overrepresented phrases.
4. **Schema Key Inconsistency (`action: fix`, 70 rows):**
   Legacy v1 labels using key `"prio"` instead of `"priority"`. Renamed to `"priority"`.
5. **Legacy Priority Encoding (`action: fix`, 70 rows):**
   V1 priority levels recorded as `"high"`, `"medium"`, `"low"`. Mapped to `"P1"`, `"P2"`, `"P3"` and validated against v2 business logic.
6. **Date Formatting (`action: fix`, 30 rows):**
   Dates formatted as `DD/MM/YYYY` instead of ISO `YYYY-MM-DD`. Converted to standard `YYYY-MM-DD`.
7. **Unnormalized Transaction IDs (`action: fix`, 20 rows):**
   Transaction IDs with spaces (`NP 81633 47892`), hyphens (`NP-9718716109`), or lowercase letters (`np9891267518`). Normalized to uppercase `NP` + 10 digits (`NP1234567890`).
8. **Rupee-Paise Unit Errors & Decimals (`action: fix`, 19 rows):**
   Rupee amounts recorded directly without $\times 100$ paise conversion (e.g. ₹267 stored as `267` instead of `26700`), and truncated decimal amounts (e.g. ₹1,68,787.25 stored as `168787` instead of `16878725`). Corrected per `SCHEMA.md`.

All actions are logged in `cleaning_log.csv` (`id,action,reason`), producing `data/train_clean.jsonl` (~1,330 high-quality rows).

---

## 3. Part B: Baseline Evaluation

Evaluated on `data/dev.jsonl` (200 rows) using `score.py`:
1. **Zero-Shot Base Model:** `Qwen2.5-1.5B-Instruct` with the standard fixed system prompt:
   `"You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."`
   Measures out-of-the-box JSON compliance and domain inference capability.
2. **Rules-in-Prompt Reference Baseline:** The same model with the entire `SCHEMA.md` rules text injected into the system prompt. Establishes the upper bound achievable through pure in-context prompting.

---

## 4. Part C: Fine-Tuning Pipeline & Architecture

### Model Selection
* **Architecture:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters).
* **Rationale:** Ungated weights (zero Colab token barriers), state-of-the-art native JSON structured generation, and superior multilingual tokenization for romanized Hindi (Hinglish) compared to English-centric models like Llama 3.2.

### Tokenization & SFT Loss Masking
* **Prompt Format:** Native ChatML format (`<|im_start|>system ... <|im_end|>\n<|im_start|>user ... <|im_end|>\n<|im_start|>assistant ... <|im_end|>`).
* **Sequence Length:** Ticket inputs $\le 180$ tokens, output JSON $\le 60$ tokens. `max_seq_length = 512` guarantees zero sequence truncation.
* **Completion-Only Masking:** Mask input prompt tokens (`DataCollatorForCompletionOnlyLM` on `<|im_start|>assistant\n`) so loss is computed exclusively on the triage JSON tokens.

### LoRA Hyperparameters
* **Target Modules:** All linear attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Rank ($r$):** 16, **Alpha ($\alpha$):** 32, **Dropout:** 0.05.
* **Trainable Parameters:** $\approx 18.4\text{M}$ (36.8% of the 50M parameter ceiling).
* **Precision:** 16-bit LoRA (bfloat16 / float16) and 4-bit QLoRA (NF4).
* **Optimizer & Schedule:** AdamW / Paged-AdamW, learning rate $2\times 10^{-4}$, cosine decay with 5% warmup, batch size 16 (batch 4 $\times$ grad accumulation 4), 3 epochs.

---

## 5. Part D: Controlled Experiments Specification

Four controlled experiments altering one variable at a time:
1. **Experiment 1 (Data Quality):** Raw `train.jsonl` vs. Cleaned `train_clean.jsonl` (16-bit LoRA, $r=16$).
2. **Experiment 2 (Quantization & Precision):** 16-bit LoRA vs. 4-bit QLoRA on cleaned data.
3. **Experiment 3 (Capacity Ablation):** LoRA rank $r=8$ ($\sim 9.2\text{M}$ params) vs. $r=16$ ($\sim 18.4\text{M}$ params).
4. **Experiment 4 (Custom Idea - Loss Masking):** Completion-only loss masking vs. Full-sequence cross-entropy loss.

Metrics reported via `score.py`: Exact Match %, Mean Field Acc %, Peak VRAM (GB), and Training Time (min).

---

## 6. Part E: Error Analysis & Slices

In-depth evaluation on `data/dev.jsonl` slices:
- Error rates by field: `category`, `priority`, `amount_paise`, `txn_date`, `txn_id`, `channel`, `language`, `needs_human`.
- Slices: Hinglish vs. English, IVR transcripts vs. App Chat, relative date resolutions, edge-case P1 legal threats.
- Analysis of at least 10 specific failure cases with root causes and actionable improvements.

---

## 7. Submission Deliverables & Directory Layout

```
.
├── candidate_pack/
│   ├── data/ (train.jsonl, dev.jsonl, test_inputs.jsonl)
│   ├── ASSIGNMENT.md, SCHEMA.md, score.py
├── data/
│   ├── train_clean.jsonl
│   └── cleaning_log.csv
├── adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── notebook.ipynb
├── report.md
├── README.md
└── predictions_test.jsonl
```
