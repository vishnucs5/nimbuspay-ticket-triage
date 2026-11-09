# NimbusPay Support Ticket Triage LLM Fine-Tuning
**Assessment Report & Technical Implementation Brief**  
*Candidate: Machine Learning Intern – LLM Fine-Tuning, VIMA3YA*  
*Hardware Platform: Google Colab Free Tier (1x NVIDIA Tesla T4 16GB VRAM, 12GB RAM)*  
*Artifacts: `adapter/` (18.5M params PEFT), `notebook.ipynb` (verified outputs), `cleaning_log.csv` (327 records)*  

---

## 1. Data Audit Findings & Cleaning Rationale (Part A)

Production training data routinely suffers from pipeline glitches and label degradation. Auditing `data/train.jsonl` (1,445 tickets) against `SCHEMA.md` revealed **8 distinct corruption failure modes** affecting 327 rows. Every intervention is recorded in `cleaning_log.csv` (`id,action,reason`).

```
Initial Training Corpus: 1,445 rows
├── Dropped Records: 115 rows (8.0%)
│   ├── Malformed / Truncated JSON Syntax: 25 rows
│   ├── Train-Dev Contamination (Leakage): 30 rows
│   └── Exact Duplicate Tickets: 60 rows
├── Repaired Records: 212 rows (14.7%)
│   ├── Schema Key Renaming ('prio' -> 'priority'): 70 rows
│   ├── Legacy Priority Value Remapping ('high'/'medium'/'low' -> 'P1'/'P2'/'P3'): 70 rows
│   ├── Date Normalization (DD/MM/YYYY -> YYYY-MM-DD): 30 rows
│   ├── Transaction ID Normalization (regex -> NP+10 digits): 20 rows
│   └── Currency Denomination Conversions (Rupees -> Integer Paise): 22 rows
└── Clean Production Corpus: 1,330 rows (100% schema conformant)
```

### Detection Methodology & Remediation Breakdown
1. **Truncated/Malformed JSON (25 dropped):** Parser rejection (`json.loads`) on severed strings. Dropped because ground truth cannot be reliably reconstructed.
2. **Train-Dev Leakage (30 dropped):** SHA-256 hash collision on user ticket text between train and dev. **Justification:** Retaining identical dev prompts causes evaluation contamination, artificially inflating dev scores and masking true generalization deficits.
3. **Duplicate Training Tickets (60 dropped):** Duplicate ticket hashing within train. Kept the first occurrence to avoid overfitting and gradient bias.
4. **Legacy Schema Key (70 fixed):** Renamed legacy key `"prio"` to `"priority"`.
5. **Legacy Priority Values (70 fixed):** Mapped legacy string values: `high -> P1`, `medium -> P2`, `low -> P3`.
6. **Date Format Normalization (30 fixed):** Regex converted `DD/MM/YYYY` to ISO-8601 `YYYY-MM-DD`.
7. **Transaction ID Canonicalization (20 fixed):** Converted lowercase/spaced IDs (`np 12345 67890`) to uppercase `NP` + 10 digits (`NP1234567890`).
8. **Rupee-to-Paise Currency Normalization (22 fixed):** Converted fractional rupee floats (e.g. `1299.0`) to integer paise (`129900`).
9. **Null Fields Rule Enforcement:** Enforced schema rule 3: `amount_paise`, `txn_date`, `txn_id`, and `channel` are `null` for non-transaction tickets (`account_access`, `kyc`, `other`).

---

## 2. Baselines & Benchmark Measurements (Part B)

Evaluated on clean `dev.jsonl` (200 tickets) using official `score.py`:

| Evaluation Setup | Valid JSON | Exact Match | Mean Field Acc | Latency / Ticket | Prompt Tokens |
|---|---|---|---|---|---|
| **Zero-Shot Base (`Qwen2.5-1.5B-Instruct`)** | 60.5% | **0.0%** | 5.8% | 142 ms | ~45 tokens |
| **Rules-in-Prompt Reference Baseline** | 97.0% | **60.5%** | 89.2% | 680 ms | ~1,250 tokens |
| **Our Fine-Tuned Model (Part C/D)** | **100.0%** | **80.0%** | **95.8%** | **148 ms** | **~45 tokens** |

### Why Zero-Shot Fails & Fine-Tuning Closes the Gap
- **Zero-Shot Breakdown (0.0% EM):** The un-fine-tuned model lacks knowledge of NimbusPay's private 8-key schema. It hallucinates arbitrary keys (`issue_type`), wraps outputs in markdown backticks, and fails on currency units. Only `language` scored non-zero (46.0%).
- **Rules-in-Prompt Limitations (60.5% EM):** Injecting the full `SCHEMA.md` into the prompt achieves 60.5% EM but incurs severe overhead: **27.7x** token consumption (1,250 vs 45 tokens) and **4.6x** higher latency (680 ms vs 148 ms). Attention dilution on long prompts causes 39.5% errors on nested conditional rules.
- **The Fine-Tuning Advantage (+19.5% EM over rules-in-prompt):** Fine-tuning embeds business logic directly into model weights, achieving 80.0% EM with minimal 45-token prompts, satisfying all production SLA requirements.

---

## 3. Fine-Tuning Pipeline Decisions (Part C)

- **Model Selection:** `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters $\le 4\text{B}$ ceiling). Selected for state-of-the-art reasoning, code-switched Hinglish (`hi-en`) tokenization support, and low memory footprint (3.1 GB weights).
- **Chat Template:** Standard ChatML syntax (`<|im_start|>system...<|im_start|>user...<|im_start|>assistant...<|im_end|>`).
- **Sequence Length Evidence:** Token length analysis across 1,330 train and 200 dev tickets shows median 172 tokens, 99th percentile 442 tokens, max 488 tokens. `max_seq_length = 512` guarantees **100.0% coverage with zero truncation**.
- **Analytical LoRA Parameter Budget ($\le 50\text{M}$):**
  - Architecture: 28 decoder layers, hidden dimension $d = 1536$, intermediate size $d_{ffn} = 8960$, 2 KV heads ($d_{kv} = 256$).
  - Target modules (all 7 linear projections): `q, k, v, o, gate, up, down_proj`.
  - Trainable parameters per layer for rank $r$:
    $$\Delta W_{\text{layer}} = [3072 + 1792 + 1792 + 3072 + 10496 + 10496 + 10496] \cdot r = 41,216 \cdot r$$
  - For $r = 16$: $28 \times 41,216 \times 16 = \mathbf{18,464,768} \text{ parameters } (18.46\text{M}) \le 50\text{M}$ limit (1.18% of base model).
- **Completion-Only Loss Masking:** Prompt tokens are masked with `-100` up to `<|im_start|>assistant\n`. Gradients update weights strictly on assistant JSON tokens, focusing model capacity entirely on structured serialization.

---

## 4. Controlled Experiments Matrix & Analysis (Part D)

Four controlled single-variable ablations were benchmarked on `dev.jsonl` (200 tickets):

| What Changed | Dev Exact Match | Dev Mean Field Acc | Peak GPU Memory | Training Minutes |
|---|---|---|---|---|
| **Cleaned data, 16-bit LoRA, $r=16$, completion masking (Reference)** | **80.0%** | **95.8%** | **8.8 GB** | **18.5 min** |
| Raw training data (uncleaned noise & corruptions) | 51.0% (-29.0%) | 83.1% (-12.7%) | 8.9 GB | 20.2 min |
| 4-bit QLoRA (NF4 quantization) | 77.5% (-2.5%) | 96.1% (+0.3%) | **5.4 GB** (-38.6%) | 24.5 min (+32%) |
| LoRA rank $r=8$ (9.2M trainable params) | 75.0% (-5.0%) | 95.6% (-0.2%) | 8.2 GB | 16.8 min |
| Full-sequence cross-entropy loss (unmasked prompt) | 69.5% (-10.5%) | 89.1% (-6.7%) | 8.8 GB | 18.8 min |

### Controlled Ablation Findings
1. **Data Quality (+29.0% EM gain):** Training on raw data severely degrades performance (51.0% EM). The model memorizes corrupted schema keys (`"prio"`) and fails JSON syntax on 5.0% of tickets. Data cleaning is the single highest-leverage factor.
2. **16-bit LoRA vs 4-bit QLoRA (-2.5% EM on QLoRA):** QLoRA reduces peak VRAM from 8.8 GB to 5.4 GB (-38.6%), making it feasible for $\le 8\text{GB}$ GPUs. However, double-dequantization quantization noise slightly degrades subtle boundary logic (77.5% vs 80.0% EM), and run time increases by 32% due to dequantization overhead.
3. **Rank $r=8$ vs $r=16$ (-5.0% EM on $r=8$):** Halving parameters to 9.2M ($r=8$) reduces expressive capacity needed to simultaneously retain rupee-to-paise conversion and hierarchical priority precedence.
4. **Completion-Only Loss Masking (+10.5% EM gain):** Full-sequence loss wastes capacity modeling predictable user preambles, dropping exact match to 69.5% and causing markdown syntax hallucinations.

---

## 5. Diagnostic Error Analysis & 1-Week Roadmap (Part E)

### Field & Cascading Dynamics
Auditing all 40 dev failures (80.0% EM, 160/200 correct) on the best model:
- **Flawless Fields (100%):** `amount_paise` (0 errors), `txn_id` (0 errors), `language` (0 errors).
- **Weakest Fields:** `priority` (**86.0%**, 28 errors) and `needs_human` (**91.0%**, 18 errors) constitute >70% of all field mismatches.
- **Cascading Failure:** Rule 1 links $P1 \implies needs\_human = true$. When high-value refund tickets (e.g. ₹2.94L in `dv-00167`) are misclassified as P2, `needs_human` cascades to false.

### Slice Diagnostics
- **Category Slice:** `refund` (67.3% EM) and `account_access` (68.4% EM) are weakest due to boundary ambiguity and default rule lapses. `fraud` achieved the highest accuracy (**93.3% EM**) due to absolute P1 dominance.
- **Source Slice:** Email (84.1% EM) and App-Chat (80.2% EM) outperform IVR transcripts (76.7% EM, oral disfluencies) and Twitter (77.5% EM, truncated text). Multi-turn email reply chains (`> ...`) caused historical date extraction errors.
- **Language Slice:** English achieved **81.5% EM** vs **75.5% EM** for Hinglish (`hi-en`), driven by Hindi repeat contact cues (*"Teesri baar call kiya"*).

### Representative Failure Cases

| ID | Cat | Gold vs. Pred Mismatch | Root Cause Diagnostic |
|---|---|---|---|
| `dv-00167` | `refund` | Gold: `P1, needs_human:true` <br> Pred: `P2, needs_human:false` | **High-Value Threshold Cascade:** Disputed ₹2,94,830 missed $\ge ₹50,000$ P1 rule, cascading to `needs_human: false`. |
| `dv-00045` | `account_access` | Gold: `P2, needs_human:false` <br> Pred: `P3, needs_human:false` | **Default Priority Lapse:** Schema Rule 2 mandates P2 for `account_access`. Model defaulted to P3. |
| `dv-00088` | `payment_failed` | Gold: `P2, needs_human:true` <br> Pred: `P2, needs_human:false` | **Hinglish Repeat Contact Miss:** Customer wrote *"Baar baar transaction fail ho raha hai"*; missed repeat cue. |
| `dv-00112` | `refund` | Gold: `txn_date: 2026-08-10` <br> Pred: `txn_date: 2026-08-01` | **Quoted Email Distraction:** Extracted date from quoted reply header (`> Sent Aug 1`) instead of current message. |
| `dv-00134` | `kyc` | Gold: `channel: null` <br> Pred: `channel: "upi"` | **Null Field Violation:** Ticket mentioned *"UPI KYC"*; model hallucinated channel on non-transaction ticket. |
| `dv-00159` | `payment_failed` | Gold: `P1, needs_human:true` <br> Pred: `P2, needs_human:false` | **Legal Threat Miss:** Customer mentioned *"banking ombudsman"*; model failed to trigger P1 escalation. |
| `dv-00176` | `offers` | Gold: `category: offers` <br> Pred: `category: payment_failed` | **Category Ambiguity:** Uncredited promo cash back misclassified as payment failure. |
| `dv-00189` | `other` | Gold: `channel: null` <br> Pred: `channel: "card"` | **Card Inquiry Distraction:** Customer asked how to order physical debit card; model set `channel: card`. |
| `dv-00192` | `refund` | Gold: `P2, needs_human:true` <br> Pred: `P2, needs_human:false` | **Repeat Contact Cue Miss:** Ticket stated *"third time reaching out"*; missed repeat contact indicator. |
| `dv-00198` | `fraud` | Gold: `txn_date: 2026-06-20` <br> Pred: `txn_date: 2026-06-19` | **Relative Date Arithmetic:** Calendar off-by-one parsing *"Day before yesterday"* against `Received` date. |

### Prioritized 1-Week Roadmap
1. **Synthetic Boundary Augmentation (Days 1–2):** Generate 400 hard-negative synthetic tickets near thresholds (₹4,999 vs ₹5,001; ₹49,999 vs ₹50,001), subtle legal terms (ombudsman, consumer court), and Hinglish repeat phrases.
2. **Focal Token Loss on Priority/Escalation (Day 3):** Weight cross-entropy loss higher for `priority` and `needs_human` tokens to counteract label imbalance and prevent cascade failures.
3. **Quoted Email Rejection via DPO (Day 4):** Train DPO pairs where negative samples extract details from `> ` quoted email lines, teaching the model to ignore quoted email history.
4. **Calendar Engine Grounding (Day 5):** Implement lightweight symbolic date resolution for relative expressions (*"parso"*, *"yesterday"*) grounded against the `Received:` header.
5. **Two-Stage Modular Decomposition (Days 6–7):** If single-model accuracy plateaus, decouple into Stage 1 (slot extraction) and Stage 2 (deterministic business rule engine).

---

## 6. AI Tools & Assistant Usage Disclosure

- **Scaffolding & Boilerplate:** Anthropic Claude assisted in regex drafting for `src/audit.py`, structuring unit tests in `tests/`, and compiling Markdown tables.
- **Candidate Ownership:** All data auditing discoveries, hyperparameter choices ($r=16, \alpha=32$, `max_seq_length=512`), LoRA parameter calculations, ablation designs, diagnostic error root-cause analyses, and roadmap proposals were developed, verified, and defended directly by the candidate. All metrics reflect genuine Colab T4 execution logs.
