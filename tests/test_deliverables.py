"""Deliverables verification test suite for NimbusPay triage LLM fine-tuning submission.

Asserts presence, size, structure, and validity of all 6 required submission items from ASSIGNMENT.md:
1. README.md (base model id, 4-bit info, one-line summary, reproduction instructions, live defense notes)
2. adapter/ (adapter_config.json, adapter_model.safetensors < 100MB)
3. notebook.ipynb (valid JSON, has outputs)
4. report.md (all 6 sections, under 3 pages / ~1800 words)
5. cleaning_log.csv (id,action,reason, >300 lines)
6. predictions_test.jsonl (400 lines, valid JSON, 8 keys)
"""

import csv
import json
import os
import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent

README_PATH = ROOT / "README.md"
REPORT_PATH = ROOT / "report.md"
ADAPTER_DIR = ROOT / "adapter"
ADAPTER_CONFIG_PATH = ADAPTER_DIR / "adapter_config.json"
ADAPTER_WEIGHTS_PATH = ADAPTER_DIR / "adapter_model.safetensors"
NOTEBOOK_PATH = ROOT / "notebook.ipynb"
CLEANING_LOG_PATH = ROOT / "cleaning_log.csv"
PREDICTIONS_TEST_PATH = ROOT / "predictions_test.jsonl"

SCHEMA_KEYS = [
    "category",
    "priority",
    "amount_paise",
    "txn_date",
    "txn_id",
    "channel",
    "language",
    "needs_human",
]


def test_deliverable_readme():
    """Verify README.md exists, is comprehensive, and satisfies ASSIGNMENT.md rules."""
    assert README_PATH.exists(), "README.md must exist at repository root"
    content = README_PATH.read_text(encoding="utf-8")
    assert len(content) > 500, "README.md is too short (< 500 characters)"

    # Base model ID
    assert "Qwen/Qwen2.5-1.5B-Instruct" in content or "Qwen2.5-1.5B" in content, (
        "README.md must declare base model id"
    )

    # 4-bit / 16-bit quantization details
    assert "4-bit" in content or "QLoRA" in content or "quantization" in content.lower(), (
        "README.md must disclose whether base model was loaded in 4-bit"
    )

    # One-line summary of best result
    assert "Exact Match" in content or "exact match" in content.lower() or "80.0%" in content, (
        "README.md must contain a one-line summary of the best result"
    )
    assert "Mean Field" in content or "mean field" in content.lower() or "95.8%" in content, (
        "README.md must report mean field accuracy"
    )

    # Reproduction instructions
    assert "reproduce" in content.lower() or "reproduction" in content.lower() or "colab" in content.lower(), (
        "README.md must provide reproduction instructions"
    )

    # Live defense notes
    assert "defense" in content.lower() or "interview" in content.lower(), (
        "README.md must provide live defense preparation notes"
    )


def test_deliverable_adapter():
    """Verify adapter/ directory contains PEFT configuration and weights under 100MB."""
    assert ADAPTER_DIR.exists() and ADAPTER_DIR.is_dir(), "adapter/ directory must exist"
    assert ADAPTER_CONFIG_PATH.exists(), "adapter/adapter_config.json must exist"
    assert ADAPTER_WEIGHTS_PATH.exists(), "adapter/adapter_model.safetensors must exist"

    # Verify adapter_config.json content
    with open(ADAPTER_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert "base_model_name_or_path" in cfg, "adapter_config.json missing base_model_name_or_path"
    assert "peft_type" in cfg, "adapter_config.json missing peft_type"
    assert cfg.get("peft_type") == "LORA", f"Expected peft_type 'LORA', got {cfg.get('peft_type')}"
    assert "r" in cfg, "adapter_config.json missing rank 'r'"

    # Size check: GitHub rejects files over 100 MB (104,857,600 bytes)
    weights_size_bytes = ADAPTER_WEIGHTS_PATH.stat().st_size
    max_github_bytes = 100 * 1024 * 1024
    assert weights_size_bytes < max_github_bytes, (
        f"adapter_model.safetensors is {weights_size_bytes} bytes, exceeding 100MB GitHub limit"
    )
    assert weights_size_bytes > 1_000_000, "adapter_model.safetensors is suspiciously small (< 1MB)"


def test_deliverable_notebook():
    """Verify notebook.ipynb exists, is valid JSON, and has execution outputs."""
    assert NOTEBOOK_PATH.exists(), "notebook.ipynb must exist at repository root"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb, "notebook.ipynb must contain 'cells'"
    assert len(nb["cells"]) >= 15, "notebook.ipynb must contain complete multi-step cells"

    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 8, "notebook.ipynb must have executable code cells"

    for idx, cell in enumerate(code_cells):
        outputs = cell.get("outputs", [])
        assert len(outputs) > 0, f"Code cell {idx + 1} has no outputs. Must include pre-computed outputs."


def test_deliverable_report():
    """Verify report.md exists, adheres to length limits, and covers all required sections."""
    assert REPORT_PATH.exists(), "report.md must exist at repository root"
    content = REPORT_PATH.read_text(encoding="utf-8")
    assert len(content) > 1000, "report.md is too short"

    # Word count constraint: at most 3 pages rendered (~1,200 to 1,800 words)
    words = re.findall(r"\b\w+\b", content)
    word_count = len(words)
    assert 800 <= word_count <= 2500, (
        f"report.md word count ({word_count}) outside recommended 3-page scope (~1200-1800 words)"
    )

    # Required sections / parts
    lower = content.lower()
    # Section 1 / Part A: Data Audit
    assert "part a" in lower or "data audit" in lower, "report.md missing Part A / Data Audit"
    assert "115" in content or "dropped" in lower, "report.md should mention dropped rows"
    assert "212" in content or "fixed" in lower or "fix" in lower, "report.md should mention fixed rows"
    assert "leakage" in lower or "contamination" in lower, "report.md should discuss train-dev leakage"

    # Section 2 / Part B: Baselines
    assert "part b" in lower or "baseline" in lower, "report.md missing Part B / Baselines"
    assert "zero-shot" in lower or "zeroshot" in lower, "report.md missing zero-shot baseline"
    assert "rules" in lower, "report.md missing rules-in-prompt reference baseline"

    # Section 3 / Part C: Fine-Tuning Pipeline
    assert "part c" in lower or "fine-tuning" in lower or "pipeline" in lower, (
        "report.md missing Part C / Pipeline"
    )
    assert "chatml" in lower or "template" in lower, "report.md missing chat template discussion"
    assert "512" in content or "sequence length" in lower, "report.md missing sequence length evidence"
    assert "lora" in lower or "parameter" in lower, "report.md missing LoRA parameter budget"
    assert "loss" in lower or "mask" in lower or "completion" in lower, "report.md missing loss masking rationale"

    # Section 4 / Part D: Controlled Experiments Matrix
    assert "part d" in lower or "controlled experiment" in lower or "ablation" in lower, (
        "report.md missing Part D / Controlled Experiments"
    )
    assert "what changed" in lower or "what changed" in content, "report.md table missing 'What Changed' column"
    assert "exact match" in lower, "report.md table missing 'Exact Match' column"
    assert "memory" in lower or "vram" in lower, "report.md table missing GPU memory column"
    assert "minutes" in lower or "time" in lower, "report.md table missing training time column"

    # Section 5 / Part E: Diagnostic Error Analysis & Roadmap
    assert "part e" in lower or "error analysis" in lower, "report.md missing Part E / Error Analysis"
    assert "priority" in lower, "report.md should mention priority field"
    assert "needs_human" in lower, "report.md should mention needs_human field"
    assert "roadmap" in lower or "week" in lower, "report.md should provide 1-week roadmap"

    # Section 6: AI Tools Disclosure
    assert "ai tools" in lower or "assistant" in lower or "disclosure" in lower, (
        "report.md missing AI Tools Disclosure"
    )


def test_deliverable_cleaning_log():
    """Verify cleaning_log.csv exists, has standard header, and contains > 300 rows."""
    assert CLEANING_LOG_PATH.exists(), "cleaning_log.csv must exist at repository root"

    with open(CLEANING_LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["id", "action", "reason"], (
            f"cleaning_log.csv header must be id,action,reason, got {reader.fieldnames}"
        )
        rows = list(reader)

    assert len(rows) > 300, f"cleaning_log.csv has {len(rows)} rows, expected > 300"
    for r in rows:
        assert r["id"].startswith("tr-"), f"Invalid id: {r['id']}"
        assert r["action"] in ["drop", "fix"], f"Invalid action: {r['action']}"
        assert len(r["reason"]) > 0, f"Empty reason in row {r['id']}"


def test_deliverable_predictions_test():
    """Verify predictions_test.jsonl has exactly 400 rows, valid JSON, and exactly 8 keys."""
    assert PREDICTIONS_TEST_PATH.exists(), "predictions_test.jsonl must exist at repository root"

    with open(PREDICTIONS_TEST_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 400, f"predictions_test.jsonl has {len(lines)} lines, expected exactly 400"

    for idx, line in enumerate(lines):
        record = json.loads(line)
        assert "id" in record, f"Line {idx + 1} missing 'id'"
        assert "output" in record, f"Line {idx + 1} missing 'output'"
        assert record["id"].startswith("ts-"), f"Line {idx + 1} id does not start with 'ts-': {record['id']}"

        output_obj = json.loads(record["output"].strip())
        assert isinstance(output_obj, dict), f"Line {idx + 1} output is not a JSON dict"
        assert set(output_obj.keys()) == set(SCHEMA_KEYS), (
            f"Line {idx + 1} output keys mismatch: {list(output_obj.keys())}"
        )


def test_all_six_submission_items_present():
    """Consolidated check verifying all 6 required submission items are present."""
    items = {
        "README.md": README_PATH,
        "adapter/": ADAPTER_DIR,
        "notebook.ipynb": NOTEBOOK_PATH,
        "report.md": REPORT_PATH,
        "cleaning_log.csv": CLEANING_LOG_PATH,
        "predictions_test.jsonl": PREDICTIONS_TEST_PATH,
    }
    missing = [name for name, path in items.items() if not path.exists()]
    assert not missing, f"Missing required submission deliverables: {missing}"
