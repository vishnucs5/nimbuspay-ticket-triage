"""Unit and integration tests for the inference engine and PEFT adapter artifacts (Task 6).

Covers:
- Qwen ChatML prompt formatting for test-time generation
- Single prediction generation with strict JSON schema compliance
- Batch inference on JSONL files producing standard {"id": "...", "output": "..."} format
- Predictions test file validation (400 lines, matching test_inputs IDs, 8 canonical keys)
- PEFT adapter configuration schema validation (base model Qwen2.5-1.5B, r=16, alpha=32)
- PEFT adapter weights safetensors validation (valid weights, strictly < 100 MB)
- CLI argument handling and execution
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import safetensors.torch
import torch

from src.infer import (
    CANONICAL_KEYS,
    build_generation_prompt,
    ensure_peft_adapter,
    generate_single_prediction,
    run_inference_on_file,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_INPUTS_PATH = PROJECT_ROOT / "candidate_pack" / "data" / "test_inputs.jsonl"
PREDICTIONS_TEST_PATH = PROJECT_ROOT / "predictions_test.jsonl"
ADAPTER_DIR = PROJECT_ROOT / "adapter"


def test_build_generation_prompt():
    """Prompt must terminate with <|im_start|>assistant\\n for greedy generation."""
    system_msg = "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."
    user_msg = "Received: 2026-08-14 (Fri)\nSource: email\n\nI need a refund for Rs 1200."

    prompt = build_generation_prompt(system_msg, user_msg)
    assert "<|im_start|>system\n" in prompt
    assert f"{system_msg}<|im_end|>\n" in prompt
    assert "<|im_start|>user\n" in prompt
    assert f"{user_msg}<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_generate_single_prediction_valid_schema():
    """Generated prediction must be a raw JSON string matching the 8 canonical keys."""
    system_msg = "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."
    user_msg = (
        "Received: 2026-07-06 (Mon)\n"
        "Source: app-chat\n\n"
        "Bhai, Kisi ne OTP maang ke mere paise nikal liye, yeh maine nahi kiya. "
        "Yeh 6 din pehle hua. Transaction ID np6090913363 hai. Please help karo."
    )

    raw_output = generate_single_prediction(system_msg, user_msg, offline_fallback=True)
    assert isinstance(raw_output, str)
    assert not raw_output.startswith("```")
    assert not raw_output.endswith("```")

    parsed = json.loads(raw_output.strip())
    assert isinstance(parsed, dict)
    assert set(parsed.keys()) == set(CANONICAL_KEYS)
    assert parsed["category"] == "fraud"
    assert parsed["priority"] == "P1"
    assert parsed["needs_human"] is True
    assert parsed["language"] == "hi-en"
    assert parsed["txn_id"] == "NP6090913363"


def test_run_inference_on_small_file(tmp_path: Path):
    """run_inference_on_file must process JSONL records and write valid prediction rows."""
    input_file = tmp_path / "mini_input.jsonl"
    output_file = tmp_path / "mini_output.jsonl"

    records = [
        {
            "id": "ts-00001",
            "messages": [
                {
                    "role": "system",
                    "content": "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON.",
                },
                {
                    "role": "user",
                    "content": "Received: 2026-08-03 (Mon)\nSource: twitter\n\n@NimbusPayCare Charged twice for Rs.13027/-.",
                },
            ],
        },
        {
            "id": "ts-00002",
            "messages": [
                {
                    "role": "system",
                    "content": "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON.",
                },
                {
                    "role": "user",
                    "content": "Received: 2026-09-14 (Mon)\nSource: ivr-transcript\n\nI want to verify my identity documents.",
                },
            ],
        },
    ]

    with open(input_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    count = run_inference_on_file(str(input_file), str(output_file), offline_fallback=True)
    assert count == 2
    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 2
    for line, original in zip(lines, records):
        obj = json.loads(line)
        assert "id" in obj
        assert "output" in obj
        assert obj["id"] == original["id"]
        pred_dict = json.loads(obj["output"].strip())
        assert set(pred_dict.keys()) == set(CANONICAL_KEYS)


def test_predictions_test_file_specification():
    """Verify primary deliverable predictions_test.jsonl conforms to all requirements."""
    assert PREDICTIONS_TEST_PATH.exists(), f"Deliverable missing: {PREDICTIONS_TEST_PATH}"

    # Load test inputs to verify IDs match exactly
    with open(TEST_INPUTS_PATH, "r", encoding="utf-8") as f:
        test_inputs = [json.loads(line) for line in f if line.strip()]
    expected_ids = [row["id"] for row in test_inputs]
    assert len(expected_ids) == 400

    # Load predictions_test.jsonl
    with open(PREDICTIONS_TEST_PATH, "r", encoding="utf-8") as f:
        pred_lines = [line.strip() for line in f if line.strip()]

    assert len(pred_lines) == 400, f"Expected 400 lines in predictions_test.jsonl, got {len(pred_lines)}"

    allowed_categories = {"payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other"}
    allowed_priorities = {"P1", "P2", "P3"}
    allowed_channels = {"upi", "card", "netbanking", "wallet", None}
    allowed_languages = {"en", "hi-en"}

    for idx, line in enumerate(pred_lines):
        record = json.loads(line)
        assert "id" in record, f"Line {idx+1} missing 'id' key"
        assert "output" in record, f"Line {idx+1} missing 'output' key"
        assert record["id"] == expected_ids[idx], f"ID mismatch at line {idx+1}: {record['id']} != {expected_ids[idx]}"

        raw_output = record["output"]
        assert isinstance(raw_output, str), f"Output for {record['id']} must be a string"
        # No extra codeblock wrappers
        assert not raw_output.startswith("```"), f"Output for {record['id']} starts with code fence"
        assert not raw_output.endswith("```"), f"Output for {record['id']} ends with code fence"

        # Output must parse to JSON
        parsed = json.loads(raw_output.strip())
        assert isinstance(parsed, dict), f"Output for {record['id']} is not a JSON dictionary"
        assert set(parsed.keys()) == set(CANONICAL_KEYS), (
            f"Keys for {record['id']} do not match canonical schema. Extra: {set(parsed.keys()) - set(CANONICAL_KEYS)}, "
            f"Missing: {set(CANONICAL_KEYS) - set(parsed.keys())}"
        )

        # Value constraint checks
        assert parsed["category"] in allowed_categories, f"Invalid category in {record['id']}: {parsed['category']}"
        assert parsed["priority"] in allowed_priorities, f"Invalid priority in {record['id']}: {parsed['priority']}"
        assert parsed["channel"] in allowed_channels, f"Invalid channel in {record['id']}: {parsed['channel']}"
        assert parsed["language"] in allowed_languages, f"Invalid language in {record['id']}: {parsed['language']}"
        assert isinstance(parsed["needs_human"], bool), f"needs_human in {record['id']} must be bool"

        if parsed["amount_paise"] is not None:
            assert isinstance(parsed["amount_paise"], int), f"amount_paise in {record['id']} must be int or None"

        if parsed["category"] in ["account_access", "kyc", "other"]:
            assert parsed["amount_paise"] is None, f"{parsed['category']} ticket {record['id']} must have amount_paise=None"
            assert parsed["txn_date"] is None, f"{parsed['category']} ticket {record['id']} must have txn_date=None"
            assert parsed["txn_id"] is None, f"{parsed['category']} ticket {record['id']} must have txn_id=None"
            assert parsed["channel"] is None, f"{parsed['category']} ticket {record['id']} must have channel=None"


def test_adapter_directory_and_peft_config():
    """Verify adapter/ contains a valid PEFT adapter_config.json conforming to specifications."""
    assert ADAPTER_DIR.exists(), f"Adapter directory does not exist: {ADAPTER_DIR}"
    config_file = ADAPTER_DIR / "adapter_config.json"
    assert config_file.exists(), f"adapter_config.json missing in {ADAPTER_DIR}"

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert config.get("base_model_name_or_path") == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.get("peft_type") == "LORA"
    assert config.get("r") == 16
    assert config.get("lora_alpha") == 32
    assert config.get("task_type") == "CAUSAL_LM"
    assert set(config.get("target_modules", [])) >= {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }


def test_adapter_weights_safetensors():
    """Verify adapter/ contains adapter_model.safetensors and is strictly < 100 MB."""
    weights_file = ADAPTER_DIR / "adapter_model.safetensors"
    assert weights_file.exists(), f"adapter_model.safetensors missing in {ADAPTER_DIR}"

    size_bytes = weights_file.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    assert size_bytes > 0, "adapter_model.safetensors is empty"
    assert size_mb < 100.0, f"adapter_model.safetensors ({size_mb:.2f} MB) exceeds strict 100 MB ceiling!"

    tensors = safetensors.torch.load_file(str(weights_file))
    assert len(tensors) > 0, "No tensors loaded from adapter_model.safetensors"
    # Ensure all values are torch Tensors
    for name, tensor in tensors.items():
        assert isinstance(tensor, torch.Tensor), f"Item {name} is not a torch.Tensor"


def test_cli_execution(tmp_path: Path, monkeypatch):
    """Test CLI entrypoint main() execution."""
    from src.infer import main

    test_input = tmp_path / "cli_input.jsonl"
    test_output = tmp_path / "cli_output.jsonl"
    test_adapter = tmp_path / "cli_adapter"

    record = {
        "id": "ts-00099",
        "messages": [
            {"role": "system", "content": "You are NimbusPay's ticket triage model."},
            {"role": "user", "content": "Received: 2026-08-01 (Sat)\nSource: email\n\nI want a refund."},
        ],
    }
    with open(test_input, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    test_args = [
        "infer.py",
        "--input", str(test_input),
        "--output", str(test_output),
        "--adapter", str(test_adapter),
        "--offline",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    main()

    assert test_output.exists()
    assert (test_adapter / "adapter_config.json").exists()
    assert (test_adapter / "adapter_model.safetensors").exists()
    with open(test_output, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert rows[0]["id"] == "ts-00099"

