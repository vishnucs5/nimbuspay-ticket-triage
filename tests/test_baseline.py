"""Unit tests for baseline evaluators (Zero-Shot & Rules-in-Prompt)."""
import json
import os
import sys
from pathlib import Path
import pytest

# Ensure candidate_pack is importable
CANDIDATE_PACK_DIR = Path(__file__).resolve().parent.parent / "candidate_pack"
if str(CANDIDATE_PACK_DIR) not in sys.path:
    sys.path.insert(0, str(CANDIDATE_PACK_DIR))

from src.baseline import (
    format_zero_shot_messages,
    format_rules_in_prompt_messages,
    generate_zero_shot_prompt,
    generate_rules_prompt,
    evaluate_baseline_predictions,
    run_baseline_evaluation,
    simulate_zero_shot_output,
    simulate_rules_output,
)


@pytest.fixture
def sample_ticket():
    return (
        "Received: 2026-08-28 (Fri)\n"
        "Source: app-chat\n\n"
        "Good morning, What needs to be done to activate and use a card? "
        "My earlier ticket number is #54047. Please look into it."
    )


@pytest.fixture
def sample_gold_row():
    return {
        "id": "dv-00001",
        "messages": [
            {
                "role": "system",
                "content": "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."
            },
            {
                "role": "user",
                "content": "Received: 2026-08-28 (Fri)\nSource: app-chat\n\nGood morning, What needs to be done to activate and use a card? My earlier ticket number is #54047. Please look into it."
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "category": "other",
                    "priority": "P3",
                    "amount_paise": None,
                    "txn_date": None,
                    "txn_id": None,
                    "channel": None,
                    "language": "en",
                    "needs_human": False
                })
            }
        ]
    }


def test_format_zero_shot_messages(sample_ticket):
    messages = format_zero_shot_messages(sample_ticket)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == sample_ticket

    # Verify alias
    alias_messages = generate_zero_shot_prompt(sample_ticket)
    assert alias_messages == messages


def test_format_rules_in_prompt_messages(sample_ticket):
    schema_path = "candidate_pack/SCHEMA.md"
    messages = format_rules_in_prompt_messages(sample_ticket, schema_path=schema_path)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "You are NimbusPay's ticket triage model." in messages[0]["content"]
    assert "NimbusPay Triage Labelling Spec" in messages[0]["content"]
    assert "amount_paise" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == sample_ticket

    # Verify passing raw string as rules text
    custom_rules = "Custom rules text: output category and priority only."
    custom_messages = format_rules_in_prompt_messages(sample_ticket, schema_path=custom_rules)
    assert "Custom rules text" in custom_messages[0]["content"]

    # Verify alias
    alias_messages = generate_rules_prompt(sample_ticket, schema_path=schema_path)
    assert alias_messages == messages


def test_evaluate_baseline_predictions_exact_match(sample_gold_row):
    pred_rows = [{
        "id": "dv-00001",
        "output": sample_gold_row["messages"][-1]["content"]
    }]
    results = evaluate_baseline_predictions(pred_rows, [sample_gold_row])
    assert "all" in results
    assert results["all"]["n"] == 1
    assert results["all"]["json_valid"] == 1.0
    assert results["all"]["exact_match"] == 1.0
    assert results["all"]["mean_field_acc"] == 1.0


def test_evaluate_baseline_predictions_invalid_json(sample_gold_row):
    pred_rows = [{
        "id": "dv-00001",
        "output": "Not a JSON object"
    }]
    results = evaluate_baseline_predictions(pred_rows, [sample_gold_row])
    assert "all" in results
    assert results["all"]["json_valid"] == 0.0
    assert results["all"]["exact_match"] == 0.0
    assert results["all"]["mean_field_acc"] == 0.0


def test_simulate_zero_shot_output(sample_ticket):
    output = simulate_zero_shot_output(sample_ticket, seed=42)
    assert isinstance(output, str)
    # Zero-shot without fine-tuning doesn't follow NimbusPay 8-key schema
    try:
        parsed = json.loads(output)
        # If valid JSON, should not match all 8 NimbusPay keys
        assert set(parsed.keys()) != {
            "category", "priority", "amount_paise", "txn_date",
            "txn_id", "channel", "language", "needs_human"
        }
    except json.JSONDecodeError:
        # Markdown fences or non-JSON output is also a standard failure mode
        pass


def test_simulate_rules_output(sample_ticket):
    output = simulate_rules_output(sample_ticket, seed=42)
    assert isinstance(output, str)
    parsed = json.loads(output.strip())
    expected_keys = {
        "category", "priority", "amount_paise", "txn_date",
        "txn_id", "channel", "language", "needs_human"
    }
    assert set(parsed.keys()) == expected_keys
    assert parsed["category"] == "other"
    assert parsed["priority"] == "P3"
    assert parsed["language"] in ["en", "hi-en"]


def test_run_baseline_evaluation_tmp(tmp_path):
    # Create mini dev file
    mini_dev = tmp_path / "mini_dev.jsonl"
    gold_record = {
        "id": "dv-00001",
        "messages": [
            {"role": "system", "content": "System prompt"},
            {
                "role": "user",
                "content": "Received: 2026-08-28 (Fri)\nSource: app-chat\n\nActivate card please."
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "category": "other", "priority": "P3", "amount_paise": None,
                    "txn_date": None, "txn_id": None, "channel": None,
                    "language": "en", "needs_human": False
                })
            }
        ]
    }
    mini_dev.write_text(json.dumps(gold_record) + "\n", encoding="utf-8")

    out_zero = tmp_path / "zero.jsonl"
    out_rules = tmp_path / "rules.jsonl"

    res = run_baseline_evaluation(
        dev_path=str(mini_dev),
        schema_path="candidate_pack/SCHEMA.md",
        output_zeroshot_path=str(out_zero),
        output_rules_path=str(out_rules),
        force_simulation=True,
    )

    assert "zeroshot" in res
    assert "rules_in_prompt" in res
    assert out_zero.exists()
    assert out_rules.exists()

    # Verify predictions file lines
    with open(out_zero, encoding="utf-8") as f:
        z_lines = [json.loads(line) for line in f]
    assert len(z_lines) == 1
    assert "id" in z_lines[0]
    assert "output" in z_lines[0]
