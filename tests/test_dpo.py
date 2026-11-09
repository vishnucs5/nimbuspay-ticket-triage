"""Unit tests for DPOPairGenerator."""
import json
from pathlib import Path
import pytest
from src.dpo import DPOPairGenerator


def test_generate_preference_pair_null_channel():
    gold = {
        "category": "kyc",
        "priority": "P2",
        "amount_paise": None,
        "txn_date": None,
        "txn_id": None,
        "channel": None,
        "language": "en",
        "needs_human": False,
    }
    pair = DPOPairGenerator.generate_preference_pair("test-01", "Ticket text", gold, "null_channel")
    assert pair["id"] == "test-01"
    chosen = json.loads(pair["chosen"])
    rejected = json.loads(pair["rejected"])
    assert chosen["channel"] is None
    assert rejected["channel"] == "upi"


def test_generate_preference_pair_under_escalated():
    gold = {
        "category": "refund",
        "priority": "P1",
        "amount_paise": 6000000,
        "txn_date": "2026-07-01",
        "txn_id": "NP123456",
        "channel": "upi",
        "language": "en",
        "needs_human": True,
    }
    pair = DPOPairGenerator.generate_preference_pair("test-02", "Ticket text", gold, "under_escalated")
    chosen = json.loads(pair["chosen"])
    rejected = json.loads(pair["rejected"])
    assert chosen["priority"] == "P1"
    assert chosen["needs_human"] is True
    assert rejected["priority"] == "P2"
    assert rejected["needs_human"] is False


def test_harvest_and_save_from_actual_clean_dataset(tmp_path):
    clean_path = Path("data/train_clean.jsonl")
    if not clean_path.exists():
        pytest.skip("train_clean.jsonl not found")

    out_file = tmp_path / "dpo_test.jsonl"
    count = DPOPairGenerator.save_preference_dataset(clean_path, out_file, max_pairs=50)
    assert count > 0
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        assert "prompt" in first_line
        assert "chosen" in first_line
        assert "rejected" in first_line
        assert "violation_type" in first_line
