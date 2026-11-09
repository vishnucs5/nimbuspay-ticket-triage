"""Unit tests for BoundaryAugmenter."""
import json
from pathlib import Path
import pytest
from src.augment import BoundaryAugmenter, CANONICAL_SCHEMA_KEYS


def test_generate_threshold_pair():
    pair_a, pair_b = BoundaryAugmenter.generate_threshold_pair("test-01")
    assert pair_a["id"] == "test-01-a"
    assert pair_b["id"] == "test-01-b"

    gold_a = json.loads(pair_a["messages"][-1]["content"])
    gold_b = json.loads(pair_b["messages"][-1]["content"])

    # Under threshold
    assert gold_a["amount_paise"] == 4950000
    assert gold_a["priority"] == "P2"
    assert gold_a["needs_human"] is False

    # Over threshold
    assert gold_b["amount_paise"] == 5050000
    assert gold_b["priority"] == "P1"
    assert gold_b["needs_human"] is True


def test_generate_repeat_contact_variation():
    row_hi = BoundaryAugmenter.generate_repeat_contact_variation("test-hi", language="hi-en")
    gold_hi = json.loads(row_hi["messages"][-1]["content"])
    assert gold_hi["language"] == "hi-en"
    assert gold_hi["needs_human"] is True

    row_en = BoundaryAugmenter.generate_repeat_contact_variation("test-en", language="en")
    gold_en = json.loads(row_en["messages"][-1]["content"])
    assert gold_en["language"] == "en"
    assert gold_en["needs_human"] is True


def test_generate_distractor_null_ticket():
    row_kyc = BoundaryAugmenter.generate_distractor_null_ticket("test-kyc", category="kyc")
    gold_kyc = json.loads(row_kyc["messages"][-1]["content"])
    assert gold_kyc["category"] == "kyc"
    assert gold_kyc["channel"] is None
    assert gold_kyc["amount_paise"] is None


def test_create_boundary_dataset_schema_conformity():
    dataset = BoundaryAugmenter.create_boundary_dataset(
        num_threshold_pairs=5,
        num_repeat_contacts=5,
        num_distractors=5,
    )
    assert len(dataset) == (5 * 2) + 5 + 5  # 20 samples

    for row in dataset:
        assert "id" in row
        assert "messages" in row
        assert len(row["messages"]) == 3
        gold = json.loads(row["messages"][-1]["content"])
        assert set(gold.keys()) == set(CANONICAL_SCHEMA_KEYS)


def test_save_augmented_dataset(tmp_path):
    base_file = tmp_path / "base.jsonl"
    with open(base_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "base-01", "messages": []}) + "\n")

    out_file = tmp_path / "augmented.jsonl"
    total_saved = BoundaryAugmenter.save_augmented_dataset(base_file, out_file, num_samples=20)
    assert total_saved > 1
    assert out_file.exists()
