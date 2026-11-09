"""Tests for Hybrid Triage Inference Engine."""
import json
from pathlib import Path
import pytest
from src.infer import generate_single_prediction, run_inference_on_file
from src.rules_engine import RulesEngine


def test_hybrid_single_prediction():
    ticket = (
        "Received: 2026-07-02 (Thu)\n"
        "Source: email\n\n"
        "Dear NimbusPay support,\n"
        "Give me a refund. Mode: wallet. It is Rs 60,000 in total. Txn NP 2300785999."
    )
    # With rules engine enabled, amount >= 50k should force P1 and needs_human=True
    pred_str = generate_single_prediction(
        system_msg="Triage engine",
        user_msg=ticket,
        offline_fallback=True,
        use_rules_engine=True,
    )
    pred = json.loads(pred_str)
    assert pred["priority"] == "P1"
    assert pred["needs_human"] is True
    assert pred["channel"] == "wallet"


def test_hybrid_run_inference_on_file(tmp_path):
    input_file = tmp_path / "dev_sample.jsonl"
    output_file = tmp_path / "pred_hybrid.jsonl"

    ticket = {
        "id": "samp-001",
        "messages": [
            {"role": "system", "content": "Triage engine"},
            {"role": "user", "content": "Video KYC verification status? No money involved."},
        ],
    }
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(ticket) + "\n")

    count = run_inference_on_file(
        input_path=input_file,
        output_path=output_file,
        offline_fallback=True,
        use_rules_engine=True,
    )
    assert count == 1
    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        row = json.loads(f.readline())
        assert row["id"] == "samp-001"
        data = json.loads(row["output"])
        assert data["category"] == "kyc"
        assert data["channel"] is None
        assert data["amount_paise"] is None


def test_hybrid_improves_dev_predictions(tmp_path):
    """Verify that applying RulesEngine on dev predictions yields >= 85% exact match."""
    from candidate_pack.score import load, score

    gold = load("candidate_pack/data/dev.jsonl")
    pred = load("data/dev_pred_best.jsonl")
    ticket_dict = {g["id"]: g["messages"][1]["content"] for g in gold}

    engine = RulesEngine()
    hybrid_preds = []
    for p in pred:
        pid = p["id"]
        t_text = ticket_dict[pid]
        refined = engine.postprocess_ticket(t_text, p["output"])
        hybrid_preds.append({"id": pid, "output": json.dumps(refined, ensure_ascii=False)})

    res = score(gold, hybrid_preds)
    # Confirms >= 85% exact match (baseline was 80.0%)
    assert res["all"]["exact_match"] >= 0.85
    assert res["all"]["mean_field_acc"] >= 0.96
