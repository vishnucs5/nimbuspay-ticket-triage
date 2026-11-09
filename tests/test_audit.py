"""Tests for data audit and cleaning pipeline (src/audit.py).

Covers all 8 corruption failure modes and null fields rule:
1. Truncated/Syntax Error JSON
2. Train-Dev Leakage
3. Duplicate Tickets in Train
4. Schema Key Rename (prio -> priority)
5. Priority Value Mapping (high/medium/low -> P1/P2/P3)
6. Date Format Normalization (DD/MM/YYYY -> YYYY-MM-DD)
7. Transaction ID Normalization (np... -> NP...)
8. Rupee-Paise Conversion (raw rupees / truncated decimals -> integer paise)
9. Null Fields Rule for non-transaction categories
"""

import json
import pytest
from src.audit import (
    audit_and_clean_data,
    normalize_txn_id,
    normalize_date,
    detect_rupee_paise_conversion,
    clean_row,
    KEYS,
)


def make_sample_row(
    row_id: str = "tr-00001",
    user_content: str = "Received: 2026-07-25 (Sat)\nSource: app-chat\n\nPayment failed for Rs 500.",
    assistant_dict: dict = None,
    raw_assistant_str: str = None,
) -> dict:
    if raw_assistant_str is not None:
        content = raw_assistant_str
    else:
        if assistant_dict is None:
            assistant_dict = {
                "category": "payment_failed",
                "priority": "P3",
                "amount_paise": 50000,
                "txn_date": "2026-07-25",
                "txn_id": "NP1234567890",
                "channel": "upi",
                "language": "en",
                "needs_human": False,
            }
        content = json.dumps(assistant_dict)
    return {
        "id": row_id,
        "messages": [
            {"role": "system", "content": "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": content},
        ],
    }


def test_failure_mode_1_malformed_json():
    """Drop rows where assistant message is truncated or malformed JSON."""
    row = make_sample_row(
        row_id="tr-bad-json",
        raw_assistant_str='{"category": "refund", "priority": "P2", "amount_paise":',
    )
    cleaned, logs = clean_row(row, dev_user_msgs=set(), seen_user_msgs=set())
    assert cleaned is None
    assert len(logs) == 1
    assert logs[0] == {"id": "tr-bad-json", "action": "drop", "reason": "truncated or malformed JSON"}


def test_failure_mode_2_dev_leakage():
    """Drop rows where customer message is identical to a message in dev.jsonl."""
    leak_msg = "Received: 2026-08-01\nSource: email\n\nLeak message text."
    dev_msgs = {leak_msg}
    row = make_sample_row(row_id="tr-leak", user_content=leak_msg)
    cleaned, logs = clean_row(row, dev_user_msgs=dev_msgs, seen_user_msgs=set())
    assert cleaned is None
    assert len(logs) == 1
    assert logs[0] == {"id": "tr-leak", "action": "drop", "reason": "train-dev data leakage"}


def test_failure_mode_3_duplicate_tickets():
    """Drop subsequent duplicate customer messages in train, keeping the first."""
    dup_msg = "Received: 2026-08-02\nSource: app-chat\n\nDuplicate ticket text."
    seen_msgs = set()

    row1 = make_sample_row(row_id="tr-first", user_content=dup_msg)
    cleaned1, logs1 = clean_row(row1, dev_user_msgs=set(), seen_user_msgs=seen_msgs)
    assert cleaned1 is not None
    assert dup_msg in seen_msgs

    row2 = make_sample_row(row_id="tr-second", user_content=dup_msg)
    cleaned2, logs2 = clean_row(row2, dev_user_msgs=set(), seen_user_msgs=seen_msgs)
    assert cleaned2 is None
    assert len(logs2) == 1
    assert logs2[0] == {"id": "tr-second", "action": "drop", "reason": "duplicate training sample"}


def test_failure_mode_4_schema_key_rename():
    """Rename legacy key 'prio' to 'priority'."""
    a_dict = {
        "category": "refund",
        "prio": "P2",
        "amount_paise": 10000,
        "txn_date": "2026-07-25",
        "txn_id": "NP1234567890",
        "channel": "upi",
        "language": "en",
        "needs_human": False,
    }
    row = make_sample_row(row_id="tr-rename", assistant_dict=a_dict)
    cleaned, logs = clean_row(row, dev_user_msgs=set(), seen_user_msgs=set())
    assert cleaned is not None
    data = json.loads(cleaned["messages"][2]["content"])
    assert "priority" in data
    assert "prio" not in data
    assert data["priority"] == "P2"
    assert any(l["action"] == "fix" and l["reason"] == "v1 schema key prio renamed to priority" for l in logs)


def test_failure_mode_5_priority_mapping():
    """Map legacy values high/medium/low to P1/P2/P3."""
    for legacy, expected in [("high", "P1"), ("medium", "P2"), ("low", "P3")]:
        a_dict = {
            "category": "refund",
            "priority": legacy,
            "amount_paise": 10000,
            "txn_date": "2026-07-25",
            "txn_id": "NP1234567890",
            "channel": "upi",
            "language": "en",
            "needs_human": False,
        }
        row = make_sample_row(row_id=f"tr-prio-{legacy}", assistant_dict=a_dict)
        cleaned, logs = clean_row(row, dev_user_msgs=set(), seen_user_msgs=set())
        assert cleaned is not None
        data = json.loads(cleaned["messages"][2]["content"])
        assert data["priority"] == expected
        assert any(l["action"] == "fix" and l["reason"] == "legacy priority value mapped to v2 spec" for l in logs)


def test_failure_mode_6_date_normalization():
    """Convert DD/MM/YYYY to YYYY-MM-DD."""
    assert normalize_date("26/08/2026") == "2026-08-26"
    assert normalize_date("02/09/2026") == "2026-09-02"
    assert normalize_date("2026-08-26") == "2026-08-26"

    a_dict = {
        "category": "payment_failed",
        "priority": "P3",
        "amount_paise": 10000,
        "txn_date": "26/08/2026",
        "txn_id": "NP1234567890",
        "channel": "upi",
        "language": "en",
        "needs_human": False,
    }
    row = make_sample_row(row_id="tr-date", assistant_dict=a_dict)
    cleaned, logs = clean_row(row, dev_user_msgs=set(), seen_user_msgs=set())
    assert cleaned is not None
    data = json.loads(cleaned["messages"][2]["content"])
    assert data["txn_date"] == "2026-08-26"
    assert any(l["action"] == "fix" and l["reason"] == "normalized date format from DD/MM/YYYY to YYYY-MM-DD" for l in logs)


def test_failure_mode_7_transaction_id_normalization():
    """Normalize txn_id to uppercase NP + 10 digits."""
    assert normalize_txn_id("np9891267518") == "NP9891267518"
    assert normalize_txn_id("NP-9718716109") == "NP9718716109"
    assert normalize_txn_id("np 81633 47892") == "NP8163347892"
    assert normalize_txn_id("NP 9255627510") == "NP9255627510"

    a_dict = {
        "category": "payment_failed",
        "priority": "P3",
        "amount_paise": 10000,
        "txn_date": "2026-08-26",
        "txn_id": "np 81633 47892",
        "channel": "upi",
        "language": "en",
        "needs_human": False,
    }
    row = make_sample_row(row_id="tr-txnid", assistant_dict=a_dict)
    cleaned, logs = clean_row(row, dev_user_msgs=set(), seen_user_msgs=set())
    assert cleaned is not None
    data = json.loads(cleaned["messages"][2]["content"])
    assert data["txn_id"] == "NP8163347892"
    assert any(l["action"] == "fix" and l["reason"] == "normalized transaction ID format to uppercase NP+10digits" for l in logs)


def test_failure_mode_8_rupee_paise_conversion():
    """Convert raw rupee amounts and truncated decimals to integer paise."""
    # Raw rupee integer case: ₹267 recorded as 267 -> should be 26700
    msg_raw_rupees = "Hi, my payment failed. Amount: ₹267. Please check."
    converted = detect_rupee_paise_conversion(msg_raw_rupees, 267)
    assert converted == 26700

    # Truncated decimal case: Rs 1,68,787.25 recorded as 168787 -> should be 16878725
    msg_truncated = "The transaction amount is Rs 1,68,787.25. Please help."
    converted_dec = detect_rupee_paise_conversion(msg_truncated, 168787)
    assert converted_dec == 16878725

    # Valid paise amount should NOT be converted
    msg_normal = "It was 4.7k in total."
    assert detect_rupee_paise_conversion(msg_normal, 470000) is None

    # Full row test
    a_dict = {
        "category": "payment_failed",
        "priority": "P1",
        "amount_paise": 168787,
        "txn_date": "2026-07-15",
        "txn_id": "NP4271342441",
        "channel": "card",
        "language": "en",
        "needs_human": True,
    }
    row = make_sample_row(row_id="tr-amt", user_content=msg_truncated, assistant_dict=a_dict)
    cleaned, logs = clean_row(row, dev_user_msgs=set(), seen_user_msgs=set())
    assert cleaned is not None
    data = json.loads(cleaned["messages"][2]["content"])
    assert data["amount_paise"] == 16878725
    assert any(l["action"] == "fix" and l["reason"] == "converted rupees to paise" for l in logs)


def test_null_fields_rule():
    """For account_access, kyc, other: amount_paise, txn_date, txn_id, channel must be null."""
    for cat in ["account_access", "kyc", "other"]:
        a_dict = {
            "category": cat,
            "priority": "P2",
            "amount_paise": 5000,
            "txn_date": "2026-08-01",
            "txn_id": "NP1234567890",
            "channel": "upi",
            "language": "en",
            "needs_human": False,
        }
        row = make_sample_row(row_id=f"tr-null-{cat}", assistant_dict=a_dict)
        cleaned, logs = clean_row(row, dev_user_msgs=set(), seen_user_msgs=set())
        assert cleaned is not None
        data = json.loads(cleaned["messages"][2]["content"])
        assert data["amount_paise"] is None
        assert data["txn_date"] is None
        assert data["txn_id"] is None
        assert data["channel"] is None
        assert any(l["action"] == "fix" and "null" in l["reason"].lower() for l in logs)


def test_end_to_end_audit_and_clean(tmp_path):
    """End-to-end integration test with synthetic files."""
    train_file = tmp_path / "train.jsonl"
    dev_file = tmp_path / "dev.jsonl"
    clean_out = tmp_path / "train_clean.jsonl"
    log_out = tmp_path / "cleaning_log.csv"

    dev_rows = [
        make_sample_row("dv-001", "Dev message 1"),
        make_sample_row("dv-002", "Dev message 2"),
    ]
    with open(dev_file, "w", encoding="utf-8") as f:
        for r in dev_rows:
            f.write(json.dumps(r) + "\n")

    train_rows = [
        make_sample_row("tr-001", "Unique message 1"),
        make_sample_row("tr-002", "Dev message 1"),  # Leakage
        make_sample_row("tr-003", "Unique message 1"),  # Duplicate
        make_sample_row("tr-004", "Malformed JSON", raw_assistant_str="invalid json"),  # Malformed
        make_sample_row("tr-005", "Fixable row with Rs 500", assistant_dict={  # Multiple fixes
            "category": "refund",
            "prio": "high",
            "amount_paise": 500,
            "txn_date": "25/07/2026",
            "txn_id": "np 12345 67890",
            "channel": "upi",
            "language": "en",
            "needs_human": True,
        }),
    ]
    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")

    clean_data, log_data = audit_and_clean_data(
        str(train_file), str(dev_file), str(clean_out), str(log_out)
    )

    # tr-001 and tr-005 should be kept
    assert len(clean_data) == 2
    assert [r["id"] for r in clean_data] == ["tr-001", "tr-005"]

    # Drops: tr-004 (malformed), tr-002 (leak), tr-003 (duplicate)
    drops = [l for l in log_data if l["action"] == "drop"]
    assert len(drops) == 3
    drop_ids = {d["id"] for d in drops}
    assert drop_ids == {"tr-004", "tr-002", "tr-003"}

    # Fixes for tr-005: prio key, priority val, date, txn_id, amount_paise
    fixes = [l for l in log_data if l["id"] == "tr-005" and l["action"] == "fix"]
    assert len(fixes) == 5

    # Check files created on disk
    assert clean_out.exists()
    assert log_out.exists()


def test_actual_candidate_pack_dataset():
    """Verify auditing and cleaning on the actual candidate_pack dataset."""
    train_path = "candidate_pack/data/train.jsonl"
    dev_path = "candidate_pack/data/dev.jsonl"

    clean_rows, log_rows = audit_and_clean_data(train_path, dev_path)

    assert len(clean_rows) == 1330
    assert len(log_rows) == 327

    drops = [l for l in log_rows if l["action"] == "drop"]
    fixes = [l for l in log_rows if l["action"] == "fix"]

    assert len(drops) == 115
    assert len(fixes) == 212

    # Check drop breakdown
    malformed = [l for l in drops if l["reason"] == "truncated or malformed JSON"]
    leakage = [l for l in drops if l["reason"] == "train-dev data leakage"]
    duplicates = [l for l in drops if l["reason"] == "duplicate training sample"]
    assert len(malformed) == 25
    assert len(leakage) == 30
    assert len(duplicates) == 60

    # Check fix breakdown
    key_renames = [l for l in fixes if l["reason"] == "v1 schema key prio renamed to priority"]
    prio_maps = [l for l in fixes if l["reason"] == "legacy priority value mapped to v2 spec"]
    date_norms = [l for l in fixes if l["reason"] == "normalized date format from DD/MM/YYYY to YYYY-MM-DD"]
    txn_norms = [l for l in fixes if l["reason"] == "normalized transaction ID format to uppercase NP+10digits"]
    rupee_fixes = [l for l in fixes if l["reason"] == "converted rupees to paise"]
    assert len(key_renames) == 70
    assert len(prio_maps) == 70
    assert len(date_norms) == 30
    assert len(txn_norms) == 20
    assert len(rupee_fixes) == 22

    # Verify structural integrity of all cleaned rows
    for r in clean_rows:
        data = json.loads(r["messages"][-1]["content"])
        assert list(data.keys()) == KEYS
        assert data["category"] in {"payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other"}
        assert data["priority"] in {"P1", "P2", "P3"}
        if data["amount_paise"] is not None:
            assert isinstance(data["amount_paise"], int)
        if data["txn_date"] is not None:
            assert len(data["txn_date"]) == 10 and data["txn_date"][4] == "-" and data["txn_date"][7] == "-"
        if data["txn_id"] is not None:
            assert data["txn_id"].startswith("NP") and len(data["txn_id"]) == 12
        if data["channel"] is not None:
            assert data["channel"] in {"upi", "card", "netbanking", "wallet"}
        assert data["language"] in {"en", "hi-en"}
        assert isinstance(data["needs_human"], bool)
