"""Unit tests for RulesEngine."""
import pytest
from src.rules_engine import RulesEngine


@pytest.fixture
def engine():
    return RulesEngine()


def test_rule_1_high_value_escalation(engine):
    ticket = "My refund of Rs 60,000 is missing."
    pred = {
        "category": "refund",
        "priority": "P2",
        "amount_paise": 6000000,
        "txn_date": "2026-07-01",
        "txn_id": "NP123456",
        "channel": "upi",
        "language": "en",
        "needs_human": False,
    }
    refined = engine.postprocess_ticket(ticket, pred)
    assert refined["priority"] == "P1"
    assert refined["needs_human"] is True


def test_rule_2_fraud_category_mandate(engine):
    ticket = "Strange withdrawal on my card."
    pred = {
        "category": "fraud",
        "priority": "P2",
        "amount_paise": 100000,
        "txn_date": "2026-07-01",
        "txn_id": "NP123456",
        "channel": "card",
        "language": "en",
        "needs_human": False,
    }
    refined = engine.postprocess_ticket(ticket, pred)
    assert refined["priority"] == "P1"
    assert refined["needs_human"] is True


def test_rule_3_p1_invariant_forces_needs_human(engine):
    ticket = "Payment failed."
    pred = {
        "category": "payment_failed",
        "priority": "P1",
        "amount_paise": 50000,
        "txn_date": "2026-07-01",
        "txn_id": "NP123456",
        "channel": "upi",
        "language": "en",
        "needs_human": False,
    }
    refined = engine.postprocess_ticket(ticket, pred)
    assert refined["needs_human"] is True


def test_rule_4_account_access_floor(engine):
    ticket = "I forgot my password."
    pred = {
        "category": "account_access",
        "priority": "P3",
        "amount_paise": None,
        "txn_date": None,
        "txn_id": None,
        "channel": None,
        "language": "en",
        "needs_human": False,
    }
    refined = engine.postprocess_ticket(ticket, pred)
    assert refined["priority"] == "P2"


def test_rule_5_non_transaction_null_channel(engine):
    ticket = "I need help with Video KYC verification."
    pred = {
        "category": "kyc",
        "priority": "P2",
        "amount_paise": None,
        "txn_date": None,
        "txn_id": None,
        "channel": "upi",
        "language": "en",
        "needs_human": False,
    }
    refined = engine.postprocess_ticket(ticket, pred)
    assert refined["channel"] is None


def test_rule_6_channel_fallback_extraction(engine):
    ticket = "Order failed. Mode: wallet. Amount Rs 500. Txn ID NP12345."
    pred = {
        "category": "payment_failed",
        "priority": "P2",
        "amount_paise": 50000,
        "txn_date": "2026-07-01",
        "txn_id": "NP12345",
        "channel": None,
        "language": "en",
        "needs_human": False,
    }
    refined = engine.postprocess_ticket(ticket, pred)
    assert refined["channel"] == "wallet"


def test_rule_7_legal_threat_escalation(engine):
    ticket = "If money is not returned I will file a complaint with Banking Ombudsman."
    pred = {
        "category": "refund",
        "priority": "P2",
        "amount_paise": 100000,
        "txn_date": "2026-07-01",
        "txn_id": "NP12345",
        "channel": "upi",
        "language": "en",
        "needs_human": False,
    }
    refined = engine.postprocess_ticket(ticket, pred)
    assert refined["priority"] == "P1"
    assert refined["needs_human"] is True


def test_rule_8_repeat_contact_escalation(engine):
    ticket = "Teesri baar complaint kar raha hoon, solve karo."
    pred = {
        "category": "payment_failed",
        "priority": "P2",
        "amount_paise": 100000,
        "txn_date": "2026-07-01",
        "txn_id": "NP12345",
        "channel": "upi",
        "language": "en",
        "needs_human": False,
    }
    refined = engine.postprocess_ticket(ticket, pred)
    assert refined["needs_human"] is True


def test_postprocess_invalid_input(engine):
    assert engine.postprocess_ticket("text", "not json") == {}
    assert engine.postprocess_ticket("text", None) == {}
