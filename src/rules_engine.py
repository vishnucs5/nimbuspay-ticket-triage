"""Deterministic Business Rule and Schema Invariant Post-Processing Engine.

Enforces explicit business rules and schema invariants from candidate_pack/SCHEMA.md:
1. P1 mandates needs_human=True (Rule 1).
2. High-value transactions (>= 50,000 INR / 5,000,000 paise) mandate P1 and needs_human=True.
3. Category 'fraud' mandates P1 and needs_human=True.
4. Category 'account_access' priority floor: minimum P2 (never P3).
5. Non-transaction categories ('kyc', 'account_access', 'other') must have null channel
   unless a transaction is explicitly referenced.
6. Fallback extraction for omitted channels on transaction categories.
7. Legal threats ('ombudsman', 'consumer court', 'police') mandate P1 and needs_human=True.
8. Repeat contacts ('second time', 'doosri baar', 'teesri baar', 'again and again') mandate needs_human=True.
9. Symbolic date resolution for relative and unanchored dates.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, Optional, Union

from src.date_parser import SymbolicDateParser

# Valid schema keys and canonical types
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

LEGAL_REGEX = re.compile(
    r"\b(ombudsman|consumer\s+court|legal\s+notice|police\s+complaint|police|lawyer|fir)\b",
    re.IGNORECASE,
)

REPEAT_CONTACT_REGEX = re.compile(
    r"\b(teesri\s+baar|third\s+time|doosri\s+baar|second\s+time|baar\s+baar|"
    r"again\s+and\s+again|multiple\s+times|previous\s+ticket|repeatedly|"
    r"not\s+resolved\s+yet|already\s+complained)\b",
    re.IGNORECASE,
)


class RulesEngine:
    """Applies deterministic business rules and schema invariants."""

    def __init__(
        self,
        enable_threshold_escalation: bool = True,
        enable_fraud_mandate: bool = True,
        enable_p1_invariant: bool = True,
        enable_priority_floor: bool = True,
        enable_null_channel_invariant: bool = True,
        enable_channel_fallback: bool = True,
        enable_legal_escalation: bool = True,
        enable_repeat_contact: bool = True,
        enable_date_resolution: bool = True,
    ):
        self.enable_threshold_escalation = enable_threshold_escalation
        self.enable_fraud_mandate = enable_fraud_mandate
        self.enable_p1_invariant = enable_p1_invariant
        self.enable_priority_floor = enable_priority_floor
        self.enable_null_channel_invariant = enable_null_channel_invariant
        self.enable_channel_fallback = enable_channel_fallback
        self.enable_legal_escalation = enable_legal_escalation
        self.enable_repeat_contact = enable_repeat_contact
        self.enable_date_resolution = enable_date_resolution

    def postprocess_ticket(
        self,
        ticket_text: str,
        predicted_json: Union[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Post-process model predictions by enforcing deterministic rules and invariants."""
        if isinstance(predicted_json, str):
            try:
                data = json.loads(predicted_json.strip())
            except Exception:
                return {}
        else:
            data = copy.deepcopy(predicted_json)

        if not isinstance(data, dict):
            return {}

        cat = data.get("category")
        amt = data.get("amount_paise")
        txn_id = data.get("txn_id")
        prio = data.get("priority")
        needs_human = data.get("needs_human")
        channel = data.get("channel")
        txn_date = data.get("txn_date")

        # 1. Date resolution
        if self.enable_date_resolution:
            data["txn_date"] = SymbolicDateParser.resolve_date(ticket_text, txn_date)

        # 2. Non-transaction categories null transaction fields invariant
        if self.enable_null_channel_invariant:
            if cat in ["kyc", "account_access", "other"]:
                data["channel"] = None
                data["amount_paise"] = None
                data["txn_date"] = None
                data["txn_id"] = None

        # 3. Channel fallback extraction for transaction tickets
        if self.enable_channel_fallback:
            if data.get("channel") is None and (
                txn_id is not None or amt is not None or cat in ["payment_failed", "refund"]
            ):
                if re.search(r"\b(wallet|nimbuspay wallet)\b", ticket_text, re.IGNORECASE):
                    data["channel"] = "wallet"
                elif re.search(r"\b(netbanking|internet banking|net banking)\b", ticket_text, re.IGNORECASE):
                    data["channel"] = "netbanking"
                elif re.search(r"\b(card|credit card|debit card)\b", ticket_text, re.IGNORECASE):
                    data["channel"] = "card"
                elif re.search(r"\b(upi)\b", ticket_text, re.IGNORECASE):
                    data["channel"] = "upi"

        # 4. High-value dispute threshold (>= 50,000 INR / 5,000,000 paise)
        if self.enable_threshold_escalation:
            if data.get("amount_paise") is not None and data["amount_paise"] >= 5_000_000:
                data["priority"] = "P1"
                data["needs_human"] = True

        # 5. Fraud category mandate
        if self.enable_fraud_mandate:
            if data.get("category") == "fraud":
                data["priority"] = "P1"
                data["needs_human"] = True

        # 6. Priority floor for account_access (cannot be P3)
        if self.enable_priority_floor:
            if data.get("category") == "account_access" and data.get("priority") == "P3":
                data["priority"] = "P2"

        # 7. Legal threat escalation
        if self.enable_legal_escalation:
            if LEGAL_REGEX.search(ticket_text):
                data["priority"] = "P1"
                data["needs_human"] = True

        # 8. Repeat contact escalation
        if self.enable_repeat_contact:
            if REPEAT_CONTACT_REGEX.search(ticket_text):
                data["needs_human"] = True

        # 9. Schema Invariant: P1 -> needs_human=True
        if self.enable_p1_invariant:
            if data.get("priority") == "P1":
                data["needs_human"] = True

        # Ensure all 8 schema keys are present
        for key in SCHEMA_KEYS:
            if key not in data:
                data[key] = None

        return data
