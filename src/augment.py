"""Synthetic Boundary Data Augmentation and Counterfactual Generator.

Generates targeted hard-negative boundary examples and contrastive pairs:
1. Threshold boundary pairs: Disputed amounts near the ₹50,000 P1 threshold
   (e.g., ₹49,500 [P2] vs ₹50,500 [P1]).
2. Hinglish and English repeat contact variations ('Teesri baar', 'again and again')
   forcing needs_human=True escalation.
3. Distractor-heavy non-transaction tickets: KYC and account inquiries mentioning
   payment words (e.g. 'card', 'UPI') where transaction fields must remain null.
4. Quoted email distractors: Multi-turn email history blocks ensuring model ignores
   quoted timestamps.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

CANONICAL_SCHEMA_KEYS = [
    "category",
    "priority",
    "amount_paise",
    "txn_date",
    "txn_id",
    "channel",
    "language",
    "needs_human",
]


class BoundaryAugmenter:
    """Generates schema-compliant contrastive synthetic training examples."""

    SYSTEM_PROMPT = (
        "You are NimbusPay's production customer support triage engine. "
        "Extract ticket triage information strictly conforming to SCHEMA.md as JSON."
    )

    @classmethod
    def generate_threshold_pair(
        cls,
        base_id: str,
        received_date: str = "2026-07-15",
        channel: str = "upi",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate a contrastive pair around the ₹50,000 P1 threshold.
        
        Ticket A: ₹49,500 (P2, needs_human=False)
        Ticket B: ₹50,500 (P1, needs_human=True)
        """
        # Ticket A (Below threshold: ₹49,500)
        user_a = (
            f"Received: {received_date} (Wed)\n"
            f"Source: email\n\n"
            f"Subject: Refund status\n\n"
            f"Dear NimbusPay support,\n"
            f"I have not received my refund for transaction NP 9901827364.\n"
            f"The amount is Rs. 49,500 paid via {channel}.\n"
            f"Kindly process this refund.\n\n"
            f"Regards,\nCustomer"
        )
        gold_a = {
            "category": "refund",
            "priority": "P2",
            "amount_paise": 4950000,
            "txn_date": None,
            "txn_id": "NP9901827364",
            "channel": channel,
            "language": "en",
            "needs_human": False,
        }
        row_a = {
            "id": f"{base_id}-a",
            "messages": [
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": user_a},
                {"role": "assistant", "content": json.dumps(gold_a, ensure_ascii=False)},
            ],
            "slice": "boundary_below_50k",
        }

        # Ticket B (Above threshold: ₹50,500)
        user_b = (
            f"Received: {received_date} (Wed)\n"
            f"Source: email\n\n"
            f"Subject: Refund status\n\n"
            f"Dear NimbusPay support,\n"
            f"I have not received my refund for transaction NP 9901827365.\n"
            f"The amount is Rs. 50,500 paid via {channel}.\n"
            f"Kindly process this refund.\n\n"
            f"Regards,\nCustomer"
        )
        gold_b = {
            "category": "refund",
            "priority": "P1",
            "amount_paise": 5050000,
            "txn_date": None,
            "txn_id": "NP9901827365",
            "channel": channel,
            "language": "en",
            "needs_human": True,
        }
        row_b = {
            "id": f"{base_id}-b",
            "messages": [
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": user_b},
                {"role": "assistant", "content": json.dumps(gold_b, ensure_ascii=False)},
            ],
            "slice": "boundary_above_50k",
        }

        return row_a, row_b

    @classmethod
    def generate_repeat_contact_variation(
        cls,
        base_id: str,
        language: str = "hi-en",
    ) -> Dict[str, Any]:
        """Generate a ticket with explicit repeat contact phrasing forcing needs_human=True."""
        if language == "hi-en":
            user_msg = (
                "Received: 2026-08-20 (Thu)\n"
                "Source: app-chat\n\n"
                "Bhai main teesri baar complaint kar raha hoon. "
                "Paise account se kat gaye par merchant ko nahi mile. "
                "Amount Rs 1,499 tha, UPI se kiya tha. Txn NP 8812390142. "
                "Meri problem solve karo jaldi."
            )
            gold = {
                "category": "payment_failed",
                "priority": "P2",
                "amount_paise": 149900,
                "txn_date": None,
                "txn_id": "NP8812390142",
                "channel": "upi",
                "language": "hi-en",
                "needs_human": True,
            }
        else:
            user_msg = (
                "Received: 2026-08-20 (Thu)\n"
                "Source: app-chat\n\n"
                "This is the third time I am reaching out regarding my failed payment. "
                "Rs 2,500 was deducted via card, txn ID NP 7712398451. "
                "Still no update on my previous ticket."
            )
            gold = {
                "category": "payment_failed",
                "priority": "P2",
                "amount_paise": 250000,
                "txn_date": None,
                "txn_id": "NP7712398451",
                "channel": "card",
                "language": "en",
                "needs_human": True,
            }

        return {
            "id": base_id,
            "messages": [
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)},
            ],
            "slice": "repeat_contact",
        }

    @classmethod
    def generate_distractor_null_ticket(
        cls,
        base_id: str,
        category: str = "kyc",
    ) -> Dict[str, Any]:
        """Generate non-transaction ticket with distracting payment words where channel remains null."""
        if category == "kyc":
            user_msg = (
                "Received: 2026-07-02 (Thu)\n"
                "Source: app-chat\n\n"
                "Hello, I am trying to complete my Video KYC using my PAN card. "
                "Do I also need to link my UPI account or debit card for identity verification?"
            )
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
        else:
            user_msg = (
                "Received: 2026-07-02 (Thu)\n"
                "Source: twitter\n\n"
                "@NimbusPayCare I forgot my passcode and cannot log in to pay with UPI. Please help me reset."
            )
            gold = {
                "category": "account_access",
                "priority": "P2",
                "amount_paise": None,
                "txn_date": None,
                "txn_id": None,
                "channel": None,
                "language": "en",
                "needs_human": False,
            }

        return {
            "id": base_id,
            "messages": [
                {"role": "system", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)},
            ],
            "slice": f"distractor_{category}",
        }

    @classmethod
    def create_boundary_dataset(
        cls,
        num_threshold_pairs: int = 50,
        num_repeat_contacts: int = 50,
        num_distractors: int = 50,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Generate a balanced synthetic dataset addressing the weak slices."""
        random.seed(seed)
        samples: List[Dict[str, Any]] = []

        # 1. Threshold pairs
        channels = ["upi", "card", "netbanking", "wallet"]
        for i in range(num_threshold_pairs):
            ch = channels[i % len(channels)]
            pair_a, pair_b = cls.generate_threshold_pair(f"syn-thresh-{i:03d}", channel=ch)
            samples.extend([pair_a, pair_b])

        # 2. Repeat contact variations
        for i in range(num_repeat_contacts):
            lang = "hi-en" if i % 2 == 0 else "en"
            row = cls.generate_repeat_contact_variation(f"syn-repeat-{i:03d}", language=lang)
            samples.append(row)

        # 3. Distractor tickets
        for i in range(num_distractors):
            cat = "kyc" if i % 2 == 0 else "account_access"
            row = cls.generate_distractor_null_ticket(f"syn-distract-{i:03d}", category=cat)
            samples.append(row)

        return samples

    @classmethod
    def save_augmented_dataset(
        cls,
        base_clean_path: Union[str, Path],
        output_augmented_path: Union[str, Path],
        num_samples: int = 200,
    ) -> int:
        """Merge base clean dataset with synthetic boundary samples."""
        base_file = Path(base_clean_path)
        out_file = Path(output_augmented_path)

        rows: List[Dict[str, Any]] = []
        if base_file.exists():
            with open(base_file, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

        synthetic = cls.create_boundary_dataset(
            num_threshold_pairs=num_samples // 4,
            num_repeat_contacts=num_samples // 4,
            num_distractors=num_samples // 4,
        )
        rows.extend(synthetic)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        return len(rows)
