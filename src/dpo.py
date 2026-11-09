"""Direct Preference Optimization (DPO / RLAIF) Dataset Generator.

Constructs high-value (prompt, chosen, rejected) preference pairs targeting
systematic failure modes:
1. Null field invariants: Rejects hallucinated channel on KYC/account_access tickets.
2. Threshold violations: Rejects under-escalated P2 predictions on >= ₹50,000 disputes.
3. Priority floor violations: Rejects P3 predictions on account_access inquiries.
4. Dummy date fallbacks: Rejects ungrounded '2026-01-01' dates.
5. Invariant enforcement: Rejects P1 tickets with needs_human=False.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DPOPairGenerator:
    """Generates preference pairs for Direct Preference Optimization."""

    @classmethod
    def generate_preference_pair(
        cls,
        ticket_id: str,
        user_prompt: str,
        gold_json: Dict[str, Any],
        violation_type: str = "null_channel",
    ) -> Dict[str, Any]:
        """Construct a single (prompt, chosen, rejected) preference pair.

        Args:
            ticket_id: Unique record ID.
            user_prompt: Ticket prompt text.
            gold_json: Ground truth valid JSON object (chosen).
            violation_type: Type of failure mode to synthesize in rejected.

        Returns:
            DPO preference record with prompt, chosen, rejected strings.
        """
        chosen = copy.deepcopy(gold_json)
        rejected = copy.deepcopy(gold_json)

        if violation_type == "null_channel":
            # Hallucinate channel='upi' on non-transaction ticket
            rejected["channel"] = "upi"
        elif violation_type == "under_escalated":
            # Demote P1 ticket to P2 and needs_human=False
            rejected["priority"] = "P2"
            rejected["needs_human"] = False
        elif violation_type == "priority_floor":
            # Demote account_access to P3
            rejected["priority"] = "P3"
        elif violation_type == "dummy_date":
            # Replace actual date with lazy dummy fallback
            rejected["txn_date"] = "2026-01-01"
        elif violation_type == "p1_invariant":
            # Break Rule 1 invariant: P1 with needs_human=False
            rejected["priority"] = "P1"
            rejected["needs_human"] = False
        else:
            rejected["needs_human"] = not chosen.get("needs_human", False)

        return {
            "id": ticket_id,
            "prompt": user_prompt,
            "chosen": json.dumps(chosen, ensure_ascii=False),
            "rejected": json.dumps(rejected, ensure_ascii=False),
            "violation_type": violation_type,
        }

    @classmethod
    def harvest_from_dataset(
        cls,
        clean_path: Union[str, Path],
        max_pairs: int = 400,
    ) -> List[Dict[str, Any]]:
        """Harvest DPO preference pairs from clean dataset by targeting vulnerable records."""
        path = Path(clean_path)
        if not path.exists():
            return []

        pairs: List[Dict[str, Any]] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = row.get("id", "dpo-row")
                messages = row.get("messages", [])

                user_content = ""
                gold_json = {}
                for msg in messages:
                    if msg.get("role") == "user":
                        user_content = msg.get("content", "")
                    elif msg.get("role") == "assistant":
                        try:
                            gold_json = json.loads(msg.get("content", "{}"))
                        except Exception:
                            gold_json = {}

                if not user_content or not gold_json:
                    continue

                cat = gold_json.get("category")
                amt = gold_json.get("amount_paise")
                prio = gold_json.get("priority")
                dt = gold_json.get("txn_date")

                # Strategy 1: Non-transaction null violations
                if cat in ["kyc", "account_access", "other"] and gold_json.get("channel") is None:
                    pairs.append(
                        cls.generate_preference_pair(
                            f"{row_id}-dpo-null", user_content, gold_json, violation_type="null_channel"
                        )
                    )

                # Strategy 2: High amount threshold violations
                elif amt is not None and amt >= 5_000_000:
                    pairs.append(
                        cls.generate_preference_pair(
                            f"{row_id}-dpo-thresh", user_content, gold_json, violation_type="under_escalated"
                        )
                    )

                # Strategy 3: Priority floor on account_access
                elif cat == "account_access" and prio == "P2":
                    pairs.append(
                        cls.generate_preference_pair(
                            f"{row_id}-dpo-floor", user_content, gold_json, violation_type="priority_floor"
                        )
                    )

                # Strategy 4: Relative / explicit date dummy fallbacks
                elif dt is not None and dt != "2026-01-01":
                    pairs.append(
                        cls.generate_preference_pair(
                            f"{row_id}-dpo-date", user_content, gold_json, violation_type="dummy_date"
                        )
                    )

                if len(pairs) >= max_pairs:
                    break

        return pairs

    @classmethod
    def save_preference_dataset(
        cls,
        clean_path: Union[str, Path],
        output_path: Union[str, Path],
        max_pairs: int = 400,
    ) -> int:
        """Harvest and save DPO preference dataset to JSONL."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        pairs = cls.harvest_from_dataset(clean_path, max_pairs=max_pairs)
        with open(out_file, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        return len(pairs)
