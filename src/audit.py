"""Automated data auditor and cleaner for NimbusPay triage dataset.

Audits training rows against SCHEMA.md and dev.jsonl, handling 8 corruption failure modes:
1. Truncated/Syntax Error JSON (drop)
2. Train-Dev Leakage (drop)
3. Duplicate Tickets in Train (drop)
4. Schema Key Rename (fix: prio -> priority)
5. Priority Value Mapping (fix: high/medium/low -> P1/P2/P3)
6. Date Format Normalization (fix: DD/MM/YYYY -> YYYY-MM-DD)
7. Transaction ID Normalization (fix: spaces, hyphens, np -> uppercase NP+10digits)
8. Rupee-Paise Conversion (fix: raw rupees / truncated decimals -> integer paise)
Plus null fields enforcement for non-transaction categories.
"""

import argparse
import csv
import json
import os
import re
from typing import List, Optional, Set, Tuple

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
NON_TXN_CATEGORIES = {"account_access", "kyc", "other"}
TXN_FIELDS = ["amount_paise", "txn_date", "txn_id", "channel"]
PRIORITY_MAP = {"high": "P1", "medium": "P2", "low": "P3"}

CURR_PATTERNS = [
    r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)",
    r"([\d,]+(?:\.\d+)?)\s*(?:rupees|rupaye|rs\b)",
]
AMOUNT_CONTEXT_PATTERN = r"(?:amount|rakam|total|value|is|for)\s*[:\s]\s*([\d,]+(?:\.\d+)?)"


def normalize_date(date_str: str) -> Optional[str]:
    """Normalize DD/MM/YYYY to ISO YYYY-MM-DD."""
    if not date_str:
        return None
    s = str(date_str).strip()
    if re.match(r"^\d{2}/\d{2}/\d{4}$", s):
        parts = s.split("/")
        return f"{parts[2]}-{parts[1]}-{parts[0]}"
    return s


def normalize_txn_id(txn_id: str) -> Optional[str]:
    """Normalize transaction ID to uppercase NP + 10 digits."""
    if not txn_id:
        return None
    s = re.sub(r"[\s\-]", "", str(txn_id)).upper()
    if re.match(r"^NP\d{10}$", s):
        return s
    return str(txn_id)


def detect_rupee_paise_conversion(user_msg: str, recorded_amt: Optional[int]) -> Optional[int]:
    """Detect if recorded amount was in raw rupees or truncated decimals and convert to paise."""
    if recorded_amt is None:
        return None

    # Exclude quoted earlier customer/support emails (> ...)
    unquoted_lines = [l for l in user_msg.splitlines() if not l.strip().startswith(">")]
    unquoted = "\n".join(unquoted_lines)

    matches: List[float] = []
    for pat in CURR_PATTERNS:
        for m in re.finditer(pat, unquoted, re.IGNORECASE):
            try:
                matches.append(float(m.group(1).replace(",", "")))
            except ValueError:
                pass
    for m in re.finditer(AMOUNT_CONTEXT_PATTERN, unquoted, re.IGNORECASE):
        try:
            matches.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass

    for val in matches:
        # Case 1: Raw rupee integer recorded directly (e.g. ₹267 stored as 267 instead of 26700)
        if abs(recorded_amt - val) < 1e-4:
            return int(round(val * 100))
        # Case 2: Truncated decimal (e.g. ₹1,68,787.25 stored as 168787 instead of 16878725)
        if val > 0 and abs(recorded_amt - int(val)) < 1e-4 and abs(val - int(val)) > 1e-4:
            return int(round(val * 100))

    return None


def clean_row(
    row: dict,
    dev_user_msgs: Set[str],
    seen_user_msgs: Set[str],
) -> Tuple[Optional[dict], List[dict]]:
    """Clean a single row, returning (cleaned_row_or_none, list_of_log_dicts)."""
    row_id = row.get("id", "unknown")
    logs: List[dict] = []

    # Extract user and assistant messages
    try:
        user_msg = next(m["content"] for m in row["messages"] if m["role"] == "user")
        assistant_content = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    except (KeyError, StopIteration):
        logs.append({"id": row_id, "action": "drop", "reason": "truncated or malformed JSON"})
        return None, logs

    # 1. Truncated / Syntax Error JSON
    try:
        data = json.loads(assistant_content)
        if not isinstance(data, dict):
            raise ValueError("Assistant content is not a JSON object")
    except Exception:
        logs.append({"id": row_id, "action": "drop", "reason": "truncated or malformed JSON"})
        return None, logs

    # 2. Train-Dev Data Leakage
    if user_msg in dev_user_msgs:
        logs.append({"id": row_id, "action": "drop", "reason": "train-dev data leakage"})
        return None, logs

    # 3. Duplicate Tickets in Train
    if user_msg in seen_user_msgs:
        logs.append({"id": row_id, "action": "drop", "reason": "duplicate training sample"})
        return None, logs
    seen_user_msgs.add(user_msg)

    # 4. Schema Key Rename: "prio" -> "priority"
    if "prio" in data:
        data["priority"] = data.pop("prio")
        logs.append({"id": row_id, "action": "fix", "reason": "v1 schema key prio renamed to priority"})

    # 5. Priority Value Mapping: high/medium/low -> P1/P2/P3
    prio_val = data.get("priority")
    if prio_val in PRIORITY_MAP:
        data["priority"] = PRIORITY_MAP[prio_val]
        logs.append({"id": row_id, "action": "fix", "reason": "legacy priority value mapped to v2 spec"})

    # 6. Date Format Normalization: DD/MM/YYYY -> YYYY-MM-DD
    txn_date = data.get("txn_date")
    if txn_date and re.match(r"^\d{2}/\d{2}/\d{4}$", str(txn_date)):
        data["txn_date"] = normalize_date(str(txn_date))
        logs.append({"id": row_id, "action": "fix", "reason": "normalized date format from DD/MM/YYYY to YYYY-MM-DD"})

    # 7. Transaction ID Normalization
    txn_id = data.get("txn_id")
    if txn_id is not None and not re.match(r"^NP\d{10}$", str(txn_id)):
        norm_txn = normalize_txn_id(str(txn_id))
        if re.match(r"^NP\d{10}$", norm_txn):
            data["txn_id"] = norm_txn
            logs.append({"id": row_id, "action": "fix", "reason": "normalized transaction ID format to uppercase NP+10digits"})

    # 8. Rupee-Paise Conversion
    amt = data.get("amount_paise")
    if amt is not None:
        corrected_amt = detect_rupee_paise_conversion(user_msg, amt)
        if corrected_amt is not None:
            data["amount_paise"] = corrected_amt
            logs.append({"id": row_id, "action": "fix", "reason": "converted rupees to paise"})

    # Null fields rule for non-transaction categories
    cat = data.get("category")
    if cat in NON_TXN_CATEGORIES:
        nullified = False
        for field in TXN_FIELDS:
            if data.get(field) is not None:
                data[field] = None
                nullified = True
        if nullified:
            logs.append({"id": row_id, "action": "fix", "reason": "nullified transaction fields for non-transaction category"})

    # Reconstruct row with strictly ordered keys in assistant JSON
    ordered_data = {k: data.get(k) for k in KEYS}
    new_messages = []
    for m in row["messages"]:
        if m["role"] == "assistant":
            new_messages.append({"role": "assistant", "content": json.dumps(ordered_data)})
        else:
            new_messages.append(m)

    clean_row_obj = dict(row)
    clean_row_obj["messages"] = new_messages
    return clean_row_obj, logs


def export_cleaning_log(log_rows: List[dict], output_csv_path: str) -> None:
    """Export cleaning log to CSV with header id,action,reason."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "action", "reason"])
        writer.writeheader()
        for log in log_rows:
            writer.writerow(log)


def export_clean_jsonl(clean_rows: List[dict], output_jsonl_path: str) -> None:
    """Export cleaned rows to JSONL format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl_path)), exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for r in clean_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def audit_and_clean_data(
    train_path: str,
    dev_path: str,
    output_train_path: Optional[str] = None,
    output_log_path: Optional[str] = None,
) -> Tuple[List[dict], List[dict]]:
    """Audit and clean training dataset against SCHEMA.md and dev.jsonl."""
    # Load dev user messages for leakage detection
    dev_user_msgs: Set[str] = set()
    with open(dev_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            for m in item.get("messages", []):
                if m.get("role") == "user":
                    dev_user_msgs.add(m.get("content", ""))

    seen_user_msgs: Set[str] = set()
    clean_rows: List[dict] = []
    log_rows: List[dict] = []

    with open(train_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cleaned, logs = clean_row(row, dev_user_msgs, seen_user_msgs)
            log_rows.extend(logs)
            if cleaned is not None:
                clean_rows.append(cleaned)

    if output_train_path:
        export_clean_jsonl(clean_rows, output_train_path)

    if output_log_path:
        export_cleaning_log(log_rows, output_log_path)

    return clean_rows, log_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and clean NimbusPay training data.")
    parser.add_argument("--train", default="candidate_pack/data/train.jsonl", help="Path to raw train.jsonl")
    parser.add_argument("--dev", default="candidate_pack/data/dev.jsonl", help="Path to dev.jsonl")
    parser.add_argument("--output-clean", default="data/train_clean.jsonl", help="Path to output clean.jsonl")
    parser.add_argument("--output-log", default="data/cleaning_log.csv", help="Path to output cleaning_log.csv")
    args = parser.parse_args()

    clean_rows, log_rows = audit_and_clean_data(
        train_path=args.train,
        dev_path=args.dev,
        output_train_path=args.output_clean,
        output_log_path=args.output_log,
    )

    drops = sum(1 for l in log_rows if l["action"] == "drop")
    fixes = sum(1 for l in log_rows if l["action"] == "fix")

    print(f"Audit and cleaning complete:")
    print(f"  Cleaned rows: {len(clean_rows)}")
    print(f"  Dropped rows: {drops}")
    print(f"  Fixed rows:   {fixes}")
    print(f"  Total logs:   {len(log_rows)}")
    print(f"  Clean data saved to: {args.output_clean}")
    print(f"  Cleaning log saved to: {args.output_log}")


if __name__ == "__main__":
    main()
