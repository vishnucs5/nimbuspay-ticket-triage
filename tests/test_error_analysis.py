"""Test suite for error analysis module (Part E).

Verifies:
- Loading and pairing of gold records and model predictions
- Computation of overall metrics and per-field accuracy
- Slicing breakdowns by category, source, and language
- Failure case extraction and root cause classification
- Curated 10+ case studies formatting and diversity
- One-week technical roadmap specification
- JSON summary export and Markdown report generation
"""

import json
from pathlib import Path
import pytest

from src.error_analysis import (
    load_and_pair_records,
    compute_overall_metrics,
    compute_slice_metrics,
    extract_failure_cases,
    curate_case_studies,
    generate_one_week_roadmap,
    generate_error_analysis_summary,
    generate_error_analysis_markdown,
    run_error_analysis,
    KEYS,
)


@pytest.fixture
def sample_records():
    """Create a controlled sample of paired records for unit testing."""
    return [
        {
            "id": "dv-00001",
            "source": "app-chat",
            "category": "refund",
            "language": "en",
            "received_date": "2026-08-28",
            "user_content": "Received: 2026-08-28 (Fri)\nSource: app-chat\n\nTicket 1",
            "gold": {
                "category": "refund",
                "priority": "P2",
                "amount_paise": 500000,
                "txn_date": "2026-08-27",
                "txn_id": "NP1234567890",
                "channel": "upi",
                "language": "en",
                "needs_human": False,
            },
            "pred": {
                "category": "refund",
                "priority": "P2",
                "amount_paise": 500000,
                "txn_date": "2026-08-27",
                "txn_id": "NP1234567890",
                "channel": "upi",
                "language": "en",
                "needs_human": False,
            },
            "is_valid_json": True,
            "is_exact_match": True,
            "field_matches": {k: True for k in KEYS},
            "field_diffs": {},
        },
        {
            "id": "dv-00002",
            "source": "email",
            "category": "payment_failed",
            "language": "hi-en",
            "received_date": "2026-07-15",
            "user_content": "Received: 2026-07-15 (Wed)\nSource: email\n\nTicket 2 paisa kat gaya",
            "gold": {
                "category": "payment_failed",
                "priority": "P1",
                "amount_paise": 6000000,
                "txn_date": "2026-07-14",
                "txn_id": "NP9876543210",
                "channel": "netbanking",
                "language": "hi-en",
                "needs_human": True,
            },
            "pred": {
                "category": "payment_failed",
                "priority": "P2",  # Mismatch 1: priority
                "amount_paise": 6000000,
                "txn_date": "2026-07-14",
                "txn_id": "NP9876543210",
                "channel": "netbanking",
                "language": "hi-en",
                "needs_human": False,  # Mismatch 2: needs_human
            },
            "is_valid_json": True,
            "is_exact_match": False,
            "field_matches": {
                "category": True,
                "priority": False,
                "amount_paise": True,
                "txn_date": True,
                "txn_id": True,
                "channel": True,
                "language": True,
                "needs_human": False,
            },
            "field_diffs": {
                "priority": {"gold": "P1", "pred": "P2"},
                "needs_human": {"gold": True, "pred": False},
            },
        },
        {
            "id": "dv-00003",
            "source": "twitter",
            "category": "account_access",
            "language": "en",
            "received_date": "2026-06-10",
            "user_content": "Received: 2026-06-10 (Wed)\nSource: twitter\n\n@NimbusPayCare Ticket 3",
            "gold": {
                "category": "account_access",
                "priority": "P2",
                "amount_paise": None,
                "txn_date": None,
                "txn_id": None,
                "channel": None,
                "language": "en",
                "needs_human": False,
            },
            "pred": {
                "category": "payment_failed",  # Mismatch 1: category
                "priority": "P1",  # Mismatch 2: priority
                "amount_paise": None,
                "txn_date": None,
                "txn_id": None,
                "channel": "upi",  # Mismatch 3: channel
                "language": "en",
                "needs_human": False,
            },
            "is_valid_json": True,
            "is_exact_match": False,
            "field_matches": {
                "category": False,
                "priority": False,
                "amount_paise": True,
                "txn_date": True,
                "txn_id": True,
                "channel": False,
                "language": True,
                "needs_human": True,
            },
            "field_diffs": {
                "category": {"gold": "account_access", "pred": "payment_failed"},
                "priority": {"gold": "P2", "pred": "P1"},
                "channel": {"gold": None, "pred": "upi"},
            },
        },
    ]


def test_compute_overall_metrics(sample_records):
    """Test overall accuracy aggregation on sample records."""
    metrics = compute_overall_metrics(sample_records)

    assert metrics["total_tickets"] == 3
    assert metrics["json_valid_count"] == 3
    assert metrics["json_valid_rate"] == 1.0
    assert metrics["exact_match_count"] == 1
    assert pytest.approx(metrics["exact_match_rate"], 0.001) == 1 / 3

    # Per field accuracy
    # priority: 1 match out of 3 = 0.3333
    # needs_human: 2 match out of 3 = 0.6667
    # category: 2 match out of 3 = 0.6667
    # channel: 2 match out of 3 = 0.6667
    # amount_paise: 3 match out of 3 = 1.0
    assert pytest.approx(metrics["per_field_acc"]["priority"], 0.001) == 1 / 3
    assert pytest.approx(metrics["per_field_acc"]["amount_paise"], 0.001) == 1.0

    # Weakest fields should start with priority
    assert metrics["weakest_fields"][0]["field"] == "priority"


def test_compute_slice_metrics(sample_records):
    """Test slicing aggregation across category, source, and language."""
    # Category slice
    cat_slices = compute_slice_metrics(sample_records, slice_by="category")
    assert "refund" in cat_slices
    assert cat_slices["refund"]["n"] == 1
    assert cat_slices["refund"]["exact_match_rate"] == 1.0
    assert "payment_failed" in cat_slices
    assert cat_slices["payment_failed"]["n"] == 1
    assert cat_slices["payment_failed"]["exact_match_rate"] == 0.0

    # Source slice
    src_slices = compute_slice_metrics(sample_records, slice_by="source")
    assert "app-chat" in src_slices
    assert "email" in src_slices
    assert "twitter" in src_slices
    assert src_slices["app-chat"]["exact_match_rate"] == 1.0
    assert src_slices["email"]["exact_match_rate"] == 0.0

    # Language slice
    lang_slices = compute_slice_metrics(sample_records, slice_by="language")
    assert "en" in lang_slices
    assert "hi-en" in lang_slices
    assert lang_slices["en"]["n"] == 2
    assert lang_slices["hi-en"]["n"] == 1


def test_extract_failure_cases(sample_records):
    """Test extracting failing records and populating diffs."""
    failures = extract_failure_cases(sample_records)
    assert len(failures) == 2
    f_ids = [f["id"] for f in failures]
    assert "dv-00002" in f_ids
    assert "dv-00003" in f_ids
    assert "dv-00001" not in f_ids

    case2 = next(f for f in failures if f["id"] == "dv-00002")
    assert "priority" in case2["mismatched_fields"]
    assert "needs_human" in case2["mismatched_fields"]
    assert "explanation" in case2
    assert len(case2["explanation"]) > 20


def test_one_week_roadmap():
    """Verify that roadmap contains actionable high-leverage technical proposals."""
    roadmap = generate_one_week_roadmap()
    assert len(roadmap) >= 4
    for item in roadmap:
        assert "title" in item
        assert "theme" in item
        assert "rationale" in item
        assert "implementation" in item
        assert "expected_impact" in item


def test_load_and_pair_real_data():
    """Test loading the real dev.jsonl and dev_pred_best.jsonl files."""
    gold_path = Path("candidate_pack/data/dev.jsonl")
    pred_path = Path("data/dev_pred_best.jsonl")

    records = load_and_pair_records(gold_path, pred_path)
    assert len(records) == 200

    # Check metrics match Task 4 best score (80.0% exact match)
    overall = compute_overall_metrics(records)
    assert overall["total_tickets"] == 200
    assert overall["exact_match_count"] == 160
    assert pytest.approx(overall["exact_match_rate"], 0.001) == 0.800
    assert pytest.approx(overall["mean_field_acc"], 0.001) == 0.9575

    # Check weakest fields
    weakest_field_names = [f["field"] for f in overall["weakest_fields"]]
    assert weakest_field_names[:2] == ["priority", "needs_human"]


def test_curated_case_studies_diversity():
    """Verify curated case studies contains at least 10 rich cases with diversity."""
    gold_path = Path("candidate_pack/data/dev.jsonl")
    pred_path = Path("data/dev_pred_best.jsonl")

    records = load_and_pair_records(gold_path, pred_path)
    failures = extract_failure_cases(records)
    assert len(failures) == 40  # Exactly 40 failures for 80% on 200 rows

    case_studies = curate_case_studies(failures, min_cases=10)
    assert len(case_studies) >= 10

    # Check required fields
    categories_covered = set()
    sources_covered = set()
    languages_covered = set()

    for cs in case_studies:
        assert "id" in cs
        assert "category" in cs
        assert "source" in cs
        assert "language" in cs
        assert "ticket_text" in cs
        assert "gold" in cs
        assert "pred" in cs
        assert "mismatched_fields" in cs
        assert len(cs["mismatched_fields"]) > 0
        assert "root_cause_category" in cs
        assert "explanation" in cs
        assert len(cs["explanation"]) >= 40  # Meaningful detailed root cause analysis

        categories_covered.add(cs["category"])
        sources_covered.add(cs["source"])
        languages_covered.add(cs["language"])

    # Diversity criteria
    assert len(categories_covered) >= 4
    assert len(sources_covered) >= 3
    assert "en" in languages_covered
    assert "hi-en" in languages_covered


def test_end_to_end_report_generation(tmp_path):
    """Test full execution generating summary JSON and markdown report."""
    gold_path = Path("candidate_pack/data/dev.jsonl")
    pred_path = Path("data/dev_pred_best.jsonl")
    out_json = tmp_path / "error_analysis_summary.json"
    out_md = tmp_path / "error_analysis_report.md"

    summary = run_error_analysis(
        gold_path=gold_path,
        pred_path=pred_path,
        output_json=out_json,
        output_md=out_md,
    )

    assert out_json.exists()
    assert out_md.exists()

    # Verify JSON content
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "overall" in data
    assert "slices" in data
    assert "failure_case_studies" in data
    assert "roadmap" in data
    assert len(data["failure_case_studies"]) >= 10

    # Verify Markdown content
    with open(out_md, "r", encoding="utf-8") as f:
        md_text = f.read()

    assert "# Part E: Comprehensive Error Analysis & Engineering Roadmap" in md_text
    assert "## 1. Overall Performance & Weakest Fields" in md_text
    assert "## 2. Slice-Based Performance Breakdown" in md_text
    assert "## 3. Detailed Failure Case Studies" in md_text
    assert "## 4. One-Week Future Engineering Roadmap" in md_text
    # Ensure at least 10 case study headers exist
    assert md_text.count("### Case Study ") >= 10
