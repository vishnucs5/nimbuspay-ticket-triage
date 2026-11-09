"""Unit tests for SymbolicDateParser."""
import pytest
from src.date_parser import SymbolicDateParser


def test_strip_quoted_lines():
    text = (
        "Dear NimbusPay,\n"
        "My refund is missing.\n"
        "> On 12 Jul 2026 at 10:00, support wrote:\n"
        "> Your ticket was closed.\n"
        "Please re-open this immediately."
    )
    cleaned = SymbolicDateParser.strip_quoted_lines(text)
    assert "> On 12 Jul" not in cleaned
    assert "My refund is missing." in cleaned
    assert "Please re-open this immediately." in cleaned


def test_extract_received_date():
    text = "Received: 2026-07-16 (Thu)\nSource: app-chat\n\nHello, I need help."
    dt = SymbolicDateParser.extract_received_date(text)
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 7
    assert dt.day == 16


def test_resolve_date_today_aaj():
    text = "Received: 2026-06-02 (Tue)\nSource: twitter\n\nBill pay kiya tha, yeh aaj hua."
    resolved = SymbolicDateParser.resolve_date(text, current_pred_date="2026-01-01")
    assert resolved == "2026-06-02"


def test_resolve_date_yesterday_kal():
    text = "Received: 2026-07-10 (Fri)\nSource: app-chat\n\nTransaction failed yesterday."
    resolved = SymbolicDateParser.resolve_date(text, current_pred_date="2026-01-01")
    assert resolved == "2026-07-09"

    text_kal = "Received: 2026-08-15 (Sat)\nSource: app-chat\n\nKal payment kiya tha."
    resolved_kal = SymbolicDateParser.resolve_date(text_kal, current_pred_date="2026-01-01")
    assert resolved_kal == "2026-08-14"


def test_resolve_date_parso_day_before_yesterday():
    text = "Received: 2026-07-10 (Fri)\nSource: app-chat\n\nParso transfer kiya tha."
    resolved = SymbolicDateParser.resolve_date(text, current_pred_date="2026-01-01")
    assert resolved == "2026-07-08"


def test_resolve_date_indian_format():
    text = "Received: 2026-07-07 (Tue)\nSource: app-chat\n\nDated 27/06/2026."
    resolved = SymbolicDateParser.resolve_date(text, current_pred_date="2026-01-01")
    assert resolved == "2026-06-27"


def test_resolve_date_month_day_format():
    text = "Received: 2026-07-16 (Thu)\nSource: app-chat\n\nTransaction date Jul 15."
    resolved = SymbolicDateParser.resolve_date(text, current_pred_date="2026-01-01")
    assert resolved == "2026-07-15"

    text_d_m = "Received: 2026-07-01 (Wed)\nSource: email\n\nTransaction date 16 Jun."
    resolved_d_m = SymbolicDateParser.resolve_date(text_d_m, current_pred_date="2026-01-01")
    assert resolved_d_m == "2026-06-16"


def test_resolve_date_retains_valid_non_dummy_prediction():
    text = "Received: 2026-08-01 (Sat)\nSource: email\n\nPayment on 2026-07-28."
    resolved = SymbolicDateParser.resolve_date(text, current_pred_date="2026-07-28")
    assert resolved == "2026-07-28"
