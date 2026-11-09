"""Symbolic Date Grounding and Resolution Engine.

Resolves relative dates (e.g. 'yesterday', 'aaj', 'parso', '3 days ago')
and diverse date formats against the ticket's 'Received: YYYY-MM-DD' header.
Strips quoted email reply lines ('> ...') to prevent date distraction.
"""
from __future__ import annotations

import datetime
import re
from typing import Optional


class SymbolicDateParser:
    """Grounds and normalizes transaction dates using ticket context."""

    MONTHS = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    WEEKDAYS = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    }

    @classmethod
    def strip_quoted_lines(cls, text: str) -> str:
        """Remove quoted reply lines (lines starting with '>') from email tickets."""
        cleaned_lines = [
            line for line in text.splitlines()
            if not line.strip().startswith(">")
        ]
        return "\n".join(cleaned_lines)

    @classmethod
    def extract_received_date(cls, ticket_text: str) -> Optional[datetime.date]:
        """Extract anchor date from 'Received: YYYY-MM-DD' header."""
        match = re.search(r"Received:\s*(\d{4})-(\d{2})-(\d{2})", ticket_text)
        if match:
            year, month, day = map(int, match.groups())
            try:
                return datetime.date(year, month, day)
            except ValueError:
                return None
        return None

    @classmethod
    def resolve_date(
        cls,
        ticket_text: str,
        current_pred_date: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve transaction date from ticket text, grounding relative anchors.
        
        If current_pred_date is a valid date other than the generic dummy fallback
        '2026-01-01', it is retained unless an explicit relative anchor overrides it.
        """
        # Strip quoted email history first to prevent distractors
        active_text = cls.strip_quoted_lines(ticket_text)
        rec_dt = cls.extract_received_date(ticket_text)

        # If model predicted an actual date (other than the generic dummy fallback '2026-01-01'),
        # retain the model's learned extraction
        if current_pred_date is not None and current_pred_date != "2026-01-01":
            return current_pred_date

        rec_dt = cls.extract_received_date(ticket_text)
        if rec_dt is None:
            return current_pred_date

        if rec_dt is not None:
            # 1. Check same-day anchors: 'today', 'aaj'
            if re.search(r"\b(today|aaj)\b", active_text, re.IGNORECASE):
                return rec_dt.strftime("%Y-%m-%d")

            # 2. Check 1-day offset: 'yesterday', 'kal' (past tense context)
            if re.search(r"\b(yesterday|kal)\b", active_text, re.IGNORECASE):
                return (rec_dt - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

            # 3. Check 2-day offset: 'day before yesterday', 'parso'
            if re.search(r"\b(day before yesterday|parso)\b", active_text, re.IGNORECASE):
                return (rec_dt - datetime.timedelta(days=2)).strftime("%Y-%m-%d")

            # 4. Check N-days ago: e.g. '3 days ago', '4 din pehle'
            m_days = re.search(r"\b(\d+)\s*(days? ago|din pehle)\b", active_text, re.IGNORECASE)
            if m_days:
                offset = int(m_days.group(1))
                return (rec_dt - datetime.timedelta(days=offset)).strftime("%Y-%m-%d")

        # 5. Check Indian DD/MM/YYYY or DD-MM-YYYY formats in active text
        m_ddmmyyyy = re.search(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", active_text)
        if m_ddmmyyyy:
            day, month, year = map(int, m_ddmmyyyy.groups())
            try:
                dt = datetime.date(year, month, day)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

        anchor_year = rec_dt.year if rec_dt else 2026

        # 6. Check 'Month DD' (e.g. 'Jul 15', 'August 2nd')
        m_m_d = re.search(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\b",
            active_text,
            re.IGNORECASE,
        )
        if m_m_d:
            mon_str, day_str = m_m_d.groups()
            mon = cls.MONTHS.get(mon_str.lower()[:3])
            day = int(day_str)
            if mon:
                try:
                    dt = datetime.date(anchor_year, mon, day)
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass

        # 7. Check 'DD Month' (e.g. '16 Jun', '2nd July')
        m_d_m = re.search(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
            active_text,
            re.IGNORECASE,
        )
        if m_d_m:
            day_str, mon_str = m_d_m.groups()
            mon = cls.MONTHS.get(mon_str.lower()[:3])
            day = int(day_str)
            if mon:
                try:
                    dt = datetime.date(anchor_year, mon, day)
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass

        # If no relative or explicit date was found and previous date was dummy or None
        return current_pred_date
