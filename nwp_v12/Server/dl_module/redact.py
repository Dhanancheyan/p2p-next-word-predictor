"""
redact.py
Lightweight PII scrubber for session text before storage and training.

Replaces email addresses, URLs, IP addresses, and long digit sequences with
placeholder tokens so they do not leak into the n-gram model or logs.
"""
from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"\b[\w.\-+%]+@[\w.\-]+\.\w+\b", re.IGNORECASE)
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Digit runs of 3+ are likely identifiers (phone numbers, IDs, codes).
# 1-2 digit numbers are kept for readability in training data.
_NUM_RE = re.compile(r"\b\d{3,}\b")


def redact_text(text: str) -> str:
    """
    Replace PII patterns in text with safe placeholder tokens.

    Replacements
    ------------
    email  -> <EMAIL>
    URL    -> <URL>
    IP     -> <IP>
    number -> <NUM>   (3+ consecutive digits only)
    """
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _URL_RE.sub("<URL>", text)
    text = _IP_RE.sub("<IP>", text)
    text = _NUM_RE.sub("<NUM>", text)
    return text
