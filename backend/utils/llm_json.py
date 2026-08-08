"""Repairs for malformed JSON coming back from LLM structured output."""

from typing import Any


def undo_double_escapes(obj: Any) -> Any:
    r"""Recursively undo double-escaped JSON string values from the LLM.

    Gemini structured output sporadically double-escapes newlines and quotes
    inside string values, which ``json.loads`` leaves as literal ``\n`` / ``\"``
    two-character sequences (13/1608 earnings_reports rows as of Jul 2026).
    A string containing literal ``\n`` but no real newline is the fingerprint.
    """
    if isinstance(obj, str) and "\\n" in obj and "\n" not in obj:
        return obj.replace("\\n", "\n").replace('\\"', '"')
    if isinstance(obj, dict):
        return {k: undo_double_escapes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [undo_double_escapes(v) for v in obj]
    return obj


def strip_nuls(obj: Any) -> Any:
    """Recursively remove NUL (0x00) from string values.

    Gemini sporadically emits a NUL escape mid-string. Postgres rejects NUL in
    both TEXT and JSONB, so an otherwise valid payload cannot be stored; the
    byte carries no meaning, and dropping it preserves the rest.
    """
    if isinstance(obj, str):
        return obj.replace("\x00", "")
    if isinstance(obj, dict):
        return {k: strip_nuls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_nuls(v) for v in obj]
    return obj
