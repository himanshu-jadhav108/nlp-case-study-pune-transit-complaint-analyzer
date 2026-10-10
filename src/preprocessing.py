"""Small, meaning-preserving normalization helpers."""
import re


def normalize_text(value: str) -> str:
    """Normalize whitespace and case while retaining punctuation and wording."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())
