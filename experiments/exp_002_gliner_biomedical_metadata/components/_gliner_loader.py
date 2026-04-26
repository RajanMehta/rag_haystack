"""
Module-level singleton cache for GLiNER2 models so the indexing-time and query-time
components in this experiment share weights instead of loading the model twice.
"""

import threading
from typing import Any, Dict

_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {}

DEFAULT_LABELS = ["disease", "drug", "gene", "symptom", "anatomy", "treatment"]


def get_model(model_name: str):
    if model_name in _CACHE:
        return _CACHE[model_name]
    with _LOCK:
        if model_name not in _CACHE:
            from gliner2 import GLiNER2

            _CACHE[model_name] = GLiNER2.from_pretrained(model_name)
    return _CACHE[model_name]


def normalize_entities(result: Any, labels) -> Dict[str, list]:
    """
    GLiNER2.extract_entities returns
        {"entities": {"<label>": [str, ...] | [{"text": str, ...}, ...]}}

    Flatten to {"<label>": [str, ...]} with deduplication preserving first-seen order.
    """
    if not isinstance(result, dict):
        return {}
    raw = result.get("entities") or {}
    out: Dict[str, list] = {}
    for label in labels:
        values = raw.get(label) or []
        seen = set()
        uniq = []
        for v in values:
            text = v.get("text") if isinstance(v, dict) else v
            if not isinstance(text, str):
                continue
            text = text.strip().lower()
            if not text or text in seen:
                continue
            seen.add(text)
            uniq.append(text)
        if uniq:
            out[label] = uniq
    return out
