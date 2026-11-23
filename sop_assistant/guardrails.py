"""Output guardrails (requirements 06 and 07).

The assistant must never claim that it inspected a part, approved anything
or released a batch. ``sanitize`` scans every final answer (also LLM output)
for such claims and replaces them.
"""
from __future__ import annotations

import re
from typing import List, Tuple

FORBIDDEN_CLAIMS = [
    r"\bI\s+(have\s+)?(personally\s+)?(inspected|checked|measured|tested|verified)\s+(the|this|your|these|all)?\s*(part|parts|batch|lot|product|sample)",
    r"\bI\s+(have\s+|hereby\s+)?(approved|released|accepted|passed|signed off)\b",
    r"\b(batch|lot|shipment|order)\s+[\w-]*\s*(is|has been)\s+(now\s+)?(approved|released|cleared)\s+(for|by me)",
    r"\b(is|are)\s+(now\s+)?approved for (release|shipment|dispatch)\b",
    r"\byou (may|can) (ship|release|dispatch) (it|the batch|the lot|this)\b",
]

REPLACEMENT = "[removed: the assistant cannot inspect parts or approve/release batches]"


def find_forbidden_claims(text: str) -> List[str]:
    hits = []
    for pattern in FORBIDDEN_CLAIMS:
        for m in re.finditer(pattern, text, re.I):
            hits.append(m.group(0))
    return hits


def sanitize(text: str) -> Tuple[str, List[str]]:
    """Return (clean_text, list_of_removed_claims)."""
    removed = find_forbidden_claims(text)
    for pattern in FORBIDDEN_CLAIMS:
        text = re.sub(pattern, REPLACEMENT, text, flags=re.I)
    return text, removed
