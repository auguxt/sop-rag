"""Requirement 08 - monitoring.

Every answered question is appended as one JSON line to logs/query_log.jsonl
with: timestamp, question, question_type, route, answer_status, source_ids,
cited node ids, processing time (ms), embedding model and generator.
"""
from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .models import Answer

_lock = threading.Lock()


class QueryLogger:
    def __init__(self, log_file: Path):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, answer: Answer, embed_model: str) -> Dict:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": answer.question,
            "question_type": answer.question_type,
            "route": answer.route,
            "answer_status": answer.status,
            "source_ids": answer.source_ids,
            "source_nodes": answer.cited_node_ids,
            "processing_ms": round(answer.processing_ms, 1),
            "embed_model": embed_model,
            "generator": answer.generator,
            "warnings": answer.warnings,
        }
        with _lock, self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read(self, last: int = 0) -> List[Dict]:
        if not self.log_file.exists():
            return []
        rows = []
        for line in self.log_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows[-last:] if last else rows

    def summary(self) -> Dict:
        rows = self.read()
        if not rows:
            return {"total": 0}
        times = [r.get("processing_ms", 0) for r in rows]
        return {
            "total": len(rows),
            "by_status": dict(Counter(r.get("answer_status") for r in rows)),
            "by_type": dict(Counter(r.get("question_type") for r in rows)),
            "avg_ms": round(sum(times) / len(times), 1),
            "max_ms": round(max(times), 1),
        }
