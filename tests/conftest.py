"""Shared pytest fixtures.

One assistant is built per test session in a temporary folder (its own index
storage and query log) so the tests never touch the project's storage/ or logs/.
The embedding back-end follows SOP_EMBED_BACKEND (default 'auto').
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sop_assistant.assistant import SOPAssistant  # noqa: E402
from sop_assistant.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def tmp_session_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("sop_session")


@pytest.fixture(scope="session")
def config(tmp_session_dir):
    return load_config(storage_dir=tmp_session_dir / "storage", log_file=tmp_session_dir / "logs" / "query_log.jsonl",
                       llm_backend="none")


@pytest.fixture(scope="session")
def assistant(config):
    return SOPAssistant(config, rebuild=True)
