"""Requirement 09 - the Streamlit UI renders, answers a question and shows the sources panel."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")


@pytest.fixture(scope="module")
def app(config):
    os.environ["SOP_STORAGE_DIR"] = str(config.storage_dir)
    os.environ["SOP_LOG_FILE"] = str(config.log_file)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    yield at
    os.environ.pop("SOP_STORAGE_DIR", None)
    os.environ.pop("SOP_LOG_FILE", None)


def _ask(at, question):
    at.text_input(key="question").input(question)
    at.button(key="FormSubmitter:ask_form-Ask").click().run()
    assert not at.exception


def test_ui_loads_without_errors(app):
    assert not app.exception
    assert any("SOP Assistant" in t.value for t in app.title)


def test_ui_answers_and_shows_sources(app):
    _ask(app, "What evidence is acceptable to show final inspection was completed?")
    assert any("Grounded" in s.value for s in app.success)
    assert any(e.label.startswith("[1] SOP-QA-002") for e in app.expander)


def test_ui_comparison_and_conflict(app):
    _ask(app, "What is different between incoming and final inspection?")
    assert any("comparison" in s.value for s in app.success)
    _ask(app, "What is the maximum allowed scratch length on a visible Class A surface?")
    assert any("Inconsistent" in w.value for w in app.warning)
