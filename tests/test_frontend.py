from pathlib import Path

import httpx
import pytest

from frontend.api_client import APIError, call


def test_frontend_network_error_is_actionable(monkeypatch):
    def fail(*args, **kwargs):
        raise httpx.ConnectError("private details")
    monkeypatch.setattr(httpx, "request", fail)
    with pytest.raises(APIError, match="Start FastAPI"):
        call("GET", "/collections")


def test_streamlit_empty_and_question_flow(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import frontend.api_client

    state = {"collections": [], "calls": []}
    def api(method, path, **kwargs):
        state["calls"].append((method, path, kwargs))
        if path == "/collections":
            return state["collections"]
        if path.endswith("/documents"):
            return []
        if path.endswith("/query"):
            return {"status": "insufficient_evidence", "answer": "Not enough evidence.", "citations": []}
        raise AssertionError(path)

    monkeypatch.setattr(frontend.api_client, "call", api)
    path = str(Path(__file__).parents[1] / "frontend" / "streamlit_app.py")
    app = AppTest.from_file(path).run()
    assert not app.exception
    assert "Create a collection" in app.info[0].value
    state["collections"] = [{"id": "arbitrary-id", "name": "Runtime collection", "document_count": 0, "chunk_count": 0}]
    app.run()
    assert not app.exception
    app.text_area[0].set_value("What is supported?")
    next(button for button in app.button if button.label == "Ask").click().run()
    assert not app.exception
    assert any(info.value == "Not enough evidence." for info in app.info)
    assert state["calls"][-1][1] == "/collections/arbitrary-id/query"
