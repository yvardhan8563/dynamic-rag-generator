import json

import httpx
import pytest

from app.config import Settings
from app.errors import AppError
from app.generation.client import LLMClient


def test_gemini_env_selects_google_without_leaking_old_keys(tmp_path, monkeypatch):
    env = tmp_path / "config.env"
    env.write_text("GEMINI_API_KEY=test-google-key\nGEMINI_MODEL=gemini-2.5-flash-lite\nHF_TOKEN=old-hf-key\nRAG_LLM_BASE_URL=https://router.huggingface.co/v1\n")
    settings = Settings(_env_file=env)
    assert settings.llm_configured
    assert "test-google-key" not in repr(settings)
    original = httpx.Client
    def handle(request):
        assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        assert request.headers["authorization"] == "Bearer test-google-key"
        body = json.loads(request.content)
        assert body["model"] == "gemini-2.5-flash-lite"
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "status": "answered", "answer": "23 days [S1]", "source_ids": ["S1"]})}}]})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    assert LLMClient(settings).generate("Question", [])["status"] == "answered"


@pytest.mark.parametrize("key", ["", "your_actual_gemini_api_key"])
def test_missing_or_placeholder_gemini_key_fails_without_hf_fallback(key):
    settings = Settings(gemini_model="gemini-2.5-flash-lite", gemini_api_key=key,
                        hf_token="old-key", llm_base_url="https://router.huggingface.co/v1",
                        llm_model="old-model", _env_file=None)
    assert not settings.llm_configured
    with pytest.raises(AppError) as exc:
        LLMClient(settings).generate("Question", [])
    assert exc.value.code == "llm_unconfigured"
