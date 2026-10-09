import json

import httpx
import pytest

from app.config import Settings
from app.errors import AppError
from app.generation.client import LLMClient


def hf_settings(**overrides):
    return Settings(llm_base_url="https://router.huggingface.co/v1",
                    llm_model="test/model", _env_file=None, **overrides)


def test_hf_token_loaded_from_env_file(tmp_path):
    env = tmp_path / "settings.env"
    env.write_text("HF_TOKEN=test-only-value\nRAG_LLM_BASE_URL=https://router.huggingface.co/v1\nRAG_LLM_MODEL=test/model\n")
    settings = Settings(_env_file=env)
    assert settings.llm_configured
    assert settings.generation_key == "test-only-value"
    assert "test-only-value" not in repr(settings)


def test_hf_token_required_and_not_forwarded_to_other_hosts():
    assert not hf_settings().llm_configured
    assert hf_settings(hf_token="test-only-value").llm_configured
    settings = Settings(llm_base_url="http://localhost:11434/v1", llm_model="local",
                        hf_token="test-only-value", _env_file=None)
    assert settings.llm_configured
    assert settings.generation_key == ""


def test_hf_authorization_header(monkeypatch):
    original_client = httpx.Client
    def handler(request):
        assert str(request.url) == "https://router.huggingface.co/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-only-value"
        assert json.loads(request.content)["model"] == "test/model"
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"status":"insufficient_evidence","answer":"No evidence","source_ids":[]}'}}]})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    assert LLMClient(hf_settings(hf_token="test-only-value")).generate("Question", [])["status"] == "insufficient_evidence"


@pytest.mark.parametrize("status,code", [(401, "llm_auth_failed"), (403, "llm_auth_failed"),
    (402, "llm_credits_required"), (429, "llm_rate_limited"), (404, "llm_request_rejected"), (500, "llm_unavailable")])
def test_hf_error_mapping_does_not_expose_provider_body(monkeypatch, status, code):
    original_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, text="private-provider-body")), **kwargs))
    with pytest.raises(AppError) as exc:
        LLMClient(hf_settings(hf_token="test-only-value")).generate("Question", [])
    assert exc.value.code == code
    assert "private-provider-body" not in exc.value.message
    assert "test-only-value" not in exc.value.message
