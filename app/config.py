from pathlib import Path
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore", populate_by_name=True)

    data_dir: Path = Path("data")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_upload_mb: int = Field(default=10, ge=1, le=100)
    max_text_chars: int = Field(default=1_000_000, ge=100)
    max_chunks: int = Field(default=10_000, ge=1)
    chunk_tokens: int = Field(default=200, ge=16)
    chunk_overlap: int = Field(default=30, ge=0)
    similarity_threshold: float = Field(default=0.3, ge=-1, le=1)
    context_chars: int = Field(default=12_000, ge=500, le=100_000)
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: SecretStr = SecretStr("")
    gemini_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="", validation_alias="GEMINI_MODEL")
    hf_token: SecretStr = Field(default=SecretStr(""), validation_alias=AliasChoices("HF_TOKEN", "RAG_HF_TOKEN"))
    llm_timeout: float = Field(default=60, gt=0, le=300)

    @model_validator(mode="after")
    def validate_chunking(self):
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("chunk_overlap must be smaller than chunk_tokens")
        if self.llm_base_url and not self.llm_base_url.startswith(("http://", "https://")):
            raise ValueError("llm_base_url must be an HTTP(S) URL")
        return self

    @property
    def llm_configured(self) -> bool:
        if self.is_gemini:
            key = self.gemini_api_key.get_secret_value().strip()
            return bool(key and key != "your_actual_gemini_api_key" and self.generation_model)
        return bool(self.llm_base_url and self.llm_model and (not self.is_huggingface or self.generation_key))

    @property
    def is_gemini(self) -> bool:
        return bool(self.gemini_model or self.gemini_api_key.get_secret_value())

    @property
    def generation_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta/openai" if self.is_gemini else self.llm_base_url

    @property
    def generation_model(self) -> str:
        return (self.gemini_model or "gemini-3.1-flash-lite") if self.is_gemini else self.llm_model

    @property
    def is_huggingface(self) -> bool:
        url = urlsplit(self.llm_base_url)
        return url.scheme == "https" and url.hostname == "router.huggingface.co"

    @property
    def generation_key(self) -> str:
        if self.is_gemini:
            return self.gemini_api_key.get_secret_value().strip()
        # Never implicitly forward HF_TOKEN to an unrelated compatible endpoint.
        if self.is_huggingface and self.hf_token.get_secret_value():
            return self.hf_token.get_secret_value()
        return self.llm_api_key.get_secret_value()
