from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CollectionRead(BaseModel):
    id: str
    name: str
    created_at: str
    document_count: int
    chunk_count: int


class DocumentRead(BaseModel):
    id: str
    collection_id: str
    filename: str
    created_at: str


class UploadResult(BaseModel):
    id: str
    collection_id: str
    filename: str
    chunk_count: int
    status: Literal["indexed"]


class Citation(BaseModel):
    source_id: str
    document_id: str
    filename: str
    chunk_id: str
    location: dict[str, int]
    excerpt: str
    score: float


class Answer(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    citations: list[Citation]
    request_id: str


class CollectionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name cannot be blank")
        return value.strip()


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question cannot be blank")
        return value.strip()
