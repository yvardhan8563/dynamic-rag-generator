import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.errors import AppError


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["answered", "insufficient_evidence"]
    answer: str = Field(min_length=1, max_length=12000)
    source_ids: list[str] = Field(max_length=20)


def validate_answer(raw: dict, sources: list[dict]) -> GeneratedAnswer:
    try:
        result = GeneratedAnswer.model_validate(raw)
    except ValidationError as exc:
        raise AppError(502, "invalid_llm_response", "The answer provider returned invalid structured output") from exc
    known = {s["source_id"] for s in sources}
    cited = set(re.findall(r"\[(S\d+)\]", result.answer))
    supplied = set(result.source_ids)
    if result.status == "answered" and (
        not result.answer.strip() or not supplied or not supplied <= known or cited != supplied
    ):
        raise AppError(502, "invalid_citations", "The generated answer could not be linked to valid sources")
    if result.status == "insufficient_evidence" and (supplied or cited):
        raise AppError(502, "invalid_citations", "The generated abstention contained unexpected citations")
    return result
