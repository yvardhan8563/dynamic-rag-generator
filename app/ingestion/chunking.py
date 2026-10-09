from uuid import uuid4

from app.errors import AppError
from app.ingestion.extractors import Passage


def chunk_passages(passages: list[Passage], tokenizer, size: int, overlap: int, limit: int) -> list[dict]:
    """Keep passage boundaries and exact character slices; overlap only inside a passage."""
    chunks = []
    for passage in passages:
        # Fast tokenizers preserve original text via offset mappings (decode can normalize text).
        offsets = tokenizer(passage.text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        for start in range(0, len(offsets), size - overlap):
            window = offsets[start:start + size]
            if not window:
                continue
            text = passage.text[window[0][0]:window[-1][1]].strip()
            if text:
                chunks.append({"id": str(uuid4()), "text": text, "location": passage.location})
                if len(chunks) > limit:
                    raise AppError(413, "chunk_limit", "Document exceeds the configured chunk limit")
            if start + size >= len(offsets):
                break
    if not chunks:
        raise AppError(422, "no_tokens", "No indexable text found")
    return chunks
