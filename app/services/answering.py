from app.generation.grounding import validate_answer


class AnswerService:
    def __init__(self, documents, llm, settings):
        self.documents, self.llm, self.settings = documents, llm, settings

    @staticmethod
    def insufficient(request_id):
        return {"status": "insufficient_evidence",
                "answer": "The selected collection does not contain enough evidence to answer this question.",
                "citations": [], "request_id": request_id}

    def answer(self, collection_id, question, top_k, request_id):
        hits = self.documents.retrieve(collection_id, question, top_k)
        sources = []
        remaining = self.settings.context_chars
        for hit in hits:
            if len(hit["text"]) > remaining:
                continue  # Keep whole chunks rather than silently removing qualifying context.
            remaining -= len(hit["text"])
            sources.append({"source_id": f"S{len(sources) + 1}", "document_id": hit["document_id"],
                            "filename": hit["filename"], "chunk_id": hit["id"],
                            "location": hit["location"], "excerpt": hit["text"], "score": hit["score"]})
        if not sources:
            return self.insufficient(request_id)
        evidence = [{"source_id": s["source_id"], "text": s["excerpt"]} for s in sources]
        result = validate_answer(self.llm.generate(question, evidence), sources)
        if result.status == "insufficient_evidence":
            return self.insufficient(request_id)
        return {"status": result.status, "answer": result.answer,
                "citations": [s for s in sources if s["source_id"] in result.source_ids],
                "request_id": request_id}
