"""Exercise a running backend with synthetic data and real configured models.

Makes up to two paid LLM requests. Deletes only the collections it creates.
Never reads or prints credentials; the running backend handles authentication.
"""
import argparse
import json
from uuid import uuid4

import httpx


def run(base_url: str):
    created = []
    with httpx.Client(base_url=base_url.rstrip("/") + "/api/v1/", timeout=240, trust_env=False) as client:
        def request(method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            if response.is_error:
                try:
                    code = response.json().get("error", {}).get("code", "unknown_error")
                except ValueError:
                    code = "unreadable_response"
                raise RuntimeError(f"{method} {path}: HTTP {response.status_code}, {code}")
            return response.json() if response.status_code != 204 else None

        try:
            for label in ("policy", "isolated"):
                collection = request("POST", "collections", json={"name": f"smoke-{label}-{uuid4().hex[:8]}"})
                created.append(collection["id"])
            cid, other = created
            print("PASS: created two independent collections", flush=True)
            policy = (
                "The Larkspur Research Workshop refund policy allows refunds within 23 days of purchase.\n"
                "Refund requests must be sent to the workshop support team.\n"
                "This policy does not specify how many days refund processing takes.\n"
            )
            document = request("POST", f"collections/{cid}/documents",
                files={"file": ("verification-policy.txt", policy.encode(), "text/plain")})
            if document["chunk_count"] < 1:
                raise RuntimeError("Upload returned no indexed chunks")
            print("PASS: extracted, chunked, embedded, and indexed TXT", flush=True)

            answer = request("POST", f"collections/{cid}/query", json={
                "question": "Within how many days of purchase does the Larkspur Research Workshop allow refunds?"})
            if answer["status"] != "answered" or not any(word in answer["answer"].lower() for word in ("23", "twenty-three", "twenty three")):
                raise RuntimeError("Expected a grounded answer stating 23 days")
            if not answer["citations"]:
                raise RuntimeError("Answer has no citations")
            for source in answer["citations"]:
                if source["document_id"] != document["id"] or source["excerpt"] not in policy:
                    raise RuntimeError("Citation does not match the uploaded evidence")
                if f"[{source['source_id']}]" not in answer["answer"]:
                    raise RuntimeError("Citation is missing from the answer text")
            print("PASS: live LLM answer and matching source citations", flush=True)
            print(json.dumps({"answer": answer["answer"], "citations": answer["citations"]}, indent=2), flush=True)

            unsupported = request("POST", f"collections/{cid}/query", json={
                "question": "How many days does the Larkspur Research Workshop take to process a refund?"})
            if unsupported["status"] != "insufficient_evidence" or unsupported["citations"]:
                raise RuntimeError("The model did not abstain for missing refund-processing evidence")
            print("PASS: abstains when a related question lacks evidence", flush=True)

            isolated = request("POST", f"collections/{other}/query", json={"question": "What is the workshop refund policy?"})
            if isolated["status"] != "insufficient_evidence":
                raise RuntimeError("Empty independent collection returned an answer")
            print("PASS: collection isolation", flush=True)
            request("DELETE", f"collections/{cid}/documents/{document['id']}")
            after = request("POST", f"collections/{cid}/query", json={"question": "What is the workshop refund policy?"})
            if after["status"] != "insufficient_evidence":
                raise RuntimeError("Deleted document remains retrievable")
            print("PASS: deletion removes evidence", flush=True)
        finally:
            for cid in created:
                response = client.delete(f"collections/{cid}")
                if response.status_code not in (204, 404):
                    print(f"Cleanup failed for test collection {cid}: HTTP {response.status_code}", flush=True)
            if created:
                print("Test collection cleanup attempted.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    try:
        run(args.base_url)
    except (RuntimeError, httpx.HTTPError) as exc:
        print(f"FAIL: {exc}" if isinstance(exc, RuntimeError) else "FAIL: backend connection or timeout error", flush=True)
        raise SystemExit(1)
