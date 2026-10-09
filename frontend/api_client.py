import os

import httpx


class APIError(Exception):
    pass


def call(method: str, path: str, **kwargs):
    base = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        response = httpx.request(method, base + "/api/v1" + path, timeout=180, **kwargs)
        if response.is_error:
            try:
                error = response.json().get("error", {})
            except ValueError:
                error = {}
            message = error.get("message", "Backend request failed")
            request_id = error.get("request_id", response.headers.get("X-Request-ID"))
            raise APIError(f"{message}" + (f" (request {request_id})" if request_id else ""))
        return None if response.status_code == 204 else response.json()
    except httpx.TimeoutException as exc:
        raise APIError("The request timed out. Refresh the document list before retrying an upload.") from exc
    except httpx.HTTPError as exc:
        raise APIError("Cannot reach the backend. Start FastAPI and check RAG_API_URL.") from exc
    except ValueError as exc:
        raise APIError("The backend returned an unreadable response.") from exc
