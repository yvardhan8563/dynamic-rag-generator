from uuid import uuid4

from starlette.responses import JSONResponse


class BodyLimitMiddleware:
    """Enforce an aggregate limit before multipart parsing, including chunked bodies."""
    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        # Buffer at most the bounded upload size, so parser errors cannot mask a 413.
        # This trades ~11MB per concurrent request for simple, reliable enforcement.
        messages = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                request_id = scope.get("state", {}).get("request_id") or str(uuid4())
                response = JSONResponse(status_code=413, headers={"X-Request-ID": request_id}, content={
                    "error": {"code": "request_limit", "message": "Request exceeds the upload size limit",
                              "request_id": request_id}})
                return await response(scope, receive, send)
            messages.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(messages)

        async def replay():
            try:
                return next(iterator)
            except StopIteration:
                return await receive()

        await self.app(scope, replay, send)
