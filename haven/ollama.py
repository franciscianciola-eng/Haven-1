"""A small client for Ollama's local HTTP API (https://github.com/ollama/ollama/blob/main/docs/api.md)."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

# Loading a model from disk can take a while before the first byte arrives.
READ_TIMEOUT = 600


class OllamaError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.message = message
        self.status = status


class OllamaUnavailable(OllamaError):
    """Nothing is answering at the Ollama address."""


class ModelNotFound(OllamaError):
    """Ollama is running, but doesn't have the model."""


@dataclass(frozen=True)
class ModelInfo:
    name: str
    tools: bool  # Ollama can pass tool definitions to it and parse its calls
    thinking: bool  # it can reason before answering
    context_length: int | None


class Ollama:
    def __init__(self, host: str, timeout: float = READ_TIMEOUT):
        self.host = host.rstrip("/")
        self.timeout = timeout
        hostname = urlsplit(self.host).hostname or ""
        local = hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or hostname.startswith("127.")
        # A local server must not be reached through an HTTP proxy from the environment.
        handlers = [urllib.request.ProxyHandler({})] if local else []
        self._opener = urllib.request.build_opener(*handlers)

    def chat(self, **payload: Any) -> Iterator[dict]:
        """Stream a chat response: one dict per chunk, the last with "done": true."""
        with self._request("POST", "/api/chat", {**payload, "stream": True}) as response:
            yield from _lines(response)

    def pull(self, model: str) -> Iterator[dict]:
        with self._request("POST", "/api/pull", {"model": model, "stream": True}) as response:
            yield from _lines(response)

    def show(self, model: str) -> dict:
        with self._request("POST", "/api/show", {"model": model}) as response:
            return json.load(response)

    def inspect(self, model: str) -> ModelInfo:
        details = self.show(model)
        capabilities = details.get("capabilities") or []
        context = next(
            (v for k, v in (details.get("model_info") or {}).items() if k.endswith(".context_length")), None
        )
        return ModelInfo(model, "tools" in capabilities, "thinking" in capabilities, context)

    def models(self) -> list[str]:
        with self._request("GET", "/api/tags", None, timeout=10) as response:
            return [m.get("name", "") for m in json.load(response).get("models", [])]

    def version(self, timeout: float = 3) -> str:
        with self._request("GET", "/api/version", None, timeout=timeout) as response:
            return json.load(response).get("version", "?")

    def _request(self, method: str, path: str, body: dict | None, timeout: float | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.host + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            return self._opener.open(request, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as error:
            message = _error_text(error.read())
            if error.code == 404 and "not found" in message.lower():
                raise ModelNotFound(message, 404) from None
            raise OllamaError(message, error.code) from None
        except urllib.error.URLError as error:
            raise OllamaUnavailable(f"nothing answered at {self.host} ({error.reason})") from None
        except (ConnectionError, TimeoutError) as error:
            raise OllamaUnavailable(f"nothing answered at {self.host} ({error})") from None


def _lines(response: Any) -> Iterator[dict]:
    try:
        for raw in response:
            line = raw.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise OllamaError(str(chunk["error"]))
            yield chunk
    except (ConnectionError, TimeoutError, http.client.HTTPException) as error:
        raise OllamaError(f"the connection to Ollama broke off ({error})") from None


def _error_text(body: bytes) -> str:
    text = body.decode("utf-8", "replace").strip()
    try:
        return str(json.loads(text).get("error", text))
    except (json.JSONDecodeError, AttributeError):
        return text or "unknown error"
