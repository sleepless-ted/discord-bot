"""Async text backends shared by the Discord bots."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from urllib import parse

import aiohttp


SUPPORTED_MESSAGE_ROLES = {"system", "user", "assistant"}


class LLMBackendError(RuntimeError):
    """Raised when a provider request fails or returns an invalid response."""


@dataclass(frozen=True, slots=True)
class LLMBackendResponse:
    text: str
    raw: Any


def normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    aliases = {
        "google": "gemini",
        "google-genai": "gemini",
        "genai": "gemini",
        "ollama-local": "ollama",
    }
    return aliases.get(normalized, normalized)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not messages:
        raise ValueError("messages must not be empty")

    normalized_messages = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("each message must be a dictionary")

        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("message role and content must be strings")
        if role not in SUPPORTED_MESSAGE_ROLES:
            raise ValueError(f"unsupported message role: {role!r}")

        normalized_messages.append({"role": role, "content": content})

    return normalized_messages


def build_ollama_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    think: bool,
    options: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "stream": False,
        "think": think,
        "messages": normalize_messages(messages),
        "options": dict(options),
    }


async def chat_ollama(
    *,
    model: str,
    base_url: str,
    messages: list[dict[str, Any]],
    think: bool,
    options: dict[str, Any],
    timeout: float = 120.0,
) -> LLMBackendResponse:
    parsed_url = parse.urlparse(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(f"Ollama base_url must be an absolute HTTP(S) URL: {base_url!r}")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    endpoint = f"{base_url.rstrip('/')}/api/chat"
    payload = build_ollama_payload(
        model=model,
        messages=messages,
        think=think,
        options=options,
    )

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.post(endpoint, json=payload) as response:
                if response.status >= 400:
                    details = await response.text()
                    raise LLMBackendError(
                        f"Ollama HTTP {response.status} on {endpoint}: {details[:1000]}"
                    )
                data = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        if isinstance(exc, LLMBackendError):
            raise
        raise LLMBackendError(f"Ollama request failed on {endpoint}: {exc}") from exc

    message = data.get("message")
    if not isinstance(message, dict):
        raise LLMBackendError(f"Unexpected Ollama response: {data!r}")

    return LLMBackendResponse(
        text=str(message.get("content", "") or "").strip(),
        raw=data,
    )


def to_gemini_input(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, str]]]:
    system_parts = []
    contents = []

    for message in normalize_messages(messages):
        if message["role"] == "system":
            system_parts.append(message["content"])
            continue

        contents.append(
            {
                "role": "model" if message["role"] == "assistant" else "user",
                "content": message["content"],
            }
        )

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _chat_gemini_sync(
    *,
    model: str,
    api_key: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int | None,
    timeout: float,
) -> LLMBackendResponse:
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:
        raise LLMBackendError(
            "Gemini requires the google-genai package. Run `pixi install`."
        ) from exc

    system_instruction, neutral_contents = to_gemini_input(messages)
    contents = [
        {
            "role": item["role"],
            "parts": [genai_types.Part.from_text(text=item["content"])],
        }
        for item in neutral_contents
    ]
    config: dict[str, Any] = {"temperature": temperature}
    if system_instruction:
        config["system_instruction"] = system_instruction
    if max_tokens is not None:
        config["max_output_tokens"] = max_tokens

    client = genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(timeout=int(max(10.0, timeout) * 1000)),
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=genai_types.GenerateContentConfig(**config),
        )
        text = str(getattr(response, "text", "") or "").strip()
        return LLMBackendResponse(text=text, raw=response)
    except Exception as exc:
        raise LLMBackendError(f"Gemini generate_content request failed: {exc}") from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


async def chat_gemini(
    *,
    model: str,
    api_key: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int | None,
    timeout: float = 120.0,
) -> LLMBackendResponse:
    if not model.strip():
        raise ValueError("Gemini model must not be empty")
    if not api_key.strip():
        raise ValueError("Gemini api_key must not be empty")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    return await asyncio.to_thread(
        _chat_gemini_sync,
        model=model,
        api_key=api_key,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
