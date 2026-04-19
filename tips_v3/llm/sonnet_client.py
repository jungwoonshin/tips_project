"""Shared Sonnet 4.5 client via OpenRouter (OpenAI-compatible).

Provides single calls, k-sample agreement, and a parallel "batch" (concurrent
futures, since OpenRouter doesn't expose Anthropic's Messages Batches API)."""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from tips_v3.config import (
    OPENROUTER_BASE_URL,
    SONNET_MODEL_ID,
    openrouter_api_key,
)

log = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def _strip_json(text: str) -> str:
    m = _JSON_RE.search(text)
    return m.group(1) if m else text.strip()


def parse_json(text: str) -> Any:
    return json.loads(_strip_json(text))


@dataclass
class Message:
    system: str
    user: str


class SonnetClient:
    """Sonnet 4.5 via OpenRouter's OpenAI-compatible chat-completions endpoint."""

    def __init__(self, model: str = SONNET_MODEL_ID, max_workers: int = 16):
        if OpenAI is None:
            raise RuntimeError("openai SDK not installed (required for OpenRouter)")
        self._client = OpenAI(
            api_key=openrouter_api_key(),
            base_url=OPENROUTER_BASE_URL,
        )
        self.model = model
        self._executor = cf.ThreadPoolExecutor(max_workers=max_workers)

    def call(
        self,
        msg: Message,
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        retries: int = 4,
        json_mode: bool = True,
    ) -> str:
        """Call Sonnet. json_mode=True (default) requests structured JSON output
        to prevent malformed JSON from code-heavy fix content. Falls back to
        free-text if the provider rejects response_format."""
        last_exc: Exception | None = None
        use_json_mode = json_mode
        for attempt in range(retries):
            try:
                kwargs = dict(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": msg.system},
                        {"role": "user", "content": msg.user},
                    ],
                    extra_headers={
                        "HTTP-Referer": "https://tips-v3.local",
                        "X-Title": "TIPS v3",
                    },
                )
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self._client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                return choice.message.content or ""
            except Exception as exc:
                last_exc = exc
                # If json_mode was the culprit, drop it and retry without.
                if use_json_mode and "response_format" in str(exc).lower():
                    log.info("provider rejected response_format; retrying without json_mode")
                    use_json_mode = False
                    continue
                delay = 2 ** attempt
                log.warning(
                    "sonnet call failed (attempt %d/%d): %s; backing off %ds",
                    attempt + 1, retries, exc, delay,
                )
                time.sleep(delay)
        raise RuntimeError(f"sonnet call failed after {retries} retries: {last_exc}")

    def k_samples(
        self,
        msg: Message,
        *,
        k: int,
        temperature: float,
        max_tokens: int = 4096,
    ) -> list[str]:
        """k independent samples, run concurrently."""
        futures = [
            self._executor.submit(
                self.call, msg, temperature=temperature, max_tokens=max_tokens
            )
            for _ in range(k)
        ]
        return [f.result() for f in futures]

    def batch(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int = 2048,
    ) -> list[str]:
        """Concurrent execution (OpenRouter has no batch API)."""
        if not messages:
            return []
        futures = [
            self._executor.submit(
                self.call, m, temperature=temperature, max_tokens=max_tokens
            )
            for m in messages
        ]
        out: list[str] = []
        for f in futures:
            try:
                out.append(f.result())
            except Exception as exc:
                log.error("batch item failed: %s", exc)
                out.append("")
        return out
