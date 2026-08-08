"""
llm/groq_client.py
------------------
Stage 2 – Groq LLM client.

Sends conversation history + a new user transcript to the Groq API and
returns the assistant's response text.  All request/response data is
persisted via ArtifactLogger.

DESIGN NOTES
~~~~~~~~~~~~
* Uses the official ``groq`` Python SDK (OpenAI-compatible interface).
* Conversation history is a list of ``{"role": ..., "content": ...}``
  dicts, exactly as required by the Groq chat-completion API.
* The system prompt is prepended to every request but NOT stored in the
  rolling history object (to avoid double-counting tokens).
* The API key is NEVER logged or saved to any artifact.
* Network/API errors are caught and reported; the caller decides whether
  to continue or abort the pipeline turn.
* On the G1, this module is the only component that requires internet.

Python 3.8 compatible.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from logging_utils.logger import get_logger
from logging_utils.artifact_logger import ArtifactLogger

logger = get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when the Groq API call fails non-transiently."""


class GroqClient:
    """
    Thin wrapper around the Groq chat-completion API.

    Parameters
    ----------
    api_key:
        Groq API key (read from env; never hard-code).
    model:
        Groq model identifier, e.g. ``llama-3.1-8b-instant``.
    system_prompt:
        The assistant persona/instructions prepended to every request.
    max_tokens:
        Maximum number of tokens in the LLM response.
    temperature:
        Sampling temperature (0.0–1.0).
    artifact_logger:
        :class:`ArtifactLogger` for saving request/response artifacts.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        artifact_logger: ArtifactLogger,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is empty.  "
                "Set it in .env before running:\n"
                "    GROQ_API_KEY=gsk_<your-key-here>"
            )
        self._model = model
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._artifact_logger = artifact_logger
        self._api_key = api_key

        # Validate the SDK is importable at init time (gives early feedback)
        # but store only the api_key; the client object is created lazily
        # so that tests can mock groq.Groq without the package being installed.
        self._groq_client: Any = None
        logger.info(
            "GroqClient ready (model=%s, max_tokens=%d, temperature=%.2f)",
            model, max_tokens, temperature,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        """Return (and lazily create) the underlying Groq SDK client."""
        if self._groq_client is None:
            try:
                from groq import Groq  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "groq SDK not installed.  Run:\n"
                    "    pip install groq"
                ) from exc
            self._groq_client = Groq(api_key=self._api_key)
        return self._groq_client

    def _build_messages(
        self,
        history: List[Dict[str, str]],
        transcript: str,
    ) -> List[Dict[str, str]]:
        """
        Construct the full messages list for the API call.

        Structure:
            [ {"role": "system",    "content": <system_prompt>},
              {"role": "user",      "content": <older turn>},
              {"role": "assistant", "content": <older response>},
              ...
              {"role": "user",      "content": <current transcript>} ]
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": transcript})
        return messages

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        counter: int,
        transcript: str,
        history: List[Dict[str, str]],
    ) -> str:
        """
        Send *transcript* + *history* to Groq and return the response text.

        Saves request and response artifacts regardless of success/failure.

        Parameters
        ----------
        counter:
            Pipeline turn counter (used for artifact filenames).
        transcript:
            The current user utterance.
        history:
            Rolling conversation history (user/assistant pairs).
            This list is NOT mutated here; the caller manages history.

        Returns
        -------
        str
            The assistant response text.

        Raises
        ------
        LLMError
            If the API call fails (network error, quota, etc.).
        """
        request_ts = time.time()
        messages = self._build_messages(history, transcript)

        # Save request artifact (history without API key)
        self._artifact_logger.log_llm_request(
            counter=counter,
            transcript=transcript,
            history=history,
            system_prompt=self._system_prompt,
            model=self._model,
            request_ts=request_ts,
        )

        logger.info(
            "[LLM #%06d] Sending to Groq (%s) | %d history msgs | %.0f chars",
            counter,
            self._model,
            len(history),
            len(transcript),
        )

        # ── API call ──────────────────────────────────────────────────────
        t_start = time.monotonic()
        response_text: str = ""
        token_usage: Optional[Dict[str, Any]] = None
        error_msg: Optional[str] = None
        http_status: Optional[int] = None

        try:
            completion = self._get_client().chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            response_text = completion.choices[0].message.content or ""
            if hasattr(completion, "usage") and completion.usage:
                token_usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }
            http_status = 200

        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            # Attempt to extract HTTP status from Groq SDK exceptions
            if hasattr(exc, "status_code"):
                http_status = exc.status_code  # type: ignore[attr-defined]
            logger.error("[LLM #%06d] Groq API error: %s", counter, exc)

        latency_s = time.monotonic() - t_start

        # ── Save response artifact ────────────────────────────────────────
        self._artifact_logger.log_llm_response(
            counter=counter,
            response_text=response_text,
            latency_s=latency_s,
            token_usage=token_usage,
            error=error_msg,
            http_status=http_status,
        )

        if error_msg:
            raise LLMError(f"Groq API error: {error_msg}")

        logger.info(
            "[LLM #%06d] Response in %.2f s (%d tokens) | %.60r",
            counter,
            latency_s,
            token_usage.get("completion_tokens", 0) if token_usage else 0,
            response_text,
        )

        return response_text
