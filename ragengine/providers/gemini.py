"""Gemini, through Google's ``google-genai`` SDK.

Two SDK generations are in the wild: the current ``google-genai`` package
(``from google import genai``) and the legacy ``google-generativeai``. This
module prefers the current one and falls back to the legacy one, so whichever
a deployment happens to have installed just works. Neither is a hard
dependency of the engine — a Claude-only deployment never imports Google code.

Role mapping is the one real translation: our neutral ``assistant`` role is
Gemini's ``model``, and the system prompt travels as ``system_instruction``
rather than as a message.
"""

from __future__ import annotations

from typing import Dict, List

from ..exceptions import ProviderError, ProviderNotInstalledError
from .base import ChatProvider, Completion

_INSTALL_HINT = (
    "No Gemini SDK is installed. Run: pip install google-genai "
    "(or the legacy google-generativeai)"
)


def _to_gemini_contents(messages: List[Dict[str, str]]) -> List[dict]:
    return [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
    ]


class GeminiProvider(ChatProvider):
    name = "gemini"

    def complete(self, system: str, messages: List[Dict[str, str]]) -> Completion:
        try:
            from google import genai  # the current SDK
        except ImportError:
            return self._complete_legacy(system, messages)
        return self._complete_genai(genai, system, messages)

    # -- current SDK (google-genai) -----------------------------------------

    def _complete_genai(self, genai, system: str, messages: List[Dict[str, str]]) -> Completion:
        from google.genai import errors, types

        client = genai.Client(api_key=self.api_key)
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=_to_gemini_contents(messages),
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=self.max_tokens,
                ),
            )
        except errors.APIError as exc:
            raise ProviderError(f"Gemini request failed: {exc}") from exc

        usage = getattr(response, "usage_metadata", None)
        return Completion(
            # .text is None when every candidate was filtered; treat as empty.
            text=(getattr(response, "text", None) or ""),
            model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            stop_reason=self._finish_reason(response),
        )

    # -- legacy SDK (google-generativeai) ------------------------------------

    def _complete_legacy(self, system: str, messages: List[Dict[str, str]]) -> Completion:
        try:
            import google.generativeai as legacy
        except ImportError as exc:
            raise ProviderNotInstalledError(_INSTALL_HINT) from exc

        legacy.configure(api_key=self.api_key)
        model = legacy.GenerativeModel(
            self.model,
            system_instruction=system,
            generation_config={"max_output_tokens": self.max_tokens},
        )
        try:
            response = model.generate_content(_to_gemini_contents(messages))
        except Exception as exc:  # the legacy SDK has no stable error hierarchy
            raise ProviderError(f"Gemini request failed: {exc}") from exc

        usage = getattr(response, "usage_metadata", None)
        try:
            text = response.text or ""
        except ValueError:  # raised when the response was safety-filtered
            text = ""
        return Completion(
            text=text,
            model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            stop_reason=self._finish_reason(response),
        )

    @staticmethod
    def _finish_reason(response) -> str:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return "no_candidates"
        reason = getattr(candidates[0], "finish_reason", None)
        return str(getattr(reason, "name", reason or "")).lower() or None
