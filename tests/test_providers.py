"""The provider layer: registry wiring and the Anthropic translation.

The Anthropic tests run against a stub client injected into the provider, so
the request shape and response handling are exercised without any network.
Gemini has no SDK installed in this environment; what can be tested is that
the failure is the explicit "not installed" error rather than a traceback.
"""

from types import SimpleNamespace

from django.test import SimpleTestCase

from ragengine.exceptions import ProviderError, ProviderNotInstalledError, UnknownProviderError
from ragengine.providers import build_provider
from ragengine.providers.anthropic import REFUSAL_TEXT, AnthropicProvider
from ragengine.providers.gemini import GeminiProvider


class RegistryTests(SimpleTestCase):
    def test_known_providers_resolve(self):
        self.assertIsInstance(build_provider("anthropic", "k", "m", 10), AnthropicProvider)
        self.assertIsInstance(build_provider("gemini", "k", "m", 10), GeminiProvider)

    def test_unknown_provider_is_refused(self):
        with self.assertRaises(UnknownProviderError):
            build_provider("openai", "k", "m", 10)


class _StubMessages:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def _response(text="the answer", stop_reason="end_turn"):
    blocks = [SimpleNamespace(type="text", text=text)] if text else []
    return SimpleNamespace(
        content=blocks,
        model="claude-opus-5",
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=12, output_tokens=7),
    )


class AnthropicProviderTests(SimpleTestCase):
    def _provider(self, response=None, error=None):
        provider = AnthropicProvider("sk-test", "claude-opus-5", 256)
        provider._client = SimpleNamespace(messages=_StubMessages(response, error))
        return provider

    def test_complete_maps_request_and_response(self):
        provider = self._provider(_response("hello"))
        completion = provider.complete("SYSTEM", [{"role": "user", "content": "hi"}])

        sent = provider._client.messages.kwargs
        self.assertEqual(sent["model"], "claude-opus-5")
        self.assertEqual(sent["system"], "SYSTEM")
        self.assertEqual(sent["messages"], [{"role": "user", "content": "hi"}])
        self.assertNotIn("temperature", sent)

        self.assertEqual(completion.text, "hello")
        self.assertEqual(completion.input_tokens, 12)
        self.assertEqual(completion.output_tokens, 7)
        self.assertEqual(completion.stop_reason, "end_turn")

    def test_empty_refusal_gets_a_polite_message(self):
        provider = self._provider(_response(text="", stop_reason="refusal"))
        completion = provider.complete("SYSTEM", [{"role": "user", "content": "hi"}])
        self.assertEqual(completion.text, REFUSAL_TEXT)
        self.assertEqual(completion.stop_reason, "refusal")

    def test_sdk_errors_become_provider_errors(self):
        import anthropic
        import httpx

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        error = anthropic.AuthenticationError(
            "bad key",
            response=httpx.Response(401, request=request),
            body=None,
        )
        provider = self._provider(error=error)
        with self.assertRaises(ProviderError):
            provider.complete("SYSTEM", [{"role": "user", "content": "hi"}])


class GeminiProviderTests(SimpleTestCase):
    def test_missing_sdk_is_an_explicit_error(self):
        try:
            import google  # noqa: F401

            self.skipTest("a Google SDK is installed here")
        except ImportError:
            pass
        provider = GeminiProvider("AIza-test", "gemini-2.5-flash", 256)
        with self.assertRaises(ProviderNotInstalledError):
            provider.complete("SYSTEM", [{"role": "user", "content": "hi"}])
