import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm_backend import (
    _chat_gemini_sync,
    build_ollama_payload,
    normalize_messages,
    normalize_provider,
    to_gemini_input,
)


class LLMBackendTests(unittest.TestCase):
    def test_normalize_provider_accepts_genai_aliases(self) -> None:
        self.assertEqual(normalize_provider("genai"), "gemini")
        self.assertEqual(normalize_provider("google-genai"), "gemini")
        self.assertEqual(normalize_provider("ollama-local"), "ollama")

    def test_ollama_payload_keeps_system_role(self) -> None:
        payload = build_ollama_payload(
            model="gemma4:12b",
            messages=[
                {"role": "system", "content": "Guide Babouin"},
                {"role": "user", "content": "Bonjour"},
            ],
            think=True,
            options={"num_ctx": 8192},
        )

        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "Guide Babouin")
        self.assertTrue(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], 8192)

    def test_gemini_extracts_native_system_instruction(self) -> None:
        system_instruction, contents = to_gemini_input(
            [
                {"role": "system", "content": "Guide Babouin"},
                {"role": "user", "content": "Bonjour"},
                {"role": "assistant", "content": "Salut"},
            ]
        )

        self.assertEqual(system_instruction, "Guide Babouin")
        self.assertEqual(
            contents,
            [
                {"role": "user", "content": "Bonjour"},
                {"role": "model", "content": "Salut"},
            ],
        )

    def test_gemini_sdk_receives_system_instruction(self) -> None:
        sdk_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=Mock(return_value=SimpleNamespace(text="Salut"))
            ),
            close=Mock(),
        )

        with patch("google.genai.Client", return_value=sdk_client):
            response = _chat_gemini_sync(
                model="gemini-test",
                api_key="test-key",
                messages=[
                    {"role": "system", "content": "Guide Babouin"},
                    {"role": "user", "content": "Bonjour"},
                ],
                temperature=0.5,
                max_tokens=100,
                timeout=30,
            )

        call = sdk_client.models.generate_content.call_args.kwargs
        self.assertEqual(response.text, "Salut")
        self.assertEqual(call["config"].system_instruction, "Guide Babouin")
        self.assertEqual(call["config"].max_output_tokens, 100)
        sdk_client.close.assert_called_once_with()

    def test_messages_reject_unknown_roles(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported message role"):
            normalize_messages([{"role": "developer", "content": "test"}])


if __name__ == "__main__":
    unittest.main()
