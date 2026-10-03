import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tuinsight.providers import (
    BedrockProvider,
    ChatGPTSignInProvider,
    OpenAIProvider,
    ProviderAuthError,
    ProviderError,
    parse_proposed_cells,
    select_provider,
)


def _response(content: str) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(
        {"choices": [{"message": {"content": content}}]}
    ).encode()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


class ParseTests(unittest.TestCase):
    def test_parse_cells_and_fences(self) -> None:
        cells = parse_proposed_cells(
            '```json\n{"cells": [{"cell_type": "code", "source": "print(1)"}]}\n```'
        )
        self.assertEqual(cells[0].cell_type, "code")
        self.assertEqual(cells[0].source, "print(1)")

    def test_invalid_json_raises(self) -> None:
        with self.assertRaises(ProviderError):
            parse_proposed_cells("hello")


class OpenAITests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_key_does_not_call_the_api(self) -> None:
        provider = OpenAIProvider(api_key="")
        self.assertFalse(provider.available())
        with patch("tuinsight.providers.urllib.request.urlopen") as urlopen:
            with self.assertRaises(ProviderAuthError):
                await provider.propose("media")
        urlopen.assert_not_called()

    async def test_api_key_is_sent_and_cells_come_back(self) -> None:
        payload = json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": "A média é 10."},
                    {"cell_type": "code", "source": "print(10)"},
                ]
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}):
            provider = OpenAIProvider()
        self.assertTrue(provider.available())
        self.assertIn("OPENAI_API_KEY", provider.status())
        with patch(
            "tuinsight.providers.urllib.request.urlopen",
            return_value=_response(payload),
        ) as urlopen:
            cells = await provider.propose("media das vendas")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer sk-env")
        body = json.loads(request.data.decode())
        self.assertEqual(body["messages"][1]["content"], "media das vendas")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(cells[1].source, "print(10)")


class ChatGPTTests(unittest.IsolatedAsyncioTestCase):
    async def test_without_a_session_it_refuses(self) -> None:
        provider = ChatGPTSignInProvider(
            token="", auth_path=Path("/tmp/tuinsight-missing-auth.json")
        )
        self.assertFalse(provider.available())
        with patch("tuinsight.providers.urllib.request.urlopen") as urlopen:
            with self.assertRaises(ProviderAuthError) as raised:
                await provider.propose("media")
        urlopen.assert_not_called()
        self.assertIn("Sign in with ChatGPT", str(raised.exception))

    async def test_saved_session_token_is_sent(self) -> None:
        payload = json.dumps({"cells": [{"cell_type": "code", "source": "print(1)"}]})
        with patch.dict(os.environ, {"CHATGPT_ACCESS_TOKEN": "chatgpt-token"}):
            provider = ChatGPTSignInProvider()
            self.assertTrue(provider.available())
            with patch(
                "tuinsight.providers.urllib.request.urlopen",
                return_value=_response(payload),
            ) as urlopen:
                cells = await provider.propose("plote")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer chatgpt-token")
        self.assertEqual(cells[0].source, "print(1)")


class BedrockTests(unittest.IsolatedAsyncioTestCase):
    async def test_without_credentials_it_refuses(self) -> None:
        provider = BedrockProvider()
        with patch.object(provider, "available", return_value=False):
            with self.assertRaises(ProviderAuthError):
                await provider.propose("media")

    async def test_converse_is_called_when_credentials_exist(self) -> None:
        provider = BedrockProvider()
        client = MagicMock()
        client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": json.dumps(
                                {"cells": [{"cell_type": "code", "source": "print(1)"}]}
                            )
                        }
                    ]
                }
            }
        }
        fake_boto3 = MagicMock()
        fake_boto3.client.return_value = client
        with (
            patch.object(provider, "available", return_value=True),
            patch.object(provider, "_boto3", return_value=fake_boto3),
        ):
            cells = await provider.propose("plote a média")
        client.converse.assert_called_once()
        sent = client.converse.call_args.kwargs["messages"][0]["content"][0]["text"]
        self.assertEqual(sent, "plote a média")
        self.assertEqual(cells[0].source, "print(1)")


class SelectTests(unittest.TestCase):
    def test_openai_key_wins(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False),
            patch.object(ChatGPTSignInProvider, "available", return_value=True),
            patch.object(BedrockProvider, "available", return_value=True),
        ):
            provider = select_provider(None)
        self.assertIsInstance(provider, OpenAIProvider)

    def test_nothing_signed_in_stays_on_openai_and_does_not_invent_cells(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False),
            patch.object(ChatGPTSignInProvider, "available", return_value=False),
            patch.object(BedrockProvider, "available", return_value=False),
        ):
            provider = select_provider(None)
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertFalse(provider.available())

    def test_unknown_provider_raises(self) -> None:
        with self.assertRaises(ProviderError):
            select_provider("nope")


if __name__ == "__main__":
    unittest.main()
