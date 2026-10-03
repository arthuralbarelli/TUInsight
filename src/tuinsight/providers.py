"""Question providers.

OpenAI (``OPENAI_API_KEY``), Sign in with ChatGPT, and Bedrock share one
``propose`` seam. ChatGPT and Bedrock answer only when a real credential is
already present. They do not invent a login or a notebook.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_BEDROCK_MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"

SYSTEM_PROMPT = """You write a short Jupyter notebook that answers the user's question.
Reply with one JSON object and nothing else:
{"cells": [{"cell_type": "markdown", "source": "..."}, {"cell_type": "code", "source": "..."}]}
cell_type is "markdown" or "code".
Use markdown for the explanation and Python code for the work.
When a table helps, display one. When a chart helps, plot it with matplotlib.
Code must run top to bottom. Do not wrap the JSON in markdown fences.
"""


class ProviderError(RuntimeError):
    """The provider could not produce notebook cells."""


class ProviderAuthError(ProviderError):
    """No real credentials. Callers must not substitute a notebook."""


@dataclass(frozen=True)
class ProposedCell:
    cell_type: str
    source: str


def parse_proposed_cells(content: str) -> list[ProposedCell]:
    """Parse the provider's JSON notebook. Raise if it is not usable cells."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderError("The provider did not return notebook JSON.") from exc

    cells = data.get("cells") if isinstance(data, dict) else data
    if not isinstance(cells, list) or not cells:
        raise ProviderError("The provider returned no notebook cells.")

    proposed: list[ProposedCell] = []
    for item in cells:
        if not isinstance(item, dict):
            raise ProviderError("The provider returned a cell that is not an object.")
        cell_type = item.get("cell_type")
        source = item.get("source", "")
        if cell_type not in {"markdown", "code"}:
            raise ProviderError(f"Unsupported cell type: {cell_type!r}.")
        if isinstance(source, list):
            source = "".join(str(part) for part in source)
        if not isinstance(source, str):
            raise ProviderError("A cell source was not text.")
        proposed.append(ProposedCell(cell_type=cell_type, source=source))
    return proposed


def _post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        message = f"The provider returned HTTP {exc.code}."
        if detail:
            message = f"{message} {detail}"
        raise ProviderError(message) from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Could not reach the provider. {exc.reason}") from exc


def _cells_from_openai(token: str, question: str, *, model: str) -> list[ProposedCell]:
    body = _post_json(
        OPENAI_CHAT_URL,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "response_format": {"type": "json_object"},
        },
        {"Authorization": f"Bearer {token}"},
    )
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(
            "The provider response did not include notebook cells."
        ) from exc
    if not isinstance(content, str):
        raise ProviderError("The provider response did not include notebook cells.")
    return parse_proposed_cells(content)


def _token_from_codex(path: Path) -> str:
    """Read an access token written by a real ChatGPT sign-in, if one exists."""
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(data, dict):
        return ""
    tokens = data.get("tokens")
    if isinstance(tokens, dict) and isinstance(tokens.get("access_token"), str):
        return tokens["access_token"].strip()
    access_token = data.get("access_token")
    if isinstance(access_token, str):
        return access_token.strip()
    return ""


class OpenAIProvider:
    """Notebook cells from the OpenAI API, authenticated with ``OPENAI_API_KEY``."""

    label = "OpenAI"

    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        self._api_key = api_key.strip()
        self._model = model or os.environ.get(
            "TUINSIGHT_OPENAI_MODEL", DEFAULT_OPENAI_MODEL
        )

    def available(self) -> bool:
        return bool(self._api_key)

    def status(self) -> str:
        if self.available():
            return "OpenAI (OPENAI_API_KEY)"
        return "OpenAI needs OPENAI_API_KEY."

    async def propose(self, question: str) -> list[ProposedCell]:
        if not self.available():
            raise ProviderAuthError(self.status())
        import asyncio

        return await asyncio.to_thread(
            _cells_from_openai, self._api_key, question, model=self._model
        )


class ChatGPTSignInProvider:
    """Notebook cells using a ChatGPT session the user already signed in for.

    A missing token is an auth error. This provider never pretends the login
    succeeded and never returns stand-in cells.
    """

    label = "ChatGPT"

    def __init__(
        self, *, token: str | None = None, auth_path: Path | None = None
    ) -> None:
        self._token = token
        self._auth_path = auth_path or (Path.home() / ".codex" / "auth.json")

    def _access_token(self) -> str:
        if self._token is not None:
            return self._token.strip()
        env_token = os.environ.get("CHATGPT_ACCESS_TOKEN", "").strip()
        if env_token:
            return env_token
        return _token_from_codex(self._auth_path)

    def available(self) -> bool:
        return bool(self._access_token())

    def status(self) -> str:
        if self.available():
            return "ChatGPT (signed in)"
        return (
            "Sign in with ChatGPT is not set up. "
            "Save a real session token in CHATGPT_ACCESS_TOKEN or ~/.codex/auth.json."
        )

    async def propose(self, question: str) -> list[ProposedCell]:
        token = self._access_token()
        if not token:
            raise ProviderAuthError(self.status())
        import asyncio

        model = os.environ.get("TUINSIGHT_OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        return await asyncio.to_thread(_cells_from_openai, token, question, model=model)


class BedrockProvider:
    """Notebook cells from Amazon Bedrock using AWS credentials already configured.

    There is no simulated sign-in. Without boto3 or AWS credentials, ``propose``
    raises and returns nothing.
    """

    label = "Bedrock"

    def __init__(self, *, model_id: str | None = None) -> None:
        self._model_id = model_id or os.environ.get(
            "TUINSIGHT_BEDROCK_MODEL", DEFAULT_BEDROCK_MODEL
        )

    def _boto3(self):
        try:
            import boto3
        except ImportError:
            return None
        return boto3

    def available(self) -> bool:
        boto3 = self._boto3()
        if boto3 is None:
            return False
        return boto3.Session().get_credentials() is not None

    def status(self) -> str:
        if self._boto3() is None:
            return "Bedrock needs boto3 and AWS credentials. boto3 is not installed."
        if self.available():
            return "Bedrock (AWS credentials)"
        return "Bedrock is not signed in. No AWS credentials were found."

    async def propose(self, question: str) -> list[ProposedCell]:
        if not self.available():
            raise ProviderAuthError(self.status())
        import asyncio

        return await asyncio.to_thread(self._converse, question)

    def _converse(self, question: str) -> list[ProposedCell]:
        boto3 = self._boto3()
        if boto3 is None:
            raise ProviderAuthError(self.status())
        client = boto3.client("bedrock-runtime")
        response = client.converse(
            modelId=self._model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": question}]}],
        )
        try:
            content = response["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Bedrock did not return notebook cells.") from exc
        if not isinstance(content, str):
            raise ProviderError("Bedrock did not return notebook cells.")
        return parse_proposed_cells(content)


def select_provider(name: str | None = None):
    """Pick a provider. An explicit name is returned even when it is not signed in."""
    providers = {
        "openai": OpenAIProvider(),
        "chatgpt": ChatGPTSignInProvider(),
        "bedrock": BedrockProvider(),
    }
    if name:
        try:
            return providers[name]
        except KeyError as exc:
            known = ", ".join(providers)
            raise ProviderError(
                f"Unknown provider {name!r}. Choose one of: {known}."
            ) from exc
    for provider in providers.values():
        if provider.available():
            return provider
    return providers["openai"]
