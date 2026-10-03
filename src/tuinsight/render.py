"""Turn kernel outputs into a short error, optional traceback, text, or an image."""

from __future__ import annotations

import base64
import binascii
import re
from io import BytesIO

from PIL import Image, UnidentifiedImageError

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_IMAGE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp")


def _text(value: object) -> str:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value)


def _strip_ansi(value: str) -> str:
    return _ANSI.sub("", value)


def error_sentence(output: dict) -> str:
    """One line for an error output. The traceback stays out of this string."""
    ename = str(output.get("ename") or "Error").strip() or "Error"
    evalue = _strip_ansi(str(output.get("evalue") or "")).strip()
    detail = evalue.splitlines()[0] if evalue else ""
    if not detail:
        return ename
    if detail.startswith(ename):
        return detail
    return f"{ename}: {detail}"


def output_traceback(output: dict) -> str:
    traceback = output.get("traceback") or []
    if isinstance(traceback, list):
        text = "\n".join(str(line) for line in traceback)
    else:
        text = str(traceback)
    return _strip_ansi(text)


def plain_text(output: dict) -> str:
    kind = output.get("output_type")
    if kind == "stream":
        return _text(output.get("text", ""))
    if kind in {"display_data", "execute_result"}:
        data = output.get("data")
        if isinstance(data, dict) and "text/plain" in data:
            return _text(data["text/plain"])
    return ""


def _decode_image(value: object) -> Image.Image | None:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        raw = value
    else:
        encoded = _text(value).strip()
        if encoded.startswith("data:") and "," in encoded:
            encoded = encoded.split(",", 1)[1]
        try:
            raw = base64.b64decode(encoded)
        except (ValueError, binascii.Error):
            return None
    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except (ValueError, UnidentifiedImageError, OSError):
        return None
    return image.convert("RGB")


def image_from_output(output: dict) -> Image.Image | None:
    if output.get("output_type") not in {"display_data", "execute_result"}:
        return None
    data = output.get("data")
    if not isinstance(data, dict):
        return None
    for mime in _IMAGE_TYPES:
        if mime in data:
            image = _decode_image(data[mime])
            if image is not None:
                return image
    return None
