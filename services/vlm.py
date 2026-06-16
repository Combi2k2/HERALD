"""Generic LangChain vision chat client."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from typing import Any

from PIL import Image
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_block(img: Image.Image | str, *, fmt: str = "PNG") -> dict[str, Any]:
    if isinstance(img, str):
        return {"type": "image", "url": img}
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return {
        "type": "image",
        "base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "mime_type": f"image/{fmt.lower()}",
    }


class VLMClient:
    """``invoke(messages, schema=...)`` → text or structured Pydantic output."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        chat: Any | None = None,
    ) -> None:
        self._model = chat or init_chat_model(model, temperature=temperature)
        self._cache: dict[type[BaseModel], Any] = {}

    def invoke(
        self,
        messages: Sequence[AnyMessage],
        *,
        schema: type[BaseModel] | None = None,
    ) -> BaseModel | str:
        if schema is None:
            try:
                resp = self._model.invoke(messages)
                text = getattr(resp, "text", None) or resp.content
                return text if isinstance(text, str) else str(text)
            except Exception:
                raise ValueError("Failed to invoke model")

        try:
            if self._cache.get(schema) is None:
                self._cache[schema] = self._model.with_structured_output(schema)
        except Exception:
            raise ValueError("Failed to create structured output pipe")

        try:
            pipe = self._cache[schema]
            resp = pipe.invoke(messages)
        except Exception:
            raise ValueError("Failed to invoke structured output pipe")

        if isinstance(resp, schema):    return resp
        if isinstance(resp, dict):      return schema.model_validate(resp)

        raise ValueError("Failed to convert structured output to Pydantic model")
