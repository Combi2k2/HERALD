"""Tests for VLMClient."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from PIL import Image
from pydantic import BaseModel, Field

from services.vlm import VLMClient, image_block, text_block


class _DemoOut(BaseModel):
    label: str = Field(...)
    score: float = Field(..., ge=0.0, le=1.0)


def test_invoke_structured_passes_messages_and_parses_model():
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = _DemoOut(label="ok", score=0.8)
    mock_chat = MagicMock()
    mock_chat.with_structured_output.return_value = mock_structured

    messages = [
        SystemMessage(content="system"),
        HumanMessage(content=[text_block("describe"), image_block(Image.new("RGB", (8, 8)))]),
    ]
    client = VLMClient("test", chat=mock_chat)
    out = client.invoke(messages, schema=_DemoOut)

    assert out.label == "ok"
    mock_structured.invoke.assert_called_once_with(messages)
    human = mock_structured.invoke.call_args.args[0][1]
    assert human.content[1]["type"] == "image"


def test_invoke_text_returns_message_text():
    mock_chat = MagicMock()
    mock_chat.invoke.return_value = AIMessage(content="hello")
    client = VLMClient("test", chat=mock_chat)

    assert client.invoke([SystemMessage(content="x")]) == "hello"
    mock_chat.with_structured_output.assert_not_called()


def test_invoke_structured_raises_on_invoke_error():
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("down")
    mock_chat = MagicMock()
    mock_chat.with_structured_output.return_value = mock_structured
    client = VLMClient("test", chat=mock_chat)

    with pytest.raises(ValueError, match="structured output pipe"):
        client.invoke([SystemMessage(content="x")], schema=_DemoOut)
