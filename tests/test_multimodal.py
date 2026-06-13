"""Multimodal message content — documents/images attached to a turn."""
import base64

import pytest

from aether import Aether, DocumentPart, ImagePart, Message, TextPart
from aether.extensions.llm.fake import FakeProvider
from aether.extensions.memory import SQLiteSessionStore
from aether.llm.contracts import text_of

# --- Contract ------------------------------------------------------------

def test_from_path_infers_media_type_and_encodes(tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    part = DocumentPart.from_path(pdf)
    assert part.media_type == "application/pdf"
    assert base64.b64decode(part.data) == b"%PDF-1.4 fake"


def test_image_from_path_media_type(tmp_path):
    png = tmp_path / "pic.png"
    png.write_bytes(b"\x89PNG\r\n")
    assert ImagePart.from_path(png).media_type == "image/png"


def test_plain_string_content_still_works():
    msg = Message(role="user", content="hello")
    assert msg.content == "hello"
    assert text_of(msg.content) == "hello"


def test_text_of_joins_text_parts_and_ignores_media():
    content = [TextPart(text="summarize "), TextPart(text="this"),
               DocumentPart(media_type="application/pdf", data="QQ==")]
    assert text_of(content) == "summarize this"


def test_content_parts_round_trip_through_serialization():
    """Pydantic must reconstruct the discriminated union from a dict — this is
    what lets sessions persist multimodal turns losslessly."""
    msg = Message(role="user", content=[
        TextPart(text="what is this?"),
        DocumentPart(media_type="application/pdf", data="QUJD"),
    ])
    restored = Message(**msg.model_dump())
    assert isinstance(restored.content[0], TextPart)
    assert isinstance(restored.content[1], DocumentPart)
    assert restored.content[1].media_type == "application/pdf"


# --- Provider translation ------------------------------------------------

def test_openai_translation_maps_parts_to_content_blocks():
    pytest.importorskip("openai")
    from aether.extensions.llm.openai import _openai_content

    blocks = _openai_content([
        TextPart(text="what's in this?"),
        ImagePart(media_type="image/png", data="QUJD"),
        DocumentPart(media_type="application/pdf", data="REVG"),
    ])
    assert blocks[0] == {"type": "text", "text": "what's in this?"}
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"] == "data:image/png;base64,QUJD"
    assert blocks[2]["type"] == "file"
    assert blocks[2]["file"]["file_data"] == "data:application/pdf;base64,REVG"


def test_openai_translation_passes_strings_through():
    pytest.importorskip("openai")
    from aether.extensions.llm.openai import _openai_content
    assert _openai_content("plain") == "plain"


def test_anthropic_translation_maps_parts_to_source_blocks():
    # anthropic SDK is imported lazily, so this helper is import-safe.
    from aether.extensions.llm.anthropic import _anthropic_content

    blocks = _anthropic_content([
        TextPart(text="read this"),
        DocumentPart(media_type="application/pdf", data="REVG"),
        ImagePart(media_type="image/jpeg", data="QUJD"),
    ])
    assert blocks[0] == {"type": "text", "text": "read this"}
    assert blocks[1] == {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": "REVG"},
    }
    assert blocks[2] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
    }


def test_gemini_translation_maps_parts():
    # Skip if the Gemini SDK (or one of its native deps) can't be imported in
    # this environment — importorskip only catches ImportError, but a broken
    # transitive dep can raise other errors.
    try:
        from aether.extensions.llm.gemini import _gemini_parts
    except BaseException:  # noqa: BLE001 - native dep can raise pyo3 PanicException
        pytest.skip("google.genai not importable in this environment")

    parts = _gemini_parts([
        TextPart(text="hi"),
        DocumentPart(media_type="application/pdf", data="REVG"),
    ])
    assert len(parts) == 2


# --- End to end ----------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_accepts_multimodal_turn():
    fake = FakeProvider(canned_response="ok")
    client = Aether(fake)
    await client.complete([Message(role="user", content=[
        TextPart(text="summarize this"),
        DocumentPart(media_type="application/pdf", data="QUJD"),
    ])])
    # The request carried the parts through untouched.
    sent = fake.calls[0].messages[0].content
    assert isinstance(sent, list)
    assert isinstance(sent[1], DocumentPart)


@pytest.mark.asyncio
async def test_multimodal_turn_persists_in_sqlite(tmp_path):
    store = SQLiteSessionStore(str(tmp_path / "m.db"))
    messages = [Message(role="user", content=[
        TextPart(text="what's in the doc?"),
        DocumentPart(media_type="application/pdf", data="QUJD"),
    ])]
    await store.save("s", messages)
    loaded = await store.load("s")
    assert loaded == messages
    assert isinstance(loaded[0].content[1], DocumentPart)
