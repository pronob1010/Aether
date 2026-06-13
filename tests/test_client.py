import pytest
from agartha import Agartha
from agartha.extensions.llm.builder import ProviderConfig
from agartha.extensions.llm.fake import FakeProvider


@pytest.mark.asyncio
async def test_agartha_ask_delegates_to_provider():
    fake = FakeProvider(canned_response="42")
    client = Agartha(fake)
    answer = await client.ask("What is the meaning of life?")
    assert answer == "42"


@pytest.mark.asyncio
async def test_agartha_complete_returns_full_response():
    fake = FakeProvider(canned_response="42")
    client = Agartha(fake)
    response = await client.complete("What is the meaning of life?")
    assert response.text == "42"
    assert response.model == "fake-model"
    assert response.input_tokens > 0
    assert response.output_tokens > 0


@pytest.mark.asyncio
async def test_agartha_complete_forwards_model_and_temperature():
    fake = FakeProvider()
    client = Agartha(fake)
    await client.complete("hi", model="gpt-4o", temperature=0.0)
    sent = fake.calls[0]
    assert sent.model == "gpt-4o"
    assert sent.temperature == 0.0


@pytest.mark.asyncio
async def test_agartha_ask_uses_complete_internally():
    """ask() is a thin wrapper — same request goes through."""
    fake = FakeProvider(canned_response="hello")
    client = Agartha(fake)
    answer = await client.ask("ping")
    assert answer == "hello"
    assert fake.calls[0].messages[0].content == "ping"


def test_agartha_config_kwarg_builds_provider():
    client = Agartha(config=ProviderConfig(name="fake"))
    assert isinstance(client._provider, FakeProvider)


def test_agartha_rejects_both_provider_and_config():
    with pytest.raises(ValueError, match="not both"):
        Agartha(FakeProvider(), config=ProviderConfig(name="fake"))


def test_agartha_unknown_provider_env_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        Agartha()


def test_agartha_missing_api_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        Agartha()


def test_agartha_fake_provider_via_env_needs_no_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    client = Agartha(
        with_retry=False,
        with_circuit_breaker=False,
        with_cost_tracking=False,
    )
    assert isinstance(client._provider, FakeProvider)
