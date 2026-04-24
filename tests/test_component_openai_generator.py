from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from haystack.utils import Secret

from haystack_api.pipeline.custom_components.openai_generator import (
    CustomOpenAIGenerator,
)


class MockSecret:
    """Mock class to simulate Haystack's secret handling."""

    def __init__(self, value):
        self._value = value

    def resolve_value(self):
        return self._value


@pytest.fixture
def custom_generator():
    """Fixture to create a CustomOpenAIGenerator instance with mocked client."""
    with patch("haystack_api.pipeline.custom_components.openai_generator.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Create a mock api_key that has resolve_value method
        api_key = MockSecret("test-api-key")

        generator = CustomOpenAIGenerator(model="gpt-3.5-turbo", api_key=api_key)
        generator.client = mock_client
        yield generator


def test_run_basic_functionality(custom_generator):
    """Test basic run functionality without overrides."""
    # Setup
    with patch("haystack.components.generators.openai.OpenAIGenerator.run") as mock_super_run:
        mock_super_run.return_value = {"replies": ["Test response"], "meta": [{"model": "gpt-3.5-turbo"}]}

        # Execute
        result = custom_generator.run(prompt="Test prompt", generation_kwargs={"temperature": 0.7})

        # Assert
        mock_super_run.assert_called_once_with(
            prompt="Test prompt", system_prompt=None, streaming_callback=None, generation_kwargs={"temperature": 0.7}
        )
        assert result == {"replies": ["Test response"], "meta": [{"model": "gpt-3.5-turbo"}]}


def test_run_with_model_override(custom_generator):
    """Test that model can be overridden at runtime."""
    # Setup
    original_model = custom_generator.model
    override_model = "gpt-4"

    with patch("haystack.components.generators.openai.OpenAIGenerator.run") as mock_super_run:
        mock_super_run.return_value = {"replies": ["Test response"], "meta": [{"model": override_model}]}

        # Execute
        result = custom_generator.run(
            prompt="Test prompt", generation_kwargs={"model": override_model, "temperature": 0.7}
        )

        # Assert
        mock_super_run.assert_called_once_with(
            prompt="Test prompt", system_prompt=None, streaming_callback=None, generation_kwargs={"temperature": 0.7}
        )
        assert custom_generator.model == original_model  # Model should be restored
        assert result == {"replies": ["Test response"], "meta": [{"model": override_model}]}


@patch("haystack_api.pipeline.custom_components.openai_generator.OpenAI")
def test_run_with_api_key_override(mock_openai, custom_generator):
    """Test that API key can be provided at runtime."""
    # Setup
    original_api_key = custom_generator.api_key
    original_client = custom_generator.client
    new_api_key = MockSecret("new-test-api-key")
    mock_new_client = MagicMock()
    mock_openai.return_value = mock_new_client

    with patch("haystack.components.generators.openai.OpenAIGenerator.run") as mock_super_run:
        mock_super_run.return_value = {"replies": ["Test response"], "meta": [{"model": "gpt-3.5-turbo"}]}

        # Execute
        result = custom_generator.run(
            prompt="Test prompt", generation_kwargs={"api_key": new_api_key, "temperature": 0.7}
        )

        # Assert
        mock_openai.assert_called_once_with(
            api_key=new_api_key.resolve_value(),
            organization=custom_generator.organization,
            base_url=custom_generator.api_base_url,
            timeout=original_client.timeout,
            max_retries=original_client.max_retries,
        )
        mock_super_run.assert_called_once_with(
            prompt="Test prompt", system_prompt=None, streaming_callback=None, generation_kwargs={"temperature": 0.7}
        )
        assert custom_generator.api_key == original_api_key  # API key should be restored
        assert custom_generator.client == original_client  # Client should be restored
        assert result == {"replies": ["Test response"], "meta": [{"model": "gpt-3.5-turbo"}]}


@patch("haystack_api.pipeline.custom_components.openai_generator.OpenAI")
def test_run_with_model_and_api_key_override(mock_openai, custom_generator):
    """Test that both model and API key can be overridden at runtime."""
    # Setup
    original_model = custom_generator.model
    original_api_key = custom_generator.api_key
    original_client = custom_generator.client
    override_model = "gpt-4"
    new_api_key = MockSecret("new-test-api-key")
    mock_new_client = MagicMock()
    mock_openai.return_value = mock_new_client

    with patch("haystack.components.generators.openai.OpenAIGenerator.run") as mock_super_run:
        mock_super_run.return_value = {"replies": ["Test response"], "meta": [{"model": override_model}]}

        # Execute
        result = custom_generator.run(
            prompt="Test prompt",
            generation_kwargs={"model": override_model, "api_key": new_api_key, "temperature": 0.7},
        )

        # Assert
        mock_openai.assert_called_once_with(
            api_key=new_api_key.resolve_value(),
            organization=custom_generator.organization,
            base_url=custom_generator.api_base_url,
            timeout=original_client.timeout,
            max_retries=original_client.max_retries,
        )
        mock_super_run.assert_called_once_with(
            prompt="Test prompt", system_prompt=None, streaming_callback=None, generation_kwargs={"temperature": 0.7}
        )
        assert custom_generator.model == original_model  # Model should be restored
        assert custom_generator.api_key == original_api_key  # API key should be restored
        assert custom_generator.client == original_client  # Client should be restored
        assert result == {"replies": ["Test response"], "meta": [{"model": override_model}]}


def test_exception_handling_restores_original_values(custom_generator):
    """Test that original values are restored even if an exception occurs."""
    # Setup
    original_model = custom_generator.model
    original_api_key = custom_generator.api_key
    original_client = custom_generator.client
    override_model = "gpt-4"
    new_api_key = MockSecret("new-test-api-key")

    # Mock the parent class's run method to raise an exception
    with patch("haystack.components.generators.openai.OpenAIGenerator.run") as mock_super_run:
        mock_super_run.side_effect = Exception("Test exception")

        # Execute
        with pytest.raises(Exception):
            custom_generator.run(
                prompt="Test prompt", generation_kwargs={"model": override_model, "api_key": new_api_key}
            )

        # Assert
        assert custom_generator.model == original_model  # Model should be restored
        assert custom_generator.api_key == original_api_key  # API key should be restored
        assert custom_generator.client == original_client  # Client should be restored


def test_deferred_init_when_api_key_missing(monkeypatch):
    """
    When OPENAI_API_KEY is unset and the Secret is non-strict, construction must
    succeed (no OpenAI client created) so the service can boot without the key.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gen = CustomOpenAIGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY", strict=False),
        model="gpt-3.5-turbo",
    )
    assert gen.client is None
    assert gen.model == "gpt-3.5-turbo"


def test_run_raises_400_when_no_api_key_available(monkeypatch):
    """
    With the generator in deferred-init (no boot-time key) and no per-request
    api_key override, `run()` must raise HTTP 400 with a clear message instead
    of crashing inside the OpenAI SDK.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gen = CustomOpenAIGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY", strict=False),
        model="gpt-3.5-turbo",
    )
    with pytest.raises(HTTPException) as excinfo:
        gen.run(prompt="hello", generation_kwargs={})
    assert excinfo.value.status_code == 400
    assert "OPENAI_API_KEY is not configured" in excinfo.value.detail
