from unittest.mock import MagicMock, patch

import pytest

from haystack_api.pipeline.custom_components.ollama_generator import (
    CustomOllamaGenerator,
)


@pytest.fixture
def mock_ollama_generator():
    """Fixture to create a mocked OllamaGenerator instance."""
    with patch("haystack_api.pipeline.custom_components.ollama_generator.OllamaGenerator") as mock_parent:
        # Create a mock instance that will be returned by the parent class
        mock_instance = MagicMock()
        mock_instance.run.return_value = {"replies": ["Test response"], "meta": [{"model": "test-model"}]}
        mock_parent.return_value = mock_instance
        yield mock_parent


@pytest.fixture
def custom_generator():
    """Fixture to create a CustomOllamaGenerator instance with mocked client."""
    generator = CustomOllamaGenerator(model="test-model", url="http://test-url:11434")
    generator._client = MagicMock()
    return generator


def test_run_with_zero_documents(custom_generator):
    """Test that run returns empty results when total_documents is 0."""
    result = custom_generator.run(prompt="Test prompt", generation_kwargs={"total_documents": 0})

    assert result == {"replies": [], "meta": []}
    custom_generator._client.generate.assert_not_called()


def test_run_with_model_override(custom_generator):
    """Test that model can be overridden at runtime."""
    # Setup
    original_model = custom_generator.model
    override_model = "override-model"

    # Mock the parent class's run method
    with patch("haystack_integrations.components.generators.ollama.OllamaGenerator.run") as mock_super_run:
        mock_super_run.return_value = {"replies": ["Test response"], "meta": [{"model": override_model}]}

        # Execute
        result = custom_generator.run(
            prompt="Test prompt", generation_kwargs={"model": override_model, "temperature": 0.7}
        )

        # Assert
        mock_super_run.assert_called_once_with(prompt="Test prompt", generation_kwargs={"temperature": 0.7})
        assert custom_generator.model == original_model  # Model should be restored
        assert result == {"replies": ["Test response"], "meta": [{"model": override_model}]}


@patch("haystack_api.pipeline.custom_components.ollama_generator.Client")
def test_run_with_api_key_override(mock_client, custom_generator):
    """Test that API key can be provided at runtime."""
    # Setup
    original_client = custom_generator._client
    api_key = "test-api-key"
    mock_new_client = MagicMock()
    mock_client.return_value = mock_new_client

    # Mock the parent class's run method
    with patch("haystack_integrations.components.generators.ollama.OllamaGenerator.run") as mock_super_run:
        mock_super_run.return_value = {"replies": ["Test response"], "meta": [{"model": "test-model"}]}

        # Execute
        result = custom_generator.run(prompt="Test prompt", generation_kwargs={"api_key": api_key, "temperature": 0.7})

        # Assert
        mock_client.assert_called_once_with(
            host=custom_generator.url, timeout=custom_generator.timeout, headers={"Authorization": f"Bearer {api_key}"}
        )
        mock_super_run.assert_called_once_with(prompt="Test prompt", generation_kwargs={"temperature": 0.7})
        assert custom_generator._client == original_client  # Client should be restored
        assert result == {"replies": ["Test response"], "meta": [{"model": "test-model"}]}


@patch("haystack_api.pipeline.custom_components.ollama_generator.Client")
def test_run_with_model_and_api_key_override(mock_client, custom_generator):
    """Test that both model and API key can be overridden at runtime."""
    # Setup
    original_model = custom_generator.model
    original_client = custom_generator._client
    override_model = "override-model"
    api_key = "test-api-key"
    mock_new_client = MagicMock()
    mock_client.return_value = mock_new_client

    # Mock the parent class's run method
    with patch("haystack_integrations.components.generators.ollama.OllamaGenerator.run") as mock_super_run:
        mock_super_run.return_value = {"replies": ["Test response"], "meta": [{"model": override_model}]}

        # Execute
        result = custom_generator.run(
            prompt="Test prompt", generation_kwargs={"model": override_model, "api_key": api_key, "temperature": 0.7}
        )

        # Assert
        mock_client.assert_called_once_with(
            host=custom_generator.url, timeout=custom_generator.timeout, headers={"Authorization": f"Bearer {api_key}"}
        )
        mock_super_run.assert_called_once_with(prompt="Test prompt", generation_kwargs={"temperature": 0.7})
        assert custom_generator.model == original_model  # Model should be restored
        assert custom_generator._client == original_client  # Client should be restored
        assert result == {"replies": ["Test response"], "meta": [{"model": override_model}]}


def test_exception_handling_restores_original_values(custom_generator):
    """Test that original values are restored even if an exception occurs."""
    # Setup
    original_model = custom_generator.model
    original_client = custom_generator._client
    override_model = "override-model"
    api_key = "test-api-key"

    # Mock the parent class's run method to raise an exception
    with patch("haystack_integrations.components.generators.ollama.OllamaGenerator.run") as mock_super_run:
        mock_super_run.side_effect = Exception("Test exception")

        # Execute
        with pytest.raises(Exception):
            custom_generator.run(prompt="Test prompt", generation_kwargs={"model": override_model, "api_key": api_key})

        # Assert
        assert custom_generator.model == original_model  # Model should be restored
        assert custom_generator._client == original_client  # Client should be restored
