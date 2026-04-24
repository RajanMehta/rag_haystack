from unittest.mock import MagicMock

import pytest
from jinja2 import Template

from haystack_api.config import OLLAMA_API_KEY, OLLAMA_MODEL
from haystack_api.pipeline.custom_components.prompt_builder import CustomPromptBuilder


class TestCustomPromptBuilder:
    @pytest.fixture
    def prompt_builder(self):
        # Create a mock template that simply returns the variables as a string
        mock_template = MagicMock(spec=Template)
        mock_template.render = lambda vars: f"Rendered with {', '.join([f'{k}={v}' for k, v in vars.items()])}"

        # Create the CustomPromptBuilder instance
        builder = CustomPromptBuilder(template="dummy_template")
        # Replace the template with our mock
        builder.template = mock_template
        # Mock the _validate_variables method to do nothing
        builder._validate_variables = MagicMock()

        return builder

    def test_basic_rendering(self, prompt_builder):
        # Test basic template rendering
        result = prompt_builder.run(template_variables={"var1": "value1", "var2": "value2"})

        assert "prompt" in result
        assert "generation_kwargs" in result
        assert "Rendered with" in result["prompt"]
        assert "var1=value1" in result["prompt"]
        assert "var2=value2" in result["prompt"]

    def test_generation_kwargs_with_model_and_api_key(self, prompt_builder):
        # Test that model and api_key are properly set in generation_kwargs
        result = prompt_builder.run(template_variables={"model": "test-model", "api_key": "test-key"})

        assert result["generation_kwargs"]["model"] == "test-model"
        assert result["generation_kwargs"]["api_key"] == "test-key"
        assert result["generation_kwargs"]["total_documents"] is None

    def test_with_documents_search_pipeline(self, prompt_builder):
        # Test the case when documents are present (search_pipeline)
        documents = [{"content": "doc1"}, {"content": "doc2"}]
        result = prompt_builder.run(template_variables={"documents": documents})

        assert result["generation_kwargs"]["model"] == OLLAMA_MODEL
        assert result["generation_kwargs"]["api_key"] == OLLAMA_API_KEY
        assert result["generation_kwargs"]["temperature"] == 0.1
        assert result["generation_kwargs"]["top_k"] == 40
        assert result["generation_kwargs"]["top_p"] == 0.9
        assert result["generation_kwargs"]["total_documents"] == 2

    def test_without_documents_gen_pipeline(self, prompt_builder):
        # Test the case when documents are not present (gen_pipeline)
        result = prompt_builder.run(template_variables={"model": "custom-model"})

        assert result["generation_kwargs"]["model"] == "custom-model"
        assert result["generation_kwargs"]["total_documents"] is None

    def test_custom_template_override(self, prompt_builder):
        # Test that providing a custom template works
        mock_env = MagicMock()
        mock_template = MagicMock(spec=Template)
        mock_template.render = lambda vars: "Custom template rendered"
        mock_env.from_string.return_value = mock_template

        prompt_builder._env = mock_env

        result = prompt_builder.run(template="custom_template", template_variables={"var": "value"})

        assert result["prompt"] == "Custom template rendered"
        mock_env.from_string.assert_called_once_with("custom_template")

    def test_kwargs_and_template_variables_combined(self, prompt_builder):
        # Test that kwargs and template_variables are properly combined
        result = prompt_builder.run(
            template_variables={"var1": "from_template_vars", "common": "from_template_vars"},
            var2="from_kwargs",
            common="from_kwargs",
        )

        assert "var1=from_template_vars" in result["prompt"]
        assert "var2=from_kwargs" in result["prompt"]
        assert "common=from_template_vars" in result["prompt"]  # template_variables should override kwargs

    def test_existing_generation_kwargs_preserved(self, prompt_builder):
        # Test that existing generation_kwargs are preserved
        existing_kwargs = {"existing_param": "value"}
        result = prompt_builder.run(template_variables={"generation_kwargs": existing_kwargs, "model": "test-model"})

        assert result["generation_kwargs"]["existing_param"] == "value"
        assert result["generation_kwargs"]["model"] == "test-model"
