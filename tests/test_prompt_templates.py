import inspect

from haystack_api.prompt_templates import TEMPLATES, get_template


def test_get_template_with_defaults():
    """Test get_template with default parameters."""
    template = get_template("search_pipeline")
    expected = TEMPLATES["search_pipeline"]["default"]["default"]

    assert template == inspect.cleandoc(expected)


def test_get_template_with_specific_model_type():
    """Test get_template with a specific model_type."""
    # First add a test entry to TEMPLATES
    TEMPLATES["test_pipeline"] = {
        "specific_type": {"default": "This is a specific model type template"},
        "default": {"default": "This is a default template"},
    }

    template = get_template("test_pipeline", model_type="specific_type")
    assert template == "This is a specific model type template"


def test_get_template_with_specific_model_name():
    """Test get_template with a specific model_name."""
    # Add a test entry with a specific model name
    TEMPLATES["test_pipeline2"] = {
        "default": {"specific_model": "This is a specific model name template", "default": "This is a default template"}
    }

    template = get_template("test_pipeline2", model_name="specific_model")
    assert template == "This is a specific model name template"


def test_get_template_with_specific_type_and_name():
    """Test get_template with both specific model_type and model_name."""
    # Add a test entry with specific type and name
    TEMPLATES["test_pipeline3"] = {
        "specific_type": {
            "specific_model": "This is a specific type and model template",
            "default": "This is a specific type default template",
        },
        "default": {"default": "This is a default template"},
    }

    template = get_template("test_pipeline3", model_type="specific_type", model_name="specific_model")
    assert template == "This is a specific type and model template"


def test_get_template_fallback_to_default_model_name():
    """Test fallback to default model_name when specific one doesn't exist."""
    # Add a test entry for fallback testing
    TEMPLATES["test_pipeline4"] = {"specific_type": {"default": "This is a default model name template"}}

    template = get_template("test_pipeline4", model_type="specific_type", model_name="nonexistent")
    assert template == "This is a default model name template"


def test_get_template_fallback_to_default_model_type():
    """Test fallback to default model_type when specific one doesn't exist."""
    # Add a test entry for fallback testing
    TEMPLATES["test_pipeline5"] = {"default": {"specific_model": "This is a specific model in default type"}}

    template = get_template("test_pipeline5", model_type="nonexistent", model_name="specific_model")
    assert template == "This is a specific model in default type"


def test_get_template_nonexistent_pipeline():
    """Test with a nonexistent pipeline_name."""
    template = get_template("nonexistent_pipeline")
    assert template == ""


def test_get_template_empty_string():
    """Test with empty strings as parameters."""
    template = get_template("", "", "")
    assert template == ""


def test_get_template_real_examples():
    """Test with real examples from the TEMPLATES dictionary."""
    search_template = get_template("search_pipeline")
    gen_template = get_template("gen_pipeline")

    assert "You are an information-retrieval assistant" in search_template
    assert "You are a specialized utterance generation assistant" in gen_template
