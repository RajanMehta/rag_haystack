from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from haystack import AsyncPipeline

from haystack_api.controller.query import check_status, generate, query
from haystack_api.schema import (
    GenerateParams,
    GenerateRequest,
    QueryParams,
    QueryRequest,
)


@pytest.fixture
def mock_pipeline():
    """Create a mock AsyncPipeline for testing."""
    mock = AsyncMock(spec=AsyncPipeline)
    mock.to_dict.return_value = {"components": {"component1": {}, "component2": {}}}
    return mock


@pytest.fixture
def mock_get_pipelines(mock_pipeline):
    """Mock the get_pipelines function to return our mock pipeline."""
    with patch("haystack_api.controller.query.get_pipelines") as mock_get:
        mock_get.return_value = {
            "search_pipeline": mock_pipeline,
            "gen_pipeline": mock_pipeline,
        }
        yield mock_get


@pytest.fixture
def mock_get_template():
    """Mock the get_template function."""
    with patch("haystack_api.controller.query.get_template") as mock_get:
        mock_get.return_value = "mocked template"
        yield mock_get


def test_check_status():
    """Test the check_status function."""
    result = check_status()
    assert result == {"success": True}


@pytest.mark.asyncio
async def test_query_search_pipeline(mock_pipeline, mock_get_pipelines, mock_get_template):
    """Test the query function with search_pipeline."""
    # Setup mock pipeline response
    mock_pipeline.run_async.return_value = {"document_ranker": {"documents": [{"content": "test doc"}]}}

    # Create request
    request = QueryRequest(
        query="test query",
        pipeline_name="search_pipeline",
        institution_id="test_inst",
        collection_name="test_collection",
        params=QueryParams(top_k=5, generate=True, threshold=0.0),
    )

    # Call the function
    response = await query(request)

    # Verify the response
    assert response["query"] == "test query"
    assert "raw_results" in response
    assert "results" in response
    assert "documents" in response["results"]

    # Verify the pipeline was called with correct parameters
    mock_pipeline.run_async.assert_called_once()
    call_args = mock_pipeline.run_async.call_args[1]["data"]
    assert call_args["text_embedder"]["text"] == "test query"
    assert call_args["embedding_retriever"]["top_k"] == 5
    assert call_args["embedding_retriever"]["index"] == "test_inst_test_collection"
    assert "prompt_builder" in call_args


@pytest.mark.asyncio
async def test_query_pipeline_not_found(mock_get_pipelines):
    """Test the query function when pipeline is not found."""
    # Create request with non-existent pipeline
    request = QueryRequest(
        query="test query",
        pipeline_name="nonexistent_pipeline",
        institution_id="test_inst",
        collection_name="test_collection",
    )

    # Expect HTTPException
    with pytest.raises(HTTPException) as excinfo:
        await query(request)

    assert excinfo.value.status_code == 501
    assert "Pipeline is not configured" in excinfo.value.detail


@pytest.mark.asyncio
async def test_generate(mock_pipeline, mock_get_pipelines, mock_get_template):
    """Test the generate function."""
    # Setup mock pipeline response
    mock_pipeline.run_async.return_value = {
        "openai_generator": {"replies": ["1. Generated utterance 1\n2. Generated utterance 2"]}
    }

    # Create request
    request = GenerateRequest(
        pipeline_name="gen_pipeline",
        params=GenerateParams(
            model_type="openai",
            model_name="gpt-3.5-turbo",
            api_key="test_api_key",
            invocation_context={},
            generation_kwargs={},
        ),
    )

    # Call the function
    response = await generate(request)

    # Verify the response
    assert "query" in response
    assert "raw_results" in response
    assert "results" in response
    assert "generated_utterances" in response["results"]
    assert response["results"]["generated_utterances"] == ["Generated utterance 1", "Generated utterance 2"]

    # Verify the pipeline was called with correct parameters
    mock_pipeline.run_async.assert_called_once()
    call_args = mock_pipeline.run_async.call_args[1]["data"]
    assert "model_type_router" in call_args
    assert "prompt_builder_openai" in call_args
    assert call_args["prompt_builder_openai"]["template"] == "mocked template"


@pytest.mark.asyncio
async def test_generate_pipeline_not_found(mock_get_pipelines):
    """Test the generate function when pipeline is not found."""
    # Create request with non-existent pipeline
    request = GenerateRequest(
        pipeline_name="nonexistent_pipeline",
        params=GenerateParams(
            model_type="openai",
            model_name="gpt-3.5-turbo",
            api_key="test_api_key",
            invocation_context={},
            generation_kwargs={},
        ),
    )

    # Expect HTTPException
    with pytest.raises(HTTPException) as excinfo:
        await generate(request)

    assert excinfo.value.status_code == 501
    assert "Pipeline is not configured" in excinfo.value.detail


def test_generate_params_openai_missing_api_key_raises():
    """
    With OPENAI_API_KEY unset at boot and no per-request key, validating
    GenerateParams for model_type=openai must fail fast (→ 422 at the API layer).
    """
    with patch("haystack_api.schema.OPENAI_API_KEY", None):
        with pytest.raises(ValueError, match="api_key is required"):
            GenerateParams(
                model_type="openai",
                model_name="gpt-3.5-turbo",
                api_key=None,
                invocation_context={},
                generation_kwargs={},
            )


def test_generate_params_openai_falls_back_to_env_key():
    """When OPENAI_API_KEY is set, a missing per-request key is backfilled from env."""
    with patch("haystack_api.schema.OPENAI_API_KEY", "env-key"):
        params = GenerateParams(
            model_type="openai",
            model_name="gpt-3.5-turbo",
            api_key=None,
            invocation_context={},
            generation_kwargs={},
        )
    assert params.api_key == "env-key"
