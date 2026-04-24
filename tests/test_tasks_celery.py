from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from haystack import Document
from haystack.core.errors import PipelineRuntimeError
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client import QdrantClient

from haystack_api.errors import EmbeddingError, ExtractionError, FetchingError
from haystack_api.tasks import (
    _extract_component_name,
    delete_points_from_qdrant,
    index_file,
    index_files,
    index_url,
    index_urls,
    on_task_prerun,
    update_embeddings,
    update_embeddings_dedup,
)


@pytest.fixture
def mock_document_store():
    """Create a mock QdrantDocumentStore."""
    mock_store = MagicMock(spec=QdrantDocumentStore)
    mock_store.index = "default_index"
    return mock_store


@pytest.fixture
def mock_pipeline(mock_document_store):
    pipeline = MagicMock()
    pipeline.get_component.side_effect = lambda name: {
        "documentwriter": MagicMock(document_store=mock_document_store()),
        "documentsplitter": MagicMock(),
        "documentcleaner": MagicMock(),
        "filetyperouter": MagicMock(),
    }.get(name)
    return pipeline


@pytest.fixture
def mock_get_pipelines(mock_pipeline):
    with patch("haystack_api.tasks.get_pipelines") as mock_get_pipelines:
        mock_get_pipelines.return_value = {"test_pipeline": mock_pipeline, "embedding_pipeline": mock_pipeline}
        yield mock_get_pipelines


@pytest.fixture
def mock_default_extract():
    with patch("haystack_api.tasks.default_extract") as mock_extract:
        mock_extract.return_value = Document(content="test content", meta={"uuid": "test-uuid", "tags": []})
        yield mock_extract


@pytest.fixture
def mock_update_embeddings():
    with patch("haystack_api.tasks.update_embeddings") as mock_update:
        mock_update.return_value = {"status": "COMPLETED", "embedded": 1}
        yield mock_update


@pytest.fixture
def mock_get_qdrant_client():
    with patch("haystack_api.tasks.get_qdrant_client") as mock_client:
        mock_client.return_value = MagicMock(spec=QdrantClient)
        yield mock_client


@pytest.fixture
def mock_group():
    with patch("haystack_api.tasks.group") as mock_group:
        mock_result = MagicMock()
        mock_group.return_value.return_value = mock_result
        yield mock_group


@pytest.fixture
def mock_chord():
    with patch("haystack_api.tasks.chord") as mock_chord:
        mock_result = MagicMock()
        mock_chord.return_value.return_value = mock_result
        yield mock_chord


class TestOnTaskPrerun:
    def test_on_task_prerun_basic(self):
        sender = MagicMock()
        task = MagicMock(name="test_task")
        task_id = "test_id"
        args = []
        kwargs = {}

        with patch("structlog.contextvars.bind_contextvars") as mock_bind:
            on_task_prerun(sender, task_id, task, args, kwargs)
            mock_bind.assert_called_once_with(task_id=task_id, task_name=task.name)

    def test_on_task_prerun_with_structlog_contextvars(self):
        sender = MagicMock()
        task = MagicMock(name="test_task")
        task_id = "test_id"
        args = []
        kwargs = {"structlog_contextvars": {"x_request_id": "req-123", "correlation-id": "corr-456"}}

        with patch("structlog.contextvars.bind_contextvars") as mock_bind:
            on_task_prerun(sender, task_id, task, args, kwargs)

            # First call for task_id and task_name
            assert mock_bind.call_args_list[0] == call(task_id=task_id, task_name=task.name)

            # Second call for the structlog_contextvars
            assert mock_bind.call_args_list[1] == call(request_id="req-123", correlation_id="corr-456")


class TestIndexUrls:
    def test_index_urls_missing_index(self):
        urls = ["http://example.com"]
        uuids = ["test-uuid"]
        params = {}
        pipeline_name = "test_pipeline"

        result = index_urls(urls, uuids, params, pipeline_name)

        assert result["error"]["code"] == "INVALID_INPUT"
        assert result["error"]["type"] == "VALIDATION_ERROR"
        assert result["error"]["message"] == "Request validation failed."
        assert result["error"]["details"]["original_error_message"] == "Index not set in params"

    def test_index_urls_success(self, mock_group):
        urls = ["http://example.com", "http://example.org"]
        uuids = ["uuid1", "uuid2"]
        params = {"index": "test_index", "tags": []}
        pipeline_name = "test_pipeline"

        task = MagicMock()
        task.request.id = "parent-task-id"

        with patch("haystack_api.tasks.index_url") as mock_index_url:
            mock_index_url.s.side_effect = lambda *args, **kwargs: MagicMock()

            result = index_urls(urls, uuids, params, pipeline_name)

            assert mock_index_url.s.call_count == 2
            assert mock_group.called
            assert result == mock_group.return_value.return_value


class TestIndexUrl:
    def test_index_url_pipeline_not_found(self, mock_get_pipelines):
        url = "http://example.com"
        uuid = "test-uuid"
        params = {"index": "test_index", "tags": []}
        pipeline_name = "non_existent_pipeline"

        with pytest.raises(ValueError, match="Pipeline 'non_existent_pipeline' not found"):
            index_url(url, uuid, params, pipeline_name, [])

    def test_index_url_success(self, mock_get_pipelines, mock_pipeline, mock_default_extract, mock_update_embeddings):
        url = "http://example.com"
        uuid = "test-uuid"
        params = {
            "index": "test_index",
            "preprocessor_params": {
                "split_by": "word",
                "split_length": 10,
                "split_overlap": 2,
                "split_respect_sentence_boundary": True,
                "remove_substrings": [],
            },
            "tags": [],
        }
        pipeline_name = "test_pipeline"
        css_selectors = [{"selector": "div.content", "type": "content"}]

        result = index_url(url, uuid, params, pipeline_name, css_selectors)

        mock_default_extract.assert_called_once_with(url, css_selectors, uuid)
        mock_pipeline.run.assert_called_once()
        mock_update_embeddings.assert_called_once_with(uuid=uuid, index=params["index"])
        assert result == {"indexing": mock_update_embeddings.return_value}

    def test_index_url_fetching_error_no_content(self, mock_get_pipelines, mock_default_extract):
        url = "invalid-url"
        uuid = "test-uuid"
        params = {
            "index": "test_index",
            "preprocessor_params": {
                "split_by": "word",
                "split_length": 100,
                "split_overlap": 10,
                "split_respect_sentence_boundary": True,
                "remove_substrings": [],
            },
            "tags": [],
        }
        pipeline_name = "test_pipeline"
        css_selectors = []

        mock_get_pipelines.return_value = {pipeline_name: MagicMock()}

        mock_default_extract.side_effect = FetchingError(
            source_info=url, message="Failed to fetch URL content", original_error=Exception("Mocked error")
        )

        mock_self = MagicMock()
        mock_self.request.id = "mock-task-id"

        result = index_url(url, uuid, params, pipeline_name, css_selectors)

        assert result["error"]["type"] == "FETCHING_ERROR"
        assert result["error"]["details"]["original_error_message"] == "Mocked error"

    def test_index_url_extraction_error(self, mock_get_pipelines, mock_default_extract):
        url = "http://example.com"
        uuid = "test-uuid"
        params = {
            "index": "test_index",
            "preprocessor_params": {
                "split_by": "word",
                "split_length": 10,
                "split_overlap": 2,
                "split_respect_sentence_boundary": True,
                "remove_substrings": [],
            },
            "tags": [],
        }
        pipeline_name = "test_pipeline"
        css_selectors = [{"selector": "div.content", "type": "content"}]

        mock_default_extract.side_effect = ExtractionError(
            code="TEST_CODE",
            stage="EXTRACTION",
            message="Simulated extraction failure",
            source_info=url,
            original_error=RuntimeError("Mock inner error"),
        )

        result = index_url(url, uuid, params, pipeline_name, css_selectors)

        assert result["document_id"] == uuid
        assert result["error"]["code"] == "TEST_CODE"
        assert result["error"]["type"] == "EXTRACTION_ERROR"
        assert result["error"]["message"] == "Simulated extraction failure"
        assert result["error"]["details"]["source_info"] == url

    def test_index_url_pipeline_runtime_error(self, mock_get_pipelines, mock_pipeline, mock_default_extract):
        url = "http://example.com"
        uuid = "test-uuid"
        params = {
            "index": "test_index",
            "preprocessor_params": {
                "split_by": "word",
                "split_length": 10,
                "split_overlap": 2,
                "split_respect_sentence_boundary": True,
                "remove_substrings": [],
            },
            "tags": [],
        }
        pipeline_name = "test_pipeline"

        mock_pipeline.run.side_effect = PipelineRuntimeError("documentsplitter", object, "documentsplitter failed")

        result = index_url(url, uuid, params, pipeline_name, [])

        assert result["document_id"] == uuid
        assert result["error"]["code"] == "PIPELINE_FAILED"
        assert result["error"]["type"] == "PIPELINE_RUNTIME_ERROR"
        assert "pipeline failed during indexing" in result["error"]["message"].lower()


def test_index_url_uncaught_error(mock_default_extract, monkeypatch):
    url = "http://example.com"
    uuid = "test-uuid"
    params = {
        "index": "test_index",
        "preprocessor_params": {
            "split_by": "word",
            "split_length": 10,
            "split_overlap": 2,
            "split_respect_sentence_boundary": True,
            "remove_substrings": [],
        },
        "tags": [],
    }
    pipeline_name = "test_pipeline"

    monkeypatch.setattr("haystack_api.tasks.get_pipelines", lambda: {pipeline_name: MagicMock()})

    def raise_unexpected_error(*args, **kwargs):
        raise RuntimeError("forced unexpected error for testing")

    monkeypatch.setattr("haystack_api.tasks.default_extract", raise_unexpected_error)

    result = index_url(url, uuid, params, pipeline_name, [])

    assert result["document_id"] == uuid
    assert result["error"]["code"] == "UNKNOWN"
    assert result["error"]["type"] == "UNKNOWN_ERROR"
    assert "pipeline failed during indexing" in result["error"]["message"].lower()


class TestIndexFiles:
    def test_index_files_length_mismatch(self):
        files = [MagicMock(), MagicMock()]
        metas = [{"uuid": "uuid1"}]  # Only one meta for two files
        params = {"index": "test_index"}
        pipeline_name = "test_pipeline"

        result = index_files(files, metas, params, pipeline_name)

        assert result["error"]["code"] == "INVALID_INPUT"
        assert result["error"]["type"] == "VALIDATION_ERROR"
        assert result["error"]["details"]["original_error_message"] == "Length of files and metas does not match"

    def test_index_files_missing_index(self):
        files = [MagicMock()]
        metas = [{"uuid": "uuid1"}]
        params = {}  # No index
        pipeline_name = "test_pipeline"

        result = index_files(files, metas, params, pipeline_name)

        assert result["error"]["code"] == "INVALID_INPUT"
        assert result["error"]["type"] == "VALIDATION_ERROR"
        assert result["error"]["details"]["original_error_message"] == "Index not set in params"

    def test_index_files_success(self, mock_group):
        files = [MagicMock(), MagicMock()]
        metas = [{"uuid": "uuid1"}, {"uuid": "uuid2"}]
        params = {"index": "test_index"}
        pipeline_name = "test_pipeline"

        task = MagicMock()
        task.request.id = "parent-task-id"

        with patch("haystack_api.tasks.index_file") as mock_index_file:
            mock_index_file.s.side_effect = lambda *args, **kwargs: MagicMock()

            result = index_files(files, metas, params, pipeline_name)

            assert mock_index_file.s.call_count == 2
            assert mock_group.called
            assert result == mock_group.return_value.return_value


class TestIndexFile:
    def test_index_file_pipeline_not_found(self, mock_get_pipelines):
        file = MagicMock()
        meta = {"uuid": "test-uuid"}
        params = {"index": "test_index"}
        pipeline_name = "non_existent_pipeline"

        with pytest.raises(ValueError, match="Pipeline 'non_existent_pipeline' not found"):
            index_file(file, meta, params, pipeline_name)

    @patch("haystack_api.tasks.update_embeddings")
    def test_index_file_success(self, mock_update, mock_get_pipelines, mock_pipeline):
        file = MagicMock(spec=Path)
        meta = {"uuid": "test-uuid"}
        params = {
            "index": "test_index",
            "preprocessor_params": {
                "split_by": "word",
                "split_length": 10,
                "split_overlap": 2,
                "split_respect_sentence_boundary": True,
                "remove_substrings": [],
            },
        }
        pipeline_name = "test_pipeline"

        mock_update.return_value = {"status": "COMPLETED", "embedded": 1}

        result = index_file(file, meta, params, pipeline_name)

        mock_pipeline.run.assert_called_once()
        file.unlink.assert_called_once()
        mock_update.assert_called_once_with(uuid=meta["uuid"], index=params["index"])
        assert result == {"indexing": mock_update.return_value}


class TestUpdateEmbeddings:
    def test_update_embeddings_pipeline_not_found(self, mock_get_pipelines):
        # Remove embedding_pipeline from the mock
        mock_get_pipelines.return_value = {"test_pipeline": MagicMock()}

        with pytest.raises(ValueError, match="Embedding pipeline not found"):
            update_embeddings("test-uuid", "test_index")

    def test_update_embeddings_document_writer_not_found(self, mock_get_pipelines, mock_pipeline):
        # Make get_component return None for documentwriter
        mock_pipeline.get_component.side_effect = lambda name: None if name == "documentwriter" else MagicMock()

        with pytest.raises(ValueError, match="Document writer not found in the pipeline"):
            update_embeddings("test-uuid", "test_index")

    def test_update_embeddings_no_documents(self, mock_get_pipelines, mock_pipeline):
        document_store = mock_pipeline.get_component("documentwriter").document_store
        document_store.filter_documents.return_value = []

        mock_get_pipelines.return_value = {"embedding_pipeline": mock_pipeline}

        with pytest.raises(ExtractionError) as excinfo:
            update_embeddings("test-uuid", "test_index")

        err = excinfo.value
        assert err.code == "CONTENT_EMPTY"
        assert err.stage == "EXTRACTION"
        assert "no documents available" in err.message.lower()
        assert err.source_info == "test-uuid"

    def test_update_embeddings_success(self, mock_get_pipelines, mock_pipeline, mock_get_qdrant_client):
        document_store = mock_pipeline.get_component("documentwriter").document_store
        documents = [{"content": "doc1"}, {"content": "doc2"}]
        document_store.filter_documents.return_value = documents

        result = update_embeddings("test-uuid", "test_index")

        # Verify the document_store.filter_documents was called with the expected arguments
        document_store.filter_documents.assert_called_once_with(
            {"field": "meta.uuid", "operator": "==", "value": "test-uuid"}
        )
        mock_pipeline.run.assert_called_once_with({"documents": documents})
        mock_get_qdrant_client.return_value.set_payload.assert_called_once()
        assert result == {"status": "COMPLETED", "embedded": 2}

    def test_update_embeddings_pipeline_exception(self, mock_get_pipelines, mock_pipeline):
        document_store = mock_pipeline.get_component("documentwriter").document_store
        documents = [{"content": "bad-doc"}]
        document_store.filter_documents.return_value = documents

        # Make pipeline.run raise an error
        mock_pipeline.run.side_effect = Exception("Simulated pipeline failure")

        # Act & Assert
        with pytest.raises(EmbeddingError) as excinfo:
            update_embeddings("test-uuid", "test_index")

        err = excinfo.value
        assert err.error_type == "EMBEDDING_ERROR"
        assert err.stage == "EMBEDDING"
        assert "simulated pipeline failure" in str(err.original_error).lower()
        assert err.http_status_code == 500

    def test_update_embeddings_qdrant_exception(self, mock_get_pipelines, mock_pipeline, mock_get_qdrant_client):
        document_store = mock_pipeline.get_component("documentwriter").document_store
        documents = [{"content": "doc1"}]
        document_store.filter_documents.return_value = documents

        mock_get_qdrant_client.return_value.set_payload.side_effect = Exception("Qdrant error")

        with pytest.raises(EmbeddingError) as excinfo:
            update_embeddings("test-uuid", "test_index")

        err = excinfo.value
        assert err.code.endswith("_FAILED")
        assert err.error_type == "WRITING_ERROR"
        assert err.stage == "WRITING"
        assert "failed to update embedding payload" in err.message.lower()


class TestDeletePointsFromQdrant:
    def test_delete_points_from_qdrant(self, mock_get_qdrant_client):
        results = {"status": "success"}
        collection_name = "test_collection"
        uuids = ["uuid1", "uuid2"]

        result = delete_points_from_qdrant(results, collection_name, uuids)

        mock_get_qdrant_client.return_value.delete.assert_called_once()
        assert result == results


class TestUpdateEmbeddingsDedup:
    def test_update_embeddings_dedup(self, mock_chord):
        collection_name = "test_collection"
        request_uuids = ["uuid1", "uuid2"]
        batch_uuids = ["uuid1", "uuid2", "uuid3"]

        with patch("haystack_api.tasks.update_embeddings") as mock_update:
            mock_update.s.side_effect = lambda *args, **kwargs: MagicMock()

            result = update_embeddings_dedup(collection_name, request_uuids, batch_uuids)

            assert mock_update.s.call_count == 2
            assert mock_chord.called
            assert result == {
                "callback": mock_chord.return_value.return_value,
                "embeddings": ["update_embeddings:uuid1", "update_embeddings:uuid2"],
            }

    def test_update_embeddings_dedup_with_structlog_contextvars(self, mock_chord):
        collection_name = "test_collection"
        request_uuids = ["uuid1"]
        batch_uuids = ["uuid1"]
        kwargs = {"structlog_contextvars": {"x_request_id": "req-123"}}

        with patch("haystack_api.tasks.update_embeddings") as mock_update:
            mock_update.s.side_effect = lambda *args, **kwargs: MagicMock()

            result = update_embeddings_dedup(collection_name, request_uuids, batch_uuids, **kwargs)

            # Check that structlog_contextvars was removed from kwargs
            mock_update.s.assert_called_once()
            call_kwargs = mock_update.s.call_args[1]
            assert "structlog_contextvars" not in call_kwargs

            assert result == {
                "callback": mock_chord.return_value.return_value,
                "embeddings": ["update_embeddings:uuid1"],
            }


class TestExtractComponentName:
    def test_extract_component_name_found(self):
        msg = (
            "Pipeline failed during indexing: "
            "The following component failed to run:\n"
            "Component name: 'documentsplitter'\n"
            "Component type: 'DocumentSplitter'"
        )
        assert _extract_component_name(msg) == "documentsplitter"
