from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from haystack_api.pipeline.custom_components.qdrant_retriever import (
    CustomQdrantRetriever,
)


@pytest.fixture
def mock_document_store():
    """Create a mock QdrantDocumentStore."""
    mock_store = MagicMock(spec=QdrantDocumentStore)
    mock_store.index = "default_index"
    return mock_store


@pytest.fixture
def custom_retriever(mock_document_store):
    """Create a CustomQdrantRetriever with a mock document store."""
    # Mock the parent class initialization
    with patch(
        "haystack_api.pipeline.custom_components.qdrant_retriever.QdrantEmbeddingRetriever.__init__",
        return_value=None,
    ):
        retriever = CustomQdrantRetriever(document_store=mock_document_store)
        retriever._document_store = mock_document_store
        return retriever


class TestCustomQdrantRetriever:
    def test_run_with_default_index(self, custom_retriever, mock_document_store):
        """Test run method without overriding the index."""
        # Mock the parent class run method
        with patch("haystack_api.pipeline.custom_components.qdrant_retriever.QdrantEmbeddingRetriever.run") as mock_run:
            # Setup mock return value
            mock_docs = [Document(content="test doc")]
            mock_run.return_value = mock_docs

            # Call the method
            query_embedding = [0.1, 0.2, 0.3]
            result = custom_retriever.run(query_embedding=query_embedding)

            # Assertions
            assert result == mock_docs
            mock_run.assert_called_once_with(
                query_embedding=query_embedding,
                filters=None,
                top_k=None,
                scale_score=None,
                return_embedding=None,
                score_threshold=None,
                group_by=None,
                group_size=None,
            )
            # Verify index was not changed
            assert mock_document_store.index == "default_index"

    def test_run_with_custom_index(self, custom_retriever, mock_document_store):
        """Test run method with index override."""
        # Mock the parent class run method
        with patch("haystack_api.pipeline.custom_components.qdrant_retriever.QdrantEmbeddingRetriever.run") as mock_run:
            # Setup mock return value
            mock_docs = [Document(content="test doc")]
            mock_run.return_value = mock_docs

            # Call the method with custom index
            query_embedding = [0.1, 0.2, 0.3]
            custom_index = "custom_index"
            result = custom_retriever.run(query_embedding=query_embedding, index=custom_index)

            # Assertions
            assert result == mock_docs
            mock_run.assert_called_once_with(
                query_embedding=query_embedding,
                filters=None,
                top_k=None,
                scale_score=None,
                return_embedding=None,
                score_threshold=None,
                group_by=None,
                group_size=None,
            )
            # Verify index was temporarily changed and then restored
            assert mock_document_store.index == "default_index"

    @pytest.mark.asyncio
    async def test_run_async_with_default_index(self, custom_retriever, mock_document_store):
        """Test run_async method without overriding the index."""
        # Mock the parent class run_async method
        with patch(
            "haystack_api.pipeline.custom_components.qdrant_retriever.QdrantEmbeddingRetriever.run_async"
        ) as mock_run_async:
            # Setup mock return value
            mock_docs = [Document(content="test doc")]
            mock_run_async.return_value = mock_docs

            # Call the method
            query_embedding = [0.1, 0.2, 0.3]
            result = await custom_retriever.run_async(query_embedding=query_embedding)

            # Assertions
            assert result == mock_docs
            mock_run_async.assert_called_once_with(
                query_embedding=query_embedding,
                filters=None,
                top_k=None,
                scale_score=None,
                return_embedding=None,
                score_threshold=None,
                group_by=None,
                group_size=None,
            )
            # Verify index was not changed
            assert mock_document_store.index == "default_index"

    @pytest.mark.asyncio
    async def test_run_async_with_custom_index(self, custom_retriever, mock_document_store):
        """Test run_async method with index override."""
        # Mock the parent class run_async method
        with patch(
            "haystack_api.pipeline.custom_components.qdrant_retriever.QdrantEmbeddingRetriever.run_async"
        ) as mock_run_async:
            # Setup mock return value
            mock_docs = [Document(content="test doc")]
            mock_run_async.return_value = mock_docs

            # Call the method with custom index
            query_embedding = [0.1, 0.2, 0.3]
            custom_index = "custom_index"
            result = await custom_retriever.run_async(query_embedding=query_embedding, index=custom_index)

            # Assertions
            assert result == mock_docs
            mock_run_async.assert_called_once_with(
                query_embedding=query_embedding,
                filters=None,
                top_k=None,
                scale_score=None,
                return_embedding=None,
                score_threshold=None,
                group_by=None,
                group_size=None,
            )
            # Verify index was temporarily changed and then restored
            assert mock_document_store.index == "default_index"

    def test_exception_handling(self, custom_retriever, mock_document_store):
        """Test that the original index is restored even if an exception occurs."""
        # Mock the parent class run method to raise an exception
        with patch(
            "haystack_api.pipeline.custom_components.qdrant_retriever.QdrantEmbeddingRetriever.run",
            side_effect=ValueError("Test exception"),
        ):
            # Call the method with custom index
            query_embedding = [0.1, 0.2, 0.3]
            custom_index = "custom_index"

            # Expect the exception to be raised
            with pytest.raises(ValueError, match="Test exception"):
                custom_retriever.run(query_embedding=query_embedding, index=custom_index)

            # Verify index was restored despite the exception
            assert mock_document_store.index == "default_index"
