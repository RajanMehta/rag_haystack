from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from haystack import Document

from haystack_api.controller.document import (
    delete_documents,
    get_documents,
    get_documents_count,
    update_document_tags,
)
from haystack_api.schema import FilterRequest, TagsRequest


@pytest.fixture
def mock_document_store():
    document_store = MagicMock()
    document_store.document_store = MagicMock()
    document_store.document_store.filter_documents_async = AsyncMock()
    document_store.document_store.wait_result_from_api = True
    return document_store


@pytest.fixture
def mock_qdrant_client():
    client = AsyncMock()
    client.count = AsyncMock()
    client.delete = AsyncMock()
    client.upsert = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_get_documents(mock_document_store):
    # Arrange
    request = FilterRequest(
        institution_id="test_institution",
        collection_name="test_collection",
        filters={"operator": "AND", "conditions": [{"field": "meta.source", "operator": "==", "value": "test.pdf"}]},
    )

    mock_docs = [
        Document(content="test content 1", meta={"source": "test.pdf"}),
        Document(content="test content 2", meta={"source": "test.pdf"}),
    ]
    # Add embeddings to test they get removed
    for doc in mock_docs:
        doc.embedding = [0.1, 0.2, 0.3]

    mock_document_store.document_store.filter_documents_async.return_value = mock_docs

    # Act
    with patch("haystack_api.controller.document.get_document_store", return_value=mock_document_store):
        result = await get_documents(request)

    # Assert
    assert len(result) == 2
    assert result[0].content == "test content 1"
    assert result[1].content == "test content 2"
    assert result[0].embedding is None
    assert result[1].embedding is None
    mock_document_store.document_store.filter_documents_async.assert_called_once_with(filters=request.filters)
    assert mock_document_store.document_store.index == "test_institution_test_collection"


@pytest.mark.asyncio
async def test_get_documents_exception(mock_document_store):
    # Arrange
    request = FilterRequest(institution_id="test_institution", collection_name="test_collection", filters={})

    mock_document_store.document_store.filter_documents_async.side_effect = Exception("Test error")

    # Act & Assert
    with patch("haystack_api.controller.document.get_document_store", return_value=mock_document_store):
        with pytest.raises(HTTPException) as excinfo:
            await get_documents(request)
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "Test error"


@pytest.mark.asyncio
async def test_get_documents_count(mock_qdrant_client):
    # Arrange
    request = FilterRequest(
        institution_id="test_institution",
        collection_name="test_collection",
        filters={"operator": "AND", "conditions": [{"field": "meta.source", "operator": "==", "value": "test.pdf"}]},
    )

    mock_count_response_total = MagicMock()
    mock_count_response_total.count = 10

    mock_count_response_embedded = MagicMock()
    mock_count_response_embedded.count = 5

    mock_qdrant_client.count.side_effect = [mock_count_response_total, mock_count_response_embedded]

    # Act
    with patch("haystack_api.controller.document.get_async_qdrant_client", return_value=mock_qdrant_client):
        with patch("haystack_api.controller.document.convert_filters_to_qdrant", side_effect=lambda x: x):
            result = await get_documents_count(request)

    # Assert
    assert mock_qdrant_client.count.call_count == 2
    assert result == {"total_docs": 10, "embedded_docs": 5}


@pytest.mark.asyncio
async def test_get_documents_count_exception(mock_qdrant_client):
    # Arrange
    request = FilterRequest(institution_id="test_institution", collection_name="test_collection", filters={})

    mock_qdrant_client.count.side_effect = Exception("Test error")

    # Act & Assert
    with patch("haystack_api.controller.document.get_async_qdrant_client", return_value=mock_qdrant_client):
        with patch("haystack_api.controller.document.convert_filters_to_qdrant", side_effect=lambda x: x):
            with pytest.raises(HTTPException) as excinfo:
                await get_documents_count(request)
            assert excinfo.value.status_code == 500
            assert excinfo.value.detail == "Test error"


@pytest.mark.asyncio
async def test_delete_documents(mock_qdrant_client):
    # Arrange
    request = FilterRequest(
        institution_id="test_institution",
        collection_name="test_collection",
        filters={"operator": "AND", "conditions": [{"field": "meta.source", "operator": "==", "value": "test.pdf"}]},
    )

    # Act
    with patch("haystack_api.controller.document.get_async_qdrant_client", return_value=mock_qdrant_client):
        with patch("haystack_api.controller.document.convert_filters_to_qdrant", side_effect=lambda x: x):
            response = await delete_documents(request)

    # Assert
    mock_qdrant_client.delete.assert_called_once_with(
        collection_name="test_institution_test_collection", points_selector=request.filters
    )
    assert response.body == b'{"success":true}'


@pytest.mark.asyncio
async def test_delete_documents_exception(mock_qdrant_client):
    # Arrange
    request = FilterRequest(institution_id="test_institution", collection_name="test_collection", filters={})

    mock_qdrant_client.delete.side_effect = Exception("Test error")

    # Act & Assert
    with patch("haystack_api.controller.document.get_async_qdrant_client", return_value=mock_qdrant_client):
        with patch("haystack_api.controller.document.convert_filters_to_qdrant", side_effect=lambda x: x):
            with pytest.raises(HTTPException) as excinfo:
                await delete_documents(request)
            assert excinfo.value.status_code == 500
            assert excinfo.value.detail == "Test error"


@pytest.mark.asyncio
async def test_update_document_tags(mock_document_store, mock_qdrant_client):
    # Arrange
    request = TagsRequest(
        institution_id="test_institution",
        collection_name="test_collection",
        filters={"operator": "AND", "conditions": [{"field": "meta.source", "operator": "==", "value": "test.pdf"}]},
        tags=["tag1", "tag2"],
    )

    mock_docs = [
        Document(content="test content 1", meta={"source": "test.pdf"}),
        Document(content="test content 2", meta={"source": "test.pdf"}),
    ]

    mock_document_store.document_store.filter_documents_async.return_value = mock_docs

    # Act
    with patch("haystack_api.controller.document.get_document_store", return_value=mock_document_store):
        with patch("haystack_api.controller.document.get_async_qdrant_client", return_value=mock_qdrant_client):
            with patch("haystack_api.controller.document.convert_haystack_documents_to_qdrant_points", return_value=[]):
                response = await update_document_tags(request)

    # Assert
    mock_document_store.document_store.filter_documents_async.assert_called_once_with(filters=request.filters)
    assert mock_docs[0].meta["tags"] == ["tag1", "tag2"]
    assert mock_docs[1].meta["tags"] == ["tag1", "tag2"]
    mock_qdrant_client.upsert.assert_called_once()
    assert response.body == b'{"success":true}'


@pytest.mark.asyncio
async def test_update_document_tags_exception(mock_document_store):
    # Arrange
    request = TagsRequest(
        institution_id="test_institution", collection_name="test_collection", filters={}, tags=["tag1", "tag2"]
    )

    mock_document_store.document_store.filter_documents_async.side_effect = Exception("Test error")

    # Act & Assert
    with patch("haystack_api.controller.document.get_document_store", return_value=mock_document_store):
        with pytest.raises(HTTPException) as excinfo:
            await update_document_tags(request)
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "Test error"
