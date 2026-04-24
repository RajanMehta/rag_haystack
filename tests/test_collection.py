import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from grpc import RpcError
from qdrant_client import grpc

from haystack_api.controller.collection import (
    create_collection,
    delete_collection,
    get_collection_info,
)
from haystack_api.schema import CreateCollectionRequest


@pytest.fixture
def mock_async_qdrant_client():
    """Fixture to mock the async Qdrant client"""
    mock_client = AsyncMock()
    mock_client.grpc_collections = AsyncMock()
    return mock_client


@pytest.fixture
def mock_document_store():
    """Fixture to mock the document store"""
    mock_store = MagicMock()
    mock_store.to_dict.return_value = {
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "embedding_dim": 768,
                    "shard_number": 1,
                    "replication_factor": 1,
                    "write_consistency_factor": 1,
                    "on_disk_payload": True,
                    "hnsw_config": None,
                    "optimizers_config": None,
                    "wal_config": None,
                    "quantization_config": None,
                }
            }
        }
    }
    return mock_store


class TestCreateCollection:
    @pytest.mark.asyncio
    @patch("haystack_api.controller.collection.get_async_qdrant_client")
    @patch("haystack_api.controller.collection.get_document_store")
    async def test_create_collection_success(
        self, mock_get_document_store, mock_get_async_qdrant_client, mock_async_qdrant_client, mock_document_store
    ):
        # Setup mocks
        mock_get_async_qdrant_client.return_value = mock_async_qdrant_client
        mock_get_document_store.return_value = mock_document_store

        # Mock the response from Qdrant
        mock_response = MagicMock()
        mock_json_response = '{"result": true, "status": "ok"}'
        mock_async_qdrant_client.grpc_collections.Create.return_value = mock_response

        with patch("haystack_api.controller.collection.MessageToJson", return_value=mock_json_response):
            # Create request
            request = CreateCollectionRequest(institution_id="test_institution", collection_name="test_collection")

            # Call the function
            response = await create_collection(request)

            # Assertions
            assert response == json.loads(mock_json_response)
            mock_async_qdrant_client.grpc_collections.Create.assert_called_once()
            create_args = mock_async_qdrant_client.grpc_collections.Create.call_args[0][0]
            assert create_args.collection_name == "test_institution_test_collection"
            assert create_args.vectors_config.params.size == 768
            assert create_args.vectors_config.params.distance == grpc.Distance.Cosine

    @pytest.mark.asyncio
    @patch("haystack_api.controller.collection.get_async_qdrant_client")
    async def test_create_collection_unexpected_response(self, mock_get_async_qdrant_client, mock_async_qdrant_client):
        # Setup mocks
        mock_get_async_qdrant_client.return_value = mock_async_qdrant_client

        # Mock the error from Qdrant using RpcError instead of UnexpectedResponse
        mock_async_qdrant_client.grpc_collections.Create.side_effect = RpcError("Test error")

        # Create request
        request = CreateCollectionRequest(institution_id="test_institution", collection_name="test_collection")

        # Call the function and expect exception
        with pytest.raises(HTTPException) as exc_info:
            await create_collection(request)

        # Assertions
        assert exc_info.value.status_code == 500
        assert "Test error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_create_collection_missing_params(self):
        # Create request with missing params
        request = CreateCollectionRequest(institution_id="", collection_name="")

        # Call the function and expect exception
        with pytest.raises(HTTPException) as exc_info:
            await create_collection(request)

        # Assertions
        assert exc_info.value.status_code == 500
        assert "The institution_id and/or collection_name are absent" in str(exc_info.value.detail)


class TestGetCollectionInfo:
    @pytest.mark.asyncio
    @patch("haystack_api.controller.collection.get_async_qdrant_client")
    async def test_get_collection_info_success(self, mock_get_async_qdrant_client, mock_async_qdrant_client):
        # Setup mocks
        mock_get_async_qdrant_client.return_value = mock_async_qdrant_client

        # Mock the response from Qdrant
        mock_response = MagicMock()
        mock_json_response = (
            '{"result": {"name": "test_institution_test_collection", "status": "green"}, "status": "ok"}'
        )
        mock_async_qdrant_client.grpc_collections.Get.return_value = mock_response

        with patch("haystack_api.controller.collection.MessageToJson", return_value=mock_json_response):
            # Call the function
            response = await get_collection_info("test_institution", "test_collection")

            # Assertions
            assert response == json.loads(mock_json_response)
            mock_async_qdrant_client.grpc_collections.Get.assert_called_once()
            get_args = mock_async_qdrant_client.grpc_collections.Get.call_args[0][0]
            assert get_args.collection_name == "test_institution_test_collection"

    @pytest.mark.asyncio
    @patch("haystack_api.controller.collection.get_async_qdrant_client")
    async def test_get_collection_info_unexpected_response(
        self, mock_get_async_qdrant_client, mock_async_qdrant_client
    ):
        # Setup mocks
        mock_get_async_qdrant_client.return_value = mock_async_qdrant_client

        # Mock the error from Qdrant using RpcError instead of UnexpectedResponse
        mock_async_qdrant_client.grpc_collections.Get.side_effect = RpcError("Collection not found")

        # Call the function and expect exception
        with pytest.raises(HTTPException) as exc_info:
            await get_collection_info("test_institution", "test_collection")

        # Assertions
        assert exc_info.value.status_code == 500
        assert "Collection not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_collection_info_missing_params(self):
        # Call the function with missing params and expect exception
        with pytest.raises(HTTPException) as exc_info:
            await get_collection_info(None, "test_collection")

        # Assertions
        assert exc_info.value.status_code == 500
        assert "The institution_id and/or collection_name are absent" in str(exc_info.value.detail)


class TestDeleteCollection:
    @pytest.mark.asyncio
    @patch("haystack_api.controller.collection.get_async_qdrant_client")
    async def test_delete_collection_success(self, mock_get_async_qdrant_client, mock_async_qdrant_client):
        # Setup mocks
        mock_get_async_qdrant_client.return_value = mock_async_qdrant_client

        # Mock the response from Qdrant
        mock_response = MagicMock()
        mock_json_response = '{"result": true, "status": "ok"}'
        mock_async_qdrant_client.grpc_collections.Delete.return_value = mock_response

        with patch("haystack_api.controller.collection.MessageToJson", return_value=mock_json_response):
            # Call the function
            response = await delete_collection("test_institution", "test_collection")

            # Assertions
            assert response == json.loads(mock_json_response)
            mock_async_qdrant_client.grpc_collections.Delete.assert_called_once()
            delete_args = mock_async_qdrant_client.grpc_collections.Delete.call_args[0][0]
            assert delete_args.collection_name == "test_institution_test_collection"

    @pytest.mark.asyncio
    @patch("haystack_api.controller.collection.get_async_qdrant_client")
    async def test_delete_collection_unexpected_response(self, mock_get_async_qdrant_client, mock_async_qdrant_client):
        # Setup mocks
        mock_get_async_qdrant_client.return_value = mock_async_qdrant_client

        # Mock the error from Qdrant using RpcError instead of UnexpectedResponse
        mock_async_qdrant_client.grpc_collections.Delete.side_effect = RpcError("Collection not found")

        # Call the function and expect exception
        with pytest.raises(HTTPException) as exc_info:
            await delete_collection("test_institution", "test_collection")

        # Assertions
        assert exc_info.value.status_code == 500
        assert "Collection not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_delete_collection_missing_params(self):
        # Call the function with missing params and expect exception
        with pytest.raises(HTTPException) as exc_info:
            await delete_collection("test_institution", None)

        # Assertions
        assert exc_info.value.status_code == 500
        assert "The institution_id and/or collection_name are absent" in str(exc_info.value.detail)
