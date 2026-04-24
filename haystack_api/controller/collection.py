import json
import logging

from fastapi import APIRouter, FastAPI, HTTPException
from google.protobuf.json_format import MessageToJson
from grpc import RpcError
from qdrant_client import grpc
from qdrant_client.http.exceptions import UnexpectedResponse

from haystack_api.config import LOG_LEVEL
from haystack_api.schema import CreateCollectionRequest
from haystack_api.utils import get_app, get_async_qdrant_client, get_document_store

logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


router = APIRouter()
app: FastAPI = get_app()


@router.post("/collection/create")
async def create_collection(request: CreateCollectionRequest):
    """
    Creates a collection with name `{institution_id}_{collection_name}`
    Uses the configuration params of `Store` component as defined in the pipeline.yaml file
    """
    if request.institution_id and request.collection_name:
        try:
            async_q_client = await get_async_qdrant_client()
            grpc_collections = async_q_client.grpc_collections
            document_store = get_document_store()
            document_store_params = (
                document_store.to_dict().get("init_parameters").get("document_store").get("init_parameters")
            )
            response = await grpc_collections.Create(
                grpc.CreateCollection(
                    collection_name=f"{request.institution_id}_{request.collection_name}",
                    vectors_config=grpc.VectorsConfig(
                        params=grpc.VectorParams(
                            size=document_store_params.get("embedding_dim"), distance=grpc.Distance.Cosine
                        )
                    ),
                    shard_number=document_store_params.get("shard_number"),
                    replication_factor=document_store_params.get("replication_factor"),
                    write_consistency_factor=document_store_params.get("write_consistency_factor"),
                    on_disk_payload=document_store_params.get("on_disk_payload"),
                    hnsw_config=document_store_params.get("hnsw_config"),
                    optimizers_config=document_store_params.get("optimizers_config"),
                    wal_config=document_store_params.get("wal_config"),
                    quantization_config=document_store_params.get("quantization_config"),
                )
            )
            return json.loads(MessageToJson(response))
        except (UnexpectedResponse, RpcError) as e:
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=500, detail="The institution_id and/or collection_name are absent.")


@router.get("/info/collection/{institution_id}")
@router.get("/info/collection/{institution_id}/{collection_name}")
async def get_collection_info(institution_id: str | None = None, collection_name: str | None = None):
    """
    Get current statistics and configuration of an existing collection
    """
    if institution_id and collection_name:
        try:
            async_q_client = await get_async_qdrant_client()
            grpc_collections = async_q_client.grpc_collections
            response = await grpc_collections.Get(
                grpc.GetCollectionInfoRequest(collection_name=f"{institution_id}_{collection_name}")
            )
            return json.loads(MessageToJson(response))
        except (UnexpectedResponse, RpcError) as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=500, detail="The institution_id and/or collection_name are absent.")


@router.delete("/collection/{institution_id}/{collection_name}")
async def delete_collection(institution_id: str | None = None, collection_name: str | None = None):
    """
    Drops a collection and all associated data
    """
    if institution_id and collection_name:
        try:
            async_q_client = await get_async_qdrant_client()
            grpc_collections = async_q_client.grpc_collections
            response = await grpc_collections.Delete(
                grpc.DeleteCollection(collection_name=f"{institution_id}_{collection_name}")
            )
            return json.loads(MessageToJson(response))
        except (UnexpectedResponse, RpcError) as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=500, detail="The institution_id and/or collection_name are absent.")
