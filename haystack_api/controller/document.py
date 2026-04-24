# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
#
# This file is based on https://github.com/deepset-ai/haystack
# See: https://github.com/deepset-ai/haystack/blob/main/rest_api/rest_api/controller/document.py

import logging
from typing import List

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from haystack import Document
from haystack_integrations.document_stores.qdrant.converters import (
    convert_haystack_documents_to_qdrant_points,
)
from haystack_integrations.document_stores.qdrant.filters import (
    convert_filters_to_qdrant,
)
from qdrant_client.http import models as rest

from haystack_api.config import LOG_LEVEL
from haystack_api.schema import FilterRequest, TagsRequest
from haystack_api.utils import get_app, get_async_qdrant_client, get_document_store

logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")

router = APIRouter()
app: FastAPI = get_app()


@router.post(
    "/documents/get_by_filters",
    response_model=List[Document],
    response_model_exclude_none=True,
)
async def get_documents(request: FilterRequest):
    """
    This endpoint allows you to retrieve documents contained in your document store.
    You can filter the documents to retrieve by metadata (like the document's name),
    or provide an empty JSON object to clear the document store.

    Example of filters:
    `{
        "operator": "AND",
        "conditions": [
            {"field": "meta.source", "operator": "==", "value": "Sick leave.pdf"}
        ]
    }`

    To get all documents you should provide an empty dict, like:
    `'{"filters": {}}'`
    """
    document_writer = get_document_store()
    try:
        if document_writer:
            document_writer.document_store.index = f"{request.institution_id}_{request.collection_name}"
            docs = await document_writer.document_store.filter_documents_async(filters=request.filters)
            for doc in docs:
                doc.embedding = None
            return docs
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/count")
async def get_documents_count(request: FilterRequest):
    """
    Finds all documents that match given filtering condition, and
    returns total documents, and number of documents that are embedded
    """
    client = await get_async_qdrant_client()

    embedded_condition = {"field": "meta.embedded", "operator": "==", "value": True}
    if request.filters and request.filters.get("conditions"):
        count_embedded_filter = {"operator": "AND", "conditions": request.filters["conditions"] + [embedded_condition]}
    else:
        count_embedded_filter = {"operator": "AND", "conditions": [embedded_condition]}

    try:
        total_docs = await client.count(
            collection_name=f"{request.institution_id}_{request.collection_name}",
            count_filter=convert_filters_to_qdrant(request.filters) if request.filters else None,
        )
        embedded_docs = await client.count(
            collection_name=f"{request.institution_id}_{request.collection_name}",
            count_filter=convert_filters_to_qdrant(count_embedded_filter),
        )
        return {"total_docs": total_docs.count, "embedded_docs": embedded_docs.count}
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/delete_by_filters", response_model=bool)
async def delete_documents(request: FilterRequest):
    """
    This endpoint allows you to delete documents contained in your document store.
    You can filter the documents to delete by metadata (like the document's name),
    or provide an empty JSON object to clear the document store.

    Example of filters:
    `'{"filters": {{"name": ["some", "more"], "category": ["only_one"]}}'`

    To get all documents you should provide an empty dict, like:
    `'{"filters": {}}'`
    """
    client = await get_async_qdrant_client()
    collection_name = f"{request.institution_id}_{request.collection_name}"
    try:
        if request.filters:
            qdrant_filter = convert_filters_to_qdrant(request.filters)
        else:
            # Empty filters means "delete all"
            qdrant_filter = rest.FilterSelector(filter=rest.Filter())
        await client.delete(collection_name=collection_name, points_selector=qdrant_filter)
        return JSONResponse(content={"success": True})
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/update_tags")
async def update_document_tags(request: TagsRequest):
    """
    Add or Delete tags to all the documents that match filtering condition
    """
    document_writer = get_document_store()
    client = await get_async_qdrant_client()
    if document_writer:
        document_writer.document_store.index = f"{request.institution_id}_{request.collection_name}"
    try:
        target_docs = await document_writer.document_store.filter_documents_async(filters=request.filters)

        for doc in target_docs:
            doc.meta["tags"] = request.tags

        batch = convert_haystack_documents_to_qdrant_points(target_docs, use_sparse_embeddings=False)

        await client.upsert(
            collection_name=f"{request.institution_id}_{request.collection_name}",
            points=batch,
            wait=document_writer.document_store.wait_result_from_api,
        )
        return JSONResponse(content={"success": True})
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


# @router.post("/documents/add")
# async def add_document(request: AddDocumentRequest):
#     """
#     Add json document
#     """
#     if not (request.institution_id and request.collection_name):
#         raise HTTPException(status_code=500, detail="The institution_id and/or collection_name are absent.")

#     embeddings = [each_content.embedding for each_content in request.data]

#     if len(request.uuids) != len(request.data):
#         raise HTTPException(
#             status_code=500,
#             detail="The number of UUIDs provided does not match the number of items.",
#         )

#     try:
#         index = f"{request.institution_id}_{request.collection_name}"
#         UUID_NAMESPACE = uuid.UUID("3896d314-1e95-4a3a-b45a-945f9f0b541d")
#         batch_uuids = [uuid.uuid5(UUID_NAMESPACE, item.content).hex for item in request.data]

#         client = get_async_qdrant_client()
#         datas = []
#         for each_content, each_uuid, each_id in zip(request.data, request.uuids, batch_uuids):
#             if embeddings[0][0]:
#                 each_content.meta = {
#                     "embedded": True,
#                     "uuid": each_uuid,
#                 }
#             else:
#                 each_content.meta = {
#                     "embedded": False,
#                     "uuid": each_uuid,
#                 }

#             for each in list(each_content.__iter__()):
#                 key = each[0]
#                 if key == "embedding":
#                     each_content.__delattr__(key)
#                 elif key != "content" and key != "meta":
#                     each_content.meta[key] = getattr(each_content, key)
#                     each_content.__delattr__(key)

#             setattr(each_content, "content_type", "text")
#             setattr(each_content, "id", each_id)
#             setattr(each_content, "id_hash_keys", [])
#             datas.append(each_content.__dict__)

#         batch = rest.Batch(
#             ids=batch_uuids,
#             vectors=embeddings,
#             payloads=datas,
#         )
#         await client.upsert(
#             collection_name=index,
#             points=batch,
#         )
#         if not embeddings[0][0]:
#             embedding_args = []
#             embedding_kwargs = {}
#             task_meta = update_embeddings_dedup.delay(
#                 index,
#                 request.uuids,
#                 batch_uuids,
#                 *embedding_args,
#                 **embedding_kwargs,
#                 structlog_contextvars=structlog_config.get_headers(),
#             )

#             return JSONResponse(content={"success": True, "task_id": task_meta.task_id})

#         return JSONResponse(content={"success": True})
#     except Exception as e:
#         logger.error(e)
#         raise HTTPException(status_code=500, detail=str(e))
