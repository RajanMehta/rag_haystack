# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
#
# This file is based on https://github.com/deepset-ai/haystack
# See: https://github.com/deepset-ai/haystack/blob/main/rest_api/rest_api/utils.py

import io
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from haystack import AsyncPipeline, Pipeline
from qdrant_client import AsyncQdrantClient, QdrantClient
from starlette.middleware.cors import CORSMiddleware

from experiments._registry import EXPERIMENT_CONFIGS
from haystack_api.config import (
    PIPELINE_CONFIG,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_HTTP_PORT,
)
from haystack_api.controller.errors.http_error import http_error_handler
from haystack_api.pipeline.pipeline_configs import en_gen_config, test_config
from haystack_api.structlog_config import StructLogMiddleware

app: FastAPI = None
pipelines = None
q_client: QdrantClient = None
async_q_client: AsyncQdrantClient = None

INDEXING_PIPELINE = "indexing_pipeline"
DOCUMENT_INDEXING_PIPELINE = "document_indexing_pipeline"
SEARCH_PIPELINE = "search_pipeline"
PIPELINE_CONFIG_MAP = {
    "en_gen": en_gen_config,
    "test": test_config,
    **EXPERIMENT_CONFIGS,
}

logger = logging.getLogger(__name__)


class QdrantClientSingleton:
    _async_instance = None
    _sync_instance = None

    @classmethod
    async def get_async_instance(cls):
        if cls._async_instance is None:
            cls._async_instance = AsyncQdrantClient(
                host=QDRANT_HOST, port=QDRANT_HTTP_PORT, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True, timeout=3.0
            )
        return cls._async_instance

    @classmethod
    def get_sync_instance(cls):
        if cls._sync_instance is None:
            cls._sync_instance = QdrantClient(
                host=QDRANT_HOST, port=QDRANT_HTTP_PORT, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True, timeout=3.0
            )
        return cls._sync_instance

    @classmethod
    async def close(cls):
        if cls._async_instance:
            await cls._async_instance.close()
            cls._async_instance = None
        if cls._sync_instance:
            cls._sync_instance.close()
            cls._sync_instance = None


def get_qdrant_client() -> QdrantClient:
    """
    Returns and caches the synchronous qdrant client
    """
    return QdrantClientSingleton.get_sync_instance()


async def get_async_qdrant_client() -> AsyncQdrantClient:
    """
    Returns and caches the asynchronous qdrant client
    """

    return await QdrantClientSingleton.get_async_instance()


def get_app() -> FastAPI:
    """
    Initializes the App object and creates the global pipelines as possible.
    """
    global app  # pylint: disable=global-statement
    if app:
        return app

    from haystack_api.config import ROOT_PATH

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup code (if any)
        yield
        # Shutdown code
        await QdrantClientSingleton.close()

    app = FastAPI(
        title="Haystack REST API",
        debug=True,
        root_path=ROOT_PATH,
        lifespan=lifespan,
    )

    # Creates the router for the API calls
    from haystack_api.controller import (
        collection,
        document,
        evaluators,
        file_upload,
        health,
        query,
        tasks,
        web_scrape,
    )

    router = APIRouter()
    router.include_router(query.router, tags=["search"])
    router.include_router(file_upload.router, tags=["file-upload"])
    router.include_router(web_scrape.router, tags=["web-scrape"])
    router.include_router(document.router, tags=["document"])
    router.include_router(collection.router, tags=["collection"])
    router.include_router(evaluators.router, tags=["evaluators"])
    router.include_router(health.router, tags=["health"])
    router.include_router(tasks.router, tags=["tasks"])

    app.add_middleware(StructLogMiddleware)

    # This middleware enables allow all cross-domain requests to the API from a browser.
    # For production deployments, it could be made more restrictive.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(HTTPException, http_error_handler)
    app.include_router(router)

    # Simplify operation IDs so that generated API clients have simpler function
    # names (see https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#using-the-path-operation-function-name-as-the-operationid).
    # The operation IDs will be the same as the route names
    # (i.e. the python method names of the endpoints)
    # Should be called only after all routes have been added.
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name

    return app


def setup_pipelines() -> Dict[str, Any]:
    # Re-import the configuration variables
    from haystack_api import config  # pylint: disable=reimported

    pipelines = {}
    chosen_config = PIPELINE_CONFIG_MAP.get(PIPELINE_CONFIG, None)
    if chosen_config:
        all_pipeline_configs = chosen_config.pipelines
        for pipeline in all_pipeline_configs:
            # Convert the JSON data to a string
            json_string = json.dumps(pipeline["configs"])

            # Create a StringIO object
            file_object = io.StringIO(json_string)
            if pipeline["type"] == "sync":
                pipelines[pipeline["name"]] = Pipeline.load(file_object)
            else:
                pipelines[pipeline["name"]] = AsyncPipeline.load(file_object)
            file_object.close()
    else:
        raise ValueError("Value is invalid for PIPELINE_CONFIG")

    # Create directory for uploaded files
    os.makedirs(config.FILE_UPLOAD_PATH, exist_ok=True)

    return pipelines


def get_pipelines():
    global pipelines  # pylint: disable=global-statement
    if not pipelines:
        pipelines = setup_pipelines()

    return pipelines


def get_indexing_pipeline():
    indexing_pipeline = get_pipelines().get(INDEXING_PIPELINE, None)
    if not indexing_pipeline:
        raise HTTPException(status_code=501, detail="Indexing Pipeline is not configured.")
    return indexing_pipeline


def get_document_store():
    indexing_pipeline = get_indexing_pipeline()
    document_store = indexing_pipeline.get_component("documentwriter")
    if not document_store:
        raise HTTPException(status_code=501, detail="Indexing Pipeline needs a DocumentStore component.")
    return document_store


def get_openapi_specs() -> dict:
    """
    Used to autogenerate OpenAPI specs file to use in the documentation.

    Returns `servers` to specify base URL for OpenAPI Playground (see https://swagger.io/docs/specification/api-host-and-base-path/)

    See `.github/utils/generate_openapi_specs.py`
    """

    app = get_app()
    return get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
        servers=[{"url": "http://localhost:31415"}],
    )
