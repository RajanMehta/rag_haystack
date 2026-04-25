# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
#
# This file is based on https://github.com/deepset-ai/haystack
# See: https://github.com/deepset-ai/haystack/blob/main/rest_api/rest_api/controller/search.py

import logging
import re

import pydash
from fastapi import APIRouter, FastAPI, HTTPException
from haystack import AsyncPipeline

from haystack_api.config import LOG_LEVEL, OLLAMA_MODEL
from haystack_api.controller.utils import make_serializable
from haystack_api.prompt_templates import get_template
from haystack_api.schema import GenerateRequest, QueryRequest, QueryResponse
from haystack_api.utils import get_app, get_pipelines

logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


router = APIRouter()
app: FastAPI = get_app()


@router.get("/initialized")
def check_status():  # pragma: no cover
    """
    This endpoint can be used during startup to understand if the
    server is ready to take any requests, or is still loading.

    The recommended approach is to call this endpoint with a short timeout,
    like 500ms, and in case of no reply, consider the server busy.
    """
    return {"success": True}


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
async def query(request: QueryRequest):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    """
    params: dict = request.params.model_dump() if request.params else {}
    params = {key: value for key, value in params.items() if value is not None}

    query_execution_settings = {
        "search_pipeline": {
            "data_generator": lambda request, params: {
                "text_embedder": {"text": request.query},
                "embedding_retriever": {
                    "filters": params.get("filters"),
                    "score_threshold": params.get("threshold"),
                    "top_k": params.get("top_k"),
                    "index": f"{request.institution_id}_{request.collection_name}",
                },
                "document_router": {"generate": params.get("generate", False), "query": request.query},
            },
            "target_key": ["documents", "replies"],
        },
    }

    pipeline: AsyncPipeline = get_pipelines().get(request.pipeline_name, None)
    query_execution_setting = query_execution_settings.get(request.pipeline_name)

    if not pipeline:
        raise HTTPException(
            status_code=501,
            detail="Query Pipeline is not configured or is invalid.",
        )

    # Generate data for the pipeline
    data = query_execution_setting["data_generator"](request, params)

    # If the configured pipeline includes a `query_filter_builder` component
    # (currently used by exp_002 to derive Qdrant filters from query entities),
    # route the user-supplied filters through it so it can AND-merge them with
    # extracted ones. The retriever's filters input is wired to the builder's
    # output in that pipeline shape.
    if "query_filter_builder" in pipeline.to_dict()["components"]:
        user_filters = data.get("embedding_retriever", {}).pop("filters", None)
        data["query_filter_builder"] = {
            "query": request.query,
            "user_filters": user_filters,
        }

    if params.get("generate"):
        template = get_template(request.pipeline_name, "local", OLLAMA_MODEL)
        data.update(
            {
                "prompt_builder": {"template": template},
            }
        )

    # Run the pipeline
    include_outputs_from = (
        pipeline.to_dict()["components"].keys() if request.debug else ["embedding_retriever", "local_generator"]
    )
    result = await pipeline.run_async(data=data, include_outputs_from=include_outputs_from)

    # Format response
    formatted_result = {}
    for key in query_execution_setting["target_key"]:
        # Find target path using the configured key
        target_path = pydash.find_key(result, key)
        if target_path:
            formatted_result[key] = pydash.get(result, target_path)[key]
        else:
            formatted_result[key] = []

    if formatted_result.get("documents"):
        formatted_result["documents"] = pydash.collections.group_by(formatted_result["documents"], "meta.source")

    # Create response
    response = {"query": request.query, "raw_results": result, "results": formatted_result}
    return response


@router.post("/generate", response_model=QueryResponse, response_model_exclude_none=True)
async def generate(request: GenerateRequest):
    """
    This endpoint receives utterances as a list and an instruction to generate data based upon those parameters
    as well as additional parameters that will be passed on to the Haystack pipeline.
    """
    gen_pipeline: AsyncPipeline = get_pipelines().get(request.pipeline_name, None)
    params: dict = request.params.model_dump() if request.params else {}
    params = {key: value for key, value in params.items() if value is not None}

    if not isinstance(gen_pipeline, AsyncPipeline):  # pragma: no cover
        raise HTTPException(
            status_code=501,
            detail="Generation Pipeline is not configured or is invalid.",
        )

    template = get_template(
        request.pipeline_name, params.get("model_type", "default"), params.get("model_name", "default")
    )

    include_outputs_from = gen_pipeline.to_dict()["components"].keys() if request.debug else None
    result = await gen_pipeline.run_async(
        data={
            "model_type_router": {"params": params},
            f"prompt_builder_{params['model_type']}": {"template": template},
        },
        include_outputs_from=include_outputs_from,
    )

    generated_utterances = result[f"{params['model_type']}_generator"]["replies"][0]
    generated_utterances = [
        re.sub(r"^\d+\.\s*", "", utterance.strip())
        for utterance in generated_utterances.split("\n")
        if utterance.strip()
    ]
    generated_utterances = [utterance.strip('"') for utterance in generated_utterances]

    response = {
        "query": template,
        "raw_results": make_serializable(result),
        "results": {
            "generated_utterances": generated_utterances,
        },
    }

    return response
