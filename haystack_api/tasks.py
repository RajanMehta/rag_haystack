import inspect
import logging
import mimetypes
from typing import Any

import structlog
from celery import chord, group, signals
from celery.app import Celery
from haystack.components.preprocessors.sentence_tokenizer import SentenceSplitter
from haystack.core.errors import PipelineRuntimeError
from haystack_integrations.document_stores.qdrant.filters import (
    convert_filters_to_qdrant,
)
from qdrant_client.http import models as rest

from haystack_api.config import (
    LOG_LEVEL,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_USER,
)
from haystack_api.errors import (
    EmbeddingError,
    ExtractionError,
    IngestionError,
    RequestValidationError,
)
from haystack_api.scrape import default_extract
from haystack_api.structlog_config import configure_structlog
from haystack_api.utils import get_pipelines, get_qdrant_client

configure_structlog(
    set_root_logger=False,
    set_loggers=True,
    log_level=getattr(logging, LOG_LEVEL),
    override=True,
)

redis_url = f"redis://{REDIS_USER}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

app = Celery(__name__, broker=redis_url, backend=redis_url)

app.conf.event_serializer = "pickle"
app.conf.task_serializer = "pickle"
app.conf.result_serializer = "pickle"
app.conf.accept_content = ["application/json", "application/x-python-serialize"]


@signals.task_prerun.connect
def on_task_prerun(sender, task_id, task, args, kwargs, **_):  # pragma: no cover
    structlog.contextvars.bind_contextvars(task_id=task_id, task_name=task.name)

    if "structlog_contextvars" in kwargs:
        new_ctx = {}
        for k, v in kwargs["structlog_contextvars"].items():
            new_ctx[k.replace("-", "_").removeprefix("x_")] = v
        structlog.contextvars.bind_contextvars(**new_ctx)


def _handle_pipeline_runtime_error(e: PipelineRuntimeError, source_info: str):
    component_name = _extract_component_name(str(e))
    mapped_error_type = COMPONENT_ERROR_TYPE_MAP.get(component_name, "PIPELINE_RUNTIME_ERROR")

    raise IngestionError(
        code=f"{(component_name or 'PIPELINE').upper()}_FAILED",
        error_type=mapped_error_type,
        message="Pipeline failed during indexing.",
        original_error=e,
        source_info=source_info,
        stage=(mapped_error_type.replace("_ERROR", "") or "PIPELINE_RUNTIME"),
    )


def _ingestion_error_payload(e: Exception, source_info: str, document_id: str = None) -> dict:
    if isinstance(e, IngestionError):
        return e.to_error_payload(document_id=document_id)
    return IngestionError(
        code="UNKNOWN",
        error_type="UNKNOWN_ERROR",
        message="Pipeline failed during indexing.",
        original_error=e,
        source_info=str(source_info),
        stage="PIPELINE_RUNTIME",
    ).to_error_payload(document_id=document_id)


@app.task(bind=True)
def index_urls(
    self,
    urls: list[str],
    uuids: list[str],
    params: dict[str, Any],
    pipeline_name,
    css_selectors: list[dict[str, str]] = [],
    **kwargs,
):
    try:
        if not params.get("index"):
            raise RequestValidationError(original_error="Index not set in params")

        tasks = []
        for url, uuid in zip(urls, uuids):
            tasks.append(
                index_url.s(url, uuid, params, pipeline_name, css_selectors, **kwargs).set(
                    task_id=f"{self.request.id}:{uuid}"
                )
            )

        res = group(tasks)()
        res.save()

        return res

    except IngestionError as e:
        return e.to_error_payload()


def _extract_component_name(error_msg: str):
    if "Component name:" in error_msg:
        return error_msg.split("Component name: '")[1].split("'")[0]
    return None


def _splitter_accepts_runtime_chunking(splitter) -> bool:
    # Duck-type: a splitter whose run() accepts a `chunking_strategy` kwarg is
    # assumed to consume preprocessor params at runtime (per-request strategy
    # selection). Anything else is treated as a stock Haystack DocumentSplitter
    # that must be configured via attribute mutation.
    try:
        return "chunking_strategy" in inspect.signature(splitter.run).parameters
    except (TypeError, ValueError):
        return False


COMPONENT_ERROR_TYPE_MAP = {
    "documentcleaner": "CLEANING_ERROR",
    "documentsplitter": "CHUNKING_ERROR",
    "smartdocumentsplitter": "CHUNKING_ERROR",
    "documentwriter": "WRITING_ERROR",
    "documentembedder": "EMBEDDING_ERROR",
    "filetyperouter": "EXTRACTION_ERROR",
    "jsonconverter": "EXTRACTION_ERROR",
    "pypdftodocument": "EXTRACTION_ERROR",
    "docxtodocument": "EXTRACTION_ERROR",
    "textfiletodocument": "EXTRACTION_ERROR",
    "markitdown_pdf": "EXTRACTION_ERROR",
    "markitdown_docx": "EXTRACTION_ERROR",
    "markitdown_txt": "EXTRACTION_ERROR",
}


@app.task(bind=True)
def index_url(self, url, uuid, params, pipeline_name, css_selectors, **kwargs):
    pipeline = get_pipelines().get(pipeline_name, None)
    if not pipeline:
        raise ValueError(f"Pipeline '{pipeline_name}' not found")

    try:
        extracted_document = default_extract(url, css_selectors, uuid)
        extracted_document.meta["tags"] = params["tags"]

        # set index at runtime
        documentwriter = pipeline.get_component("documentwriter")
        documentwriter.document_store.index = params.get("index")

        # preprocessing params are set in two components depending on the functionality
        documentsplitter = pipeline.get_component("documentsplitter")
        documentcleaner = pipeline.get_component("documentcleaner")
        preprocessor_params = params["preprocessor_params"]

        documentcleaner.remove_substrings = preprocessor_params["remove_substrings"]

        kwargs.pop("structlog_contextvars", None)

        # Build the run input based on splitter type
        run_input = {"documentcleaner": {"documents": [extracted_document]}}

        if _splitter_accepts_runtime_chunking(documentsplitter):
            # Pass chunking params at runtime via pipeline.run() data dict
            splitter_params = {}
            for key in (
                "chunking_strategy",
                "split_length",
                "split_overlap",
                "split_by",
                "secondary_split",
                "keep_headers",
                "split_threshold",
                "separators",
                "split_unit",
                "split_respect_sentence_boundary",
            ):
                val = preprocessor_params.get(key)
                if val is not None:
                    splitter_params[key] = val
            if splitter_params:
                run_input["documentsplitter"] = splitter_params
        else:
            # Stock DocumentSplitter — mutate attributes directly
            documentsplitter.split_by = preprocessor_params["split_by"]
            documentsplitter.split_length = preprocessor_params["split_length"]
            documentsplitter.split_overlap = preprocessor_params["split_overlap"]
            documentsplitter.respect_sentence_boundary = preprocessor_params["split_respect_sentence_boundary"]
            documentsplitter.sentence_splitter = SentenceSplitter(
                keep_white_spaces=True,
            )

        try:
            pipeline.run(
                run_input,
                **kwargs,
            )

        except PipelineRuntimeError as e:
            _handle_pipeline_runtime_error(e, source_info=url)

        return {
            "indexing": update_embeddings(
                uuid=uuid,
                index=params.get("index"),
            )
        }

    except IngestionError as e:
        return e.to_error_payload(document_id=uuid)
    except Exception as e:
        return _ingestion_error_payload(e, source_info=url, document_id=uuid)


@app.task(bind=True)
def index_files(self, files: list[Any], metas: list[Any], params: dict[str, Any], pipeline_name, **kwargs):
    try:
        if len(files) != len(metas):
            raise RequestValidationError(original_error="Length of files and metas does not match")

        if not params.get("index"):
            raise RequestValidationError(original_error="Index not set in params")

        tasks = []
        for file, meta in zip(files, metas):
            uuid = meta.get("uuid")

            tasks.append(
                index_file.s(file, meta, params, pipeline_name, **kwargs).set(task_id=f"{self.request.id}:{uuid}")
            )

        res = group(tasks)()
        res.save()

        return res

    except IngestionError as e:
        return e.to_error_payload()


@app.task
def index_file(file, meta, params, pipeline_name, **kwargs):  # pragma: no cover
    pipeline = get_pipelines().get(pipeline_name, None)
    if not pipeline:
        raise ValueError(f"Pipeline '{pipeline_name}' not found")

    try:
        # set index at runtime
        documentwriter = pipeline.get_component("documentwriter")
        documentwriter.document_store.index = params.get("index")

        # preprocessing params are set in two components depending on the functionality
        documentsplitter = pipeline.get_component("documentsplitter")
        documentcleaner = pipeline.get_component("documentcleaner")
        preprocessor_params = params["preprocessor_params"]

        documentcleaner.remove_substrings = preprocessor_params["remove_substrings"]

        kwargs.pop("structlog_contextvars", None)

        # Build the run input based on splitter type and available converters
        run_input = {
            "filetyperouter": {"sources": [file]},
        }

        # Map mime types to converter component names
        MIME_TO_CONVERTERS = {
            "application/json": ["jsonconverter"],
            "application/pdf": ["pypdftodocument", "markitdown_pdf"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
                "docxtodocument",
                "markitdown_docx",
            ],
            "text/plain": ["textfiletodocument", "markitdown_txt"],
        }

        # Determine which converter will receive this file and only pass meta to it
        file_mime, _ = mimetypes.guess_type(str(file))
        candidate_converters = MIME_TO_CONVERTERS.get(file_mime, [])
        for converter_name in candidate_converters:
            try:
                pipeline.get_component(converter_name)
                run_input[converter_name] = {"meta": [meta]}
            except ValueError:
                pass

        if _splitter_accepts_runtime_chunking(documentsplitter):
            # Pass chunking params at runtime via pipeline.run() data dict
            splitter_params = {}
            for key in (
                "chunking_strategy",
                "split_length",
                "split_overlap",
                "split_by",
                "secondary_split",
                "keep_headers",
                "split_threshold",
                "separators",
                "split_unit",
                "split_respect_sentence_boundary",
            ):
                val = preprocessor_params.get(key)
                if val is not None:
                    splitter_params[key] = val
            if splitter_params:
                run_input["documentsplitter"] = splitter_params
        else:
            # Stock DocumentSplitter — mutate attributes directly
            documentsplitter.split_by = preprocessor_params["split_by"]
            documentsplitter.split_length = preprocessor_params["split_length"]
            documentsplitter.split_overlap = preprocessor_params["split_overlap"]
            documentsplitter.respect_sentence_boundary = preprocessor_params["split_respect_sentence_boundary"]
            documentsplitter.sentence_splitter = SentenceSplitter(
                keep_white_spaces=True,
            )

        try:
            pipeline.run(
                run_input,
                **kwargs,
            )
        except PipelineRuntimeError as e:
            _handle_pipeline_runtime_error(e, source_info=str(file))

        # delete files after indexing
        file.unlink(missing_ok=True)

        return {
            "indexing": update_embeddings(
                uuid=meta.get("uuid"),
                index=params.get("index"),
            )
        }

    except IngestionError as e:
        return e.to_error_payload(document_id=meta.get("uuid"))
    except Exception as e:
        return _ingestion_error_payload(e, source_info=str(file), document_id=meta.get("uuid"))


@app.task
def update_embeddings(uuid, index):
    embedding_pipeline = get_pipelines().get("embedding_pipeline", None)
    if not embedding_pipeline:
        raise ValueError("Embedding pipeline not found")

    document_writer = embedding_pipeline.get_component("documentwriter")
    if not document_writer:
        raise ValueError("Document writer not found in the pipeline")

    try:
        document_writer.document_store.index = index
        docs = document_writer.document_store.filter_documents({"field": "meta.uuid", "operator": "==", "value": uuid})

        if not docs:
            raise ExtractionError(
                code="CONTENT_EMPTY",
                stage="EXTRACTION",
                message="Content empty after extraction/cleaning. No documents available to embed.",
                source_info=uuid,
            )

        try:
            embedding_pipeline.run({"documents": docs})
        except Exception as e:
            component_name = _extract_component_name(str(e))
            mapped_error_type = COMPONENT_ERROR_TYPE_MAP.get(component_name, "EMBEDDING_ERROR")

            raise EmbeddingError(
                code=f"{(component_name or 'EMBEDDING').upper()}_FAILED",
                error_type=mapped_error_type,
                message="Embedding generation failed.",
                original_error=e,
                stage=(mapped_error_type.replace("_ERROR", "") or "EMBEDDING_GENERATION"),
                source_info=index,
                http_status_code=500,
            )

        client = get_qdrant_client()
        try:
            client.set_payload(
                collection_name=index,
                payload={"embedded": True},
                key="meta",
                points=convert_filters_to_qdrant({"field": "meta.uuid", "operator": "==", "value": uuid}),
            )
        except Exception as e:
            raise EmbeddingError(
                code="DOCUMENTWRITER_FAILED",
                error_type="WRITING_ERROR",
                message="Failed to update embedding payload.",
                original_error=e,
                stage="WRITING",
                source_info=index,
                http_status_code=500,
            )

        return {"status": "COMPLETED", "embedded": len(docs)}

    except IngestionError as e:
        raise e


@app.task
def delete_points_from_qdrant(results, collection_name, uuids):
    get_qdrant_client().delete(
        collection_name=f"{collection_name}",
        points_selector=rest.PointIdsList(
            points=uuids,
        ),
    )

    return results


@app.task
def update_embeddings_dedup(collection_name, request_uuids, batch_uuids, *args, **kwargs):
    if "structlog_contextvars" in kwargs:  # pragma: no cover
        del kwargs["structlog_contextvars"]

    tasks = [
        update_embeddings.s(
            filters={"uuid": uuid},
            index=collection_name,
            *args,
            **kwargs,
        ).set(task_id=f"update_embeddings:{uuid}")
        for uuid in request_uuids
    ]

    res = chord(tasks)(delete_points_from_qdrant.s(collection_name, batch_uuids))

    return {"callback": res, "embeddings": [f"update_embeddings:{uuid}" for uuid in request_uuids]}
