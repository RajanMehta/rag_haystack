# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
#
# This file is based on https://github.com/deepset-ai/haystack
# See: https://github.com/deepset-ai/haystack/blob/main/rest_api/rest_api/controller/file_upload.py

import json
import shutil
import uuid
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from haystack_api import structlog_config
from haystack_api.config import FILE_UPLOAD_PATH
from haystack_api.controller.utils import as_form
from haystack_api.errors import RequestValidationError
from haystack_api.tasks import index_files
from haystack_api.utils import INDEXING_PIPELINE, get_app

router = APIRouter()
app: FastAPI = get_app()


@as_form
class PreprocessorParams(BaseModel):
    remove_substrings: Optional[str] = None
    chunking_strategy: Optional[str] = None
    split_by: Optional[str] = None
    split_length: Optional[int] = None
    split_overlap: Optional[int] = None
    split_respect_sentence_boundary: Optional[bool] = None
    # MarkdownHeaderSplitter params
    secondary_split: Optional[str] = None
    keep_headers: Optional[bool] = None
    split_threshold: Optional[int] = None
    # RecursiveDocumentSplitter params
    separators: Optional[str] = None
    split_unit: Optional[str] = None


class Response(BaseModel):
    file_id: str


@router.post("/file-upload")
async def upload_file(
    files: List[UploadFile] = File(...),
    institution_id: str = Form(""),
    collection_name: str = Form(""),
    # JSON serialized string
    tags: Optional[str] = Form("null"),
    meta: Optional[str] = Form("null"),
    uuids: Optional[str] = Form("null"),
    preprocessor_params: PreprocessorParams = Depends(PreprocessorParams.as_form),
):
    """
    Use this endpoint to upload a file for indexing
    :param files: files to upload
    :param institution_id: a valid institution_id
    :param collection_name: a valid collection name in which indexed file contents
                            will be stored and embedded
    :param tags: JSON stringified list of tags. Each tag will be assigned to
                each chunk of file stored in document store. tags can be used to
                filter search results
    :param meta: JSON stringified dict to add any other metadata to each chunk,
                filtering will be possible with metadata

    -- preprocessor_params exposed in front-end --
    :param remove_substrings: Remove specified substrings from the text.
                            If no value is provided an empty list is created by default.
    :param split_by: Unit for splitting the document. Can be "word", "sentence", or "passage".
                    Set to None to disable splitting.
    :param split_length: Max. number of the above split unit (e.g. words) that are allowed in one document.
                    For instance, if n -> 10 & split_by ->  "sentence",
                    then each output document will have 10 sentences.
    :param split_overlap: Word overlap between two adjacent documents after a split.
                            Setting this to a positive number essentially enables the sliding window approach.
                            For example, if split_by -> `word`,
                            split_length -> 5 & split_overlap -> 2, then the splits would be like:
                            [w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12].
                            Set the value to 0 to ensure there is no overlap among the documents after splitting.
    :param split_respect_sentence_boundary: Whether to split in partial sentences if split_by -> `word`. If set
                                            to True, the individual split will always have complete sentences &
                                            the number of words will be <= split_length.

    """
    try:
        # validate required parameters
        if not (institution_id and collection_name):
            raise RequestValidationError(original_error="The institution_id and/or collection_name are absent.")

        # validate JSON strings
        try:
            meta_form = json.loads(meta) or {}  # type: ignore
        except JSONDecodeError as e:
            raise RequestValidationError(
                original_error=(
                    f"The meta field must be a valid JSON-stringified dict or None, "
                    f"not {type(meta)}. Error: {str(e)}"
                )
            )

        # validate and add tags to file meta
        try:
            tags = json.loads(tags) or []
            uuids = json.loads(uuids) or []
        except JSONDecodeError as e:
            raise RequestValidationError(
                original_error=(
                    f"The tags or uuids field must be a valid JSON-stringified list or None, "
                    f"not {type(tags)} or {type(uuids)}. Error: {str(e)}"
                )
            )

        if len(uuids) != len(files):
            raise RequestValidationError(
                original_error="The number of UUIDs provided does not match the number of files."
            )

        preprocessor_params = preprocessor_params.model_dump()

        # validate and add remove_substrings to preprocessor_params
        if preprocessor_params["remove_substrings"]:
            try:
                remove_substrings = json.loads(preprocessor_params["remove_substrings"]) or []
            except JSONDecodeError as e:
                raise RequestValidationError(
                    original_error=(
                        f"The remove_substrings field must be a valid JSON-stringified list or None, "
                        f"not {type(preprocessor_params['remove_substrings'])}. Error: {str(e)}"
                    )
                )
            preprocessor_params["remove_substrings"] = remove_substrings

        # validate and parse separators JSON string
        if preprocessor_params.get("separators"):
            try:
                separators = json.loads(preprocessor_params["separators"]) or []
            except JSONDecodeError as e:
                raise RequestValidationError(
                    original_error=(
                        f"The separators field must be a valid JSON-stringified list or None, "
                        f"not {type(preprocessor_params['separators'])}. Error: {str(e)}"
                    )
                )
            preprocessor_params["separators"] = separators

    except RequestValidationError as e:
        return JSONResponse(content=e.to_error_payload(), status_code=400)

    file_paths: list = []
    file_metas: list = []
    meta_form["tags"] = tags

    for i, file in enumerate(files):
        try:
            file_path = Path(FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file.filename}"
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(file_path)

            file_meta = meta_form.copy()
            file_meta["source"] = file.filename
            file_meta["embedded"] = False
            file_meta["uuid"] = uuids[i]
            file_metas.append(file_meta)
        finally:
            file.file.close()

    params = {}  # type: ignore
    params["index"] = f"{institution_id}_{collection_name}"
    params["preprocessor_params"] = preprocessor_params

    task_meta = index_files.delay(
        file_paths, file_metas, params, INDEXING_PIPELINE, structlog_contextvars=structlog_config.get_headers()
    )

    return JSONResponse(content={"success": True, "task_id": task_meta.task_id})
