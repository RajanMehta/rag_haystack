# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
#
# This file is based on https://github.com/deepset-ai/haystack
# See: https://github.com/deepset-ai/haystack/blob/main/rest_api/rest_api/schema.py

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import (
    BaseConfig,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from qdrant_client.http import models as rest

from haystack_api.config import OPENAI_API_KEY

BaseConfig.arbitrary_types_allowed = True
BaseConfig.json_encoders = {
    np.ndarray: lambda x: x.tolist(),
    pd.DataFrame: lambda x: [x.columns.tolist()] + x.values.tolist(),
}


class RequestBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class Retriever(RequestBaseModel):
    top_k: Optional[int] = Field(default=10, ge=1, le=20, description="Top k must be a positive integer <= 20")
    debug: Optional[bool]
    filters: Optional[Union[dict, List[dict]]]

    @field_validator("top_k")
    @classmethod
    def set_default_top_k(cls, value):
        if value is None:
            return 10
        return value


class Ranker(RequestBaseModel):
    top_k: Optional[int] = Field(default=10, ge=1, description="Top k must be a positive integer")
    debug: Optional[bool]

    @field_validator("top_k")
    @classmethod
    def set_default_top_k(cls, value):
        if value is None:
            return 10
        return value


class Thresholder(RequestBaseModel):
    threshold: Optional[float] = Field(ge=0, le=1, description="Threshold must be between 0 and 1")
    debug: Optional[bool]

    @field_validator("threshold")
    @classmethod
    def set_default_threshold(cls, value):
        if value is None:
            return 0
        return value


class GenerateDecision(RequestBaseModel):
    generate: Optional[bool]
    debug: Optional[bool]


class QueryParams(RequestBaseModel):
    threshold: Optional[float] = Field(ge=0, le=1, description="Threshold must be between 0 and 1")
    top_k: Optional[int] = Field(default=10, ge=1, le=20, description="Top k must be a positive integer <= 20")
    generate: bool | None = None
    language: str | None = None
    filters: Union[dict, List[dict]] | None = None

    @field_validator("threshold")
    @classmethod
    def set_default_threshold(cls, value):
        if value is None:
            return 0.0
        return value

    @field_validator("top_k")
    @classmethod
    def set_default_top_k(cls, value):
        if value is None:
            return 10
        return value


class ModelName(str, Enum):
    phi_4 = "hf.co/unsloth/phi-4-GGUF:Q3_K_M"
    gpt_35 = "gpt-3.5-turbo"
    gpt_4o = "gpt-4o"
    gpt_4o_mini = "gpt-4o-mini"


class ModelType(str, Enum):
    local_model = "local"
    open_ai = "openai"


class GenerateParams(RequestBaseModel):
    model_type: ModelType
    model_name: ModelName
    api_key: Optional[str]
    generation_kwargs: Optional[Dict[str, Any]]
    invocation_context: Dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def check_api_key_for_openai(cls, data):
        if data.get("model_type") == ModelType.open_ai:
            if not data.get("api_key"):
                if not OPENAI_API_KEY:
                    raise ValueError(f"api_key is required for {ModelType.open_ai} model")
                else:
                    data["api_key"] = OPENAI_API_KEY
        return data


class QueryRequest(RequestBaseModel):
    query: str
    pipeline_name: str
    institution_id: str | None = None
    collection_name: str | None = None
    params: QueryParams | None = None
    debug: bool | None = None

    @model_validator(mode="after")
    def validate_search_pipeline_fields(self) -> "QueryRequest":
        # If pipeline is search_pipeline, these fields are required
        if self.pipeline_name == "search_pipeline":
            if not self.institution_id:
                raise ValueError("institution_id is required for search_pipeline")
            if not self.collection_name:
                raise ValueError("collection_name is required for search_pipeline")
        return self


class GenerateRequest(RequestBaseModel):
    pipeline_name: str = "gen_pipeline"
    params: GenerateParams
    debug: bool | None = False


class ExploreRequest(RequestBaseModel):
    positive: List[str]
    negative: Optional[List[str]] = None
    institution_id: str
    collection_name: str
    limit: int
    strategy: Optional[str] = "average_vector"


class CollectionRequest(RequestBaseModel):
    institution_id: str
    collection_name: str = None


class CreateCollectionRequest(CollectionRequest):
    pass


class FilterRequest(CollectionRequest):
    filters: Optional[Union[Dict[str, Any], rest.Filter]] = None


class TagsRequest(FilterRequest):
    tags: Optional[List[str]] = []


class PreprocessorParams(RequestBaseModel):
    remove_substrings: Optional[List[str]] = None
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
    separators: Optional[List[str]] = None
    split_unit: Optional[str] = None

    @field_validator("chunking_strategy")
    @classmethod
    def validate_chunking_strategy(cls, v):
        if v is None:
            return v
        allowed = {"markdown_header", "recursive", "simple"}
        if v not in allowed:
            raise ValueError(f"Invalid chunking_strategy '{v}'. Must be one of {allowed}")
        return v

    @field_validator("split_by")
    @classmethod
    def validate_split_by(cls, v):
        if v is None:
            return v
        allowed = {"function", "page", "passage", "period", "word", "line", "sentence"}
        if v not in allowed:
            raise ValueError(f"Invalid split_by '{v}'. Must be one of {allowed}")
        return v

    @field_validator("secondary_split")
    @classmethod
    def validate_secondary_split(cls, v):
        if v is None:
            return v
        allowed = {"word", "passage", "period", "line"}
        if v not in allowed:
            raise ValueError(f"Invalid secondary_split '{v}'. Must be one of {allowed}")
        return v

    @field_validator("split_unit")
    @classmethod
    def validate_split_unit(cls, v):
        if v is None:
            return v
        allowed = {"word", "char", "token"}
        if v not in allowed:
            raise ValueError(f"Invalid split_unit '{v}'. Must be one of {allowed}")
        return v

    @field_validator("split_length")
    @classmethod
    def validate_split_length(cls, v):
        if v is None:
            return v
        if v <= 0:
            raise ValueError("split_length must be a positive integer")
        return v

    @field_validator("split_overlap")
    @classmethod
    def validate_split_overlap(cls, v, values):
        if v is None:
            return v
        if v < 0:
            raise ValueError("split_overlap must be >= 0")
        split_length = values.data.get("split_length")
        if split_length is not None and v >= split_length:
            raise ValueError("split_overlap must be less than split_length")
        return v


class CssSelectorParams(RequestBaseModel):
    action: Literal["keep", "remove"]
    selector: str

    model_config = ConfigDict(extra="forbid", use_enum_values=True, ignored_types=(type(lambda: None),))

    def __getitem__(self, item):
        return self.model_dump()[item]

    def get(self, item, default=None):
        return self.model_dump().get(item, default)


class WebScraperRequest(RequestBaseModel):
    """
    A data model representing a web scraping request.

    Attributes:
        urls: A list of URLs
        uuids: A list of UUIDs
        tags: A list of tags
        institution_id: Institution ID
        collection_name: Collection name
        preprocessor_params: Parameters for preprocessing
        css_selectors: A list of dictionaries specifying CSS selectors and actions to perform on those selectors
    """

    urls: List[str] = []
    uuids: List[str] = []
    tags: List[str] = []
    institution_id: Optional[str] = None
    collection_name: Optional[str] = None
    preprocessor_params: Optional[PreprocessorParams] = None
    css_selectors: Optional[List[CssSelectorParams]] = None

    @model_validator(mode="after")
    def validate_mandatory_fields(self) -> "WebScraperRequest":
        if not self.institution_id:
            raise ValueError("institution_id is required for search_pipeline")
        if not self.collection_name:
            raise ValueError("collection_name is required for search_pipeline")
        if len(self.urls) != len(self.uuids):
            raise ValueError("Lengths of urls and uuids must be the same")
        return self


class AddDocument(BaseModel):
    model_config = ConfigDict(extra="allow")
    content: str
    embedding: list = Field(default_factory=lambda: list(np.zeros(768)))


class AddDocumentRequest(RequestBaseModel):
    data: List[AddDocument]
    uuids: List[str] = []
    institution_id: Optional[str] = None
    collection_name: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    results: Optional[Union[dict, List[dict]]] = None
    raw_results: Optional[Union[dict, List[dict]]] = None
