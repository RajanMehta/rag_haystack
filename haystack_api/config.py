# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
#
# This file is based on https://github.com/deepset-ai/haystack
# See: https://github.com/deepset-ai/haystack/blob/main/rest_api/rest_api/config.py

import logging
import os
from pathlib import Path

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(LOG_LEVEL)

FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", str((Path(__file__).parent / "file-upload").absolute()))

PIPELINE_CONFIG = os.getenv("PIPELINE_CONFIG", "en_gen")

ROOT_PATH = os.getenv("ROOT_PATH", "/")

CONCURRENT_REQUEST_PER_WORKER = int(os.getenv("CONCURRENT_REQUEST_PER_WORKER", "4"))

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-server")
QDRANT_HTTP_PORT = os.getenv("QDRANT_HTTP_PORT", "6333")
QDRANT_GRPC_PORT = os.getenv("QDRANT_GRPC_PORT", "6334")

GPU_MEMORY_PERCENT = float(os.getenv("GPU_MEMORY_PERCENT", 100.0)) / 100
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "hf.co/unsloth/phi-4-GGUF:Q3_K_M")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

REDIS_HOST = os.getenv("REDIS_HOST", "redis-server")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")
REDIS_USER = os.getenv("REDIS_USER")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")


def get_ollama_api_base() -> str:
    """
    Returns the correct ollama base api
    Important: Use OLLAMA_HOST if you want to use ollama-server locally,
    else use OLLAMA_EXT_LINK to reach an external end point.
    Using one for the other will cause errors in platform
    """

    ollama_api_base = os.getenv("OLLAMA_EXT_LINK") or os.getenv("OLLAMA_HOST")

    if not ollama_api_base:
        logger.warning(
            """No Ollama host name provided. Use either OLLAMA_HOST to use ollama-server locally, or \
OLLAMA_EXT_LINK to reach an external ollama end point"""
        )  # pragma: no cover

    return ollama_api_base


OLLAMA_API_BASE = get_ollama_api_base()


def get_openai_api_key() -> str | None:
    """
    Returns the OPENAI_API_KEY from the environment, or None if unset.

    We intentionally do NOT fall back to a sentinel value. Callers that need
    the key (OpenAI-backed generation) validate its presence at request time
    and raise a 4xx with a clear message; pipelines that don't use OpenAI
    continue to boot normally without a key configured.
    """

    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        logger.warning(
            "OPENAI_API_KEY is not set. OpenAI-backed generation requests will fail with 400 "
            "until the key is configured or supplied per-request."
        )  # pragma: no cover
        return None

    return openai_api_key  # pragma: no cover


OPENAI_API_KEY = get_openai_api_key()
