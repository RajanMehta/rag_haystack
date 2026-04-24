# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
#
# This file is based on https://github.com/deepset-ai/haystack
# See: https://github.com/deepset-ai/haystack/blob/main/rest_api/rest_api/application.py

import logging

from haystack_api.config import LOG_LEVEL
from haystack_api.structlog_config import configure_structlog
from haystack_api.utils import get_app, get_pipelines

configure_structlog(
    set_root_logger=False,
    set_loggers=True,
    log_level=getattr(logging, LOG_LEVEL),
    override=True,
)

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger("haystack")
logging.getLogger("haystack").setLevel(LOG_LEVEL)

app = get_app()
pipelines = get_pipelines()

logger.info("Open http://127.0.0.1:31415/docs to see Swagger API Documentation.")
