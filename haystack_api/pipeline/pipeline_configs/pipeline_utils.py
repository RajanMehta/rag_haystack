"""
Collection of utilities serving the pipeline configs
"""

import logging
import os

from haystack_api.config import LOCAL_MODEL_DIR, LOG_LEVEL

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(LOG_LEVEL)


def get_model_path(model_name: str) -> str:
    """
    Resolve `model_name` to a local directory when `LOCAL_MODEL_DIR` is set and the
    subdirectory exists; otherwise return the bare HuggingFace repo id. Haystack's
    SentenceTransformers / Transformers components accept either form and will cache
    downloads under ~/.cache/huggingface/hub/ on first use, reusing them thereafter.
    """

    if LOCAL_MODEL_DIR:
        model_path = os.path.join(LOCAL_MODEL_DIR, model_name)
        if os.path.isdir(model_path):
            logger.info(f"Loading {model_name} from local repository at {model_path}")
            return model_path
        logger.warning(
            f"LOCAL_MODEL_DIR is set but {model_path} is missing; "
            f"falling back to HuggingFace download for {model_name}"
        )
    else:
        logger.info(f"LOCAL_MODEL_DIR not set; {model_name} will load from HuggingFace cache or download on first use")
    return model_name
