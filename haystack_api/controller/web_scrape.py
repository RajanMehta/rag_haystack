from fastapi import APIRouter
from fastapi.responses import JSONResponse

from haystack_api import structlog_config
from haystack_api.schema import WebScraperRequest
from haystack_api.tasks import index_urls
from haystack_api.utils import DOCUMENT_INDEXING_PIPELINE

router = APIRouter()


@router.post("/web-scrape")
async def web_scrape(request: WebScraperRequest):
    """
    Use this endpoint to web scrape a url
    WebScraperRequest has the following params:
    :param urls: urls to webscrape
    :param institution_id: a valid institution_id
    :param collection_name: a valid collection name in which indexed file contents
                            will be stored and embedded

    preprocessor_params has the following params:
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

    params = {}  # type: ignore
    params["index"] = f"{request.institution_id}_{request.collection_name}"
    params["tags"] = request.tags

    if request.preprocessor_params:
        preprocessor_params = request.preprocessor_params.model_dump()
        params["preprocessor_params"] = preprocessor_params

    task_meta = index_urls.delay(
        request.urls,
        request.uuids,
        params,
        DOCUMENT_INDEXING_PIPELINE,
        request.css_selectors,
        structlog_contextvars=structlog_config.get_headers(),
    )

    return JSONResponse(content={"success": True, "task_id": task_meta.task_id})
