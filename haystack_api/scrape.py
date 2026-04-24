import json
import logging
import re

from bs4 import BeautifulSoup
from haystack import Document
from lxml.html.clean import Cleaner
from trafilatura import extract, extract_metadata, fetch_response
from trafilatura.utils import sanitize, trim

from haystack_api.errors import ExtractionError, FetchingError, IngestionError

LOGGER = logging.getLogger(__name__)

cleaner = Cleaner()
cleaner.javascript = True
cleaner.style = True
cleaner.remove_unknown_tags = False
HTML_CLEANR = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")


def default_extract(url, css_selectors, uuid):
    """
    Scrapes and processes content from a URL.

    Args:
        url (str): URL of the webpage to be scraped.
        css_selectors (List[Dict[str, str]]): A list of dictionaries
            specifying CSS selectors and actions to perform before scraping
        uuid (str): Associated UUID

    Returns:
        Document: A Document object containing the processed and cleaned text content along with associated metadata.

    This function performs the following steps:
        1. Fetches the HTML content from the provided URL.
        2. Processes and removes any specified elements in `css_selectors`.
        3. Uses the `extract` function to retrieve the main content as JSON, removing unnecessary metadata fields.
        4. Cleans the HTML content by removing JavaScript, CSS, and specified HTML tags.
        5. Returns the cleaned and processed content as a `Document`.
    """
    html_document = _fetch_html(url)
    extracted_metadata = _extract_metadata(html_document, url, uuid)

    if css_selectors:
        html_document = handle_css_selectors(html_document, css_selectors)

    extracted_text = _extract_text(html_document, url)

    for key in ["fingerprint", "id", "language", "categories"]:
        extracted_metadata.pop(key, None)

    try:
        default_response = Document(content=extracted_text, meta=extracted_metadata)
    except Exception as e:
        raise ExtractionError(
            code="DOCUMENT_CREATION_FAILED",
            stage="EXTRACTION",
            message="Failed to create Document object from extracted data.",
            original_error=e,
            source_info=url,
        )

    default_response.content = _clean_html(html_document, url)

    return default_response


def _fetch_html(url):
    try:
        status_code = None
        response = fetch_response(url, decode=True)
        html_document = response.html if response and response.data else None
        status_code = getattr(response, "status", 500)

        if status_code > 299:
            raise FetchingError(
                source_info=url, message=f"Fetching error HTTP {status_code}", http_status_code=status_code
            )

        if html_document is None:
            raise FetchingError(
                source_info=url, message="Fetched response is empty or missing HTML content", http_status_code=500
            )

        return html_document
    except IngestionError:
        raise
    except Exception as e:
        raise FetchingError(source_info=url, original_error=e, http_status_code=status_code)


def _extract_metadata(html_document, url, uuid):
    try:
        extracted_metadata = extract_metadata(filecontent=html_document).as_dict()
        extracted_metadata["uuid"] = uuid
        extracted_metadata["source"] = url
        extracted_metadata["extracted_tags"] = [tag for item in extracted_metadata["tags"] for tag in item.split(",")]
        return extracted_metadata
    except Exception as e:
        raise ExtractionError(
            code="METADATA_EXTRACTION_FAILED",
            stage="EXTRACTION",
            message="Failed to extract metadata from HTML content.",
            original_error=e,
            source_info=url,
        )


def _extract_text(html_document, url):
    try:
        extracted_data = extract(
            filecontent=html_document, include_tables=False, output_format="json", with_metadata=True, favor_recall=True
        )
        extracted_json = json.loads(extracted_data)
        return extracted_json["text"]
    except (TypeError, json.JSONDecodeError):  # pragma: no cover
        # Fallback to BS4 extract if Trafilatura fails
        LOGGER.warning(f"Trafilatura extraction empty or failed for URL: {url}. Falling back to BeautifulSoup.")
        try:
            soup = BeautifulSoup(html_document, "html.parser")
        except Exception as e:
            raise ExtractionError(
                code="HTML_PARSING_FAILED",
                stage="PARSING",
                message="Failed to parse HTML during fallback extraction.",
                original_error=e,
                source_info=url,
            )

        extracted_text = soup.get_text(separator="\n", strip=True)
        LOGGER.critical("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        LOGGER.critical(extracted_text)
        if not extracted_text or extracted_text == "<[document]>":
            LOGGER.warning(f"Content of extracted data is empty for url: {url}")
            raise ExtractionError(
                code="EMPTY_CONTENT",
                stage="EXTRACTION",
                message="Content of extracted data is empty.",
                source_info=url,
                original_error=Exception("Extracted text was empty or invalid"),
            )

        return extracted_text


def _clean_html(html_document, url):
    try:
        cleaner_document = cleaner.clean_html(html_document)  # removes js and css code
        cleaner_document = cleaner_document.replace("&amp;", "&")
        cleaner_document = cleaner_document.replace("[document]", "").strip()  # remove bs4 extract [document] tag
        cleaner_document = re.sub(HTML_CLEANR, "", cleaner_document)  # removes html texts
        cleaner_document = trim(cleaner_document)  # removes unnecessary spaces
        cleaner_document = sanitize(cleaner_document)  # discards invalid characters

        if not cleaner_document.strip():
            raise ExtractionError(
                code="CLEANING_EMPTY_CONTENT",
                stage="CLEANING",
                message="Content empty after HTML cleaning.",
                original_error=None,
                source_info=url,
            )

        return cleaner_document
    except IngestionError:
        raise
    except Exception as e:
        raise ExtractionError(
            code="TEXT_CLEANING_FAILED",
            stage="CLEANING",
            message="Unexpected failure during content cleaning.",
            original_error=e,
            source_info=url,
        )


def handle_css_selectors(html_document, css_selectors):
    """
    Processes HTML by applying removal or keep actions based on specified CSS selectors.
    """
    soup = BeautifulSoup(html_document, "html.parser")

    try:
        for selector_action in css_selectors:
            selector = selector_action.get("selector")
            action = selector_action.get("action").lower()

            # Validate selector is a non empty string
            if not isinstance(selector, str) or not selector.strip():
                raise ExtractionError(
                    code="INVALID_CSS_SELECTOR",
                    stage="PARSING",
                    message=f"Invalid selector: '{selector}'. Selector must be a non empty string.",
                    original_error=ValueError(f"Invalid selector: '{selector}'"),
                    source_info=None,
                )

            if action == "remove":
                remove_css_selectors(soup, selector)

            elif action == "keep":
                soup = keep_css_selectors(soup, selector)

            else:
                raise ExtractionError(
                    code="UNSUPPORTED_CSS_ACTION",
                    stage="PARSING",
                    message=f"Unsupported action: '{action}'. Expected 'remove' or 'keep'.",
                    original_error=ValueError(f"Unsupported CSS action: '{action}'"),
                    source_info=None,
                )

    except IngestionError:
        raise
    except Exception as e:
        raise ExtractionError(
            code="CSS_OPERATION_FAILED",
            stage="PARSING",
            message="An unexpected error occurred during CSS selector processing.",
            original_error=e,
            source_info=None,
        )

    html_document = str(soup)  # Update html_document

    return html_document


def remove_css_selectors(soup, selector):
    """
    Removes elements from HTML that match the given CSS selector. Raises ExtractionError if
    the selector is malformed or causes an unexpected error during the selection or removal process.
    """
    try:
        elements = soup.select(selector)
    except Exception as e:
        raise ExtractionError(
            code="CSS_OPERATION_FAILED",
            stage="PARSING",
            message=f"Failed to remove elements using selector '{selector}'",
            original_error=e,
            source_info=None,
        )

    for element in elements:
        element.decompose()  # Remove element from HTML


def rebuild_structure(match, parser="html.parser"):
    """
    Rebuilds the parent hierarchy for a matched element (copies tag names and attributes only)
    and appends a clone of the matched element (with its children) as the innermost child.
    """
    ancestors = list(match.parents)
    ancestors.reverse()
    new_soup = BeautifulSoup("", parser)
    current = None
    for anc in ancestors:
        new_tag = new_soup.new_tag(anc.name, **anc.attrs)
        if current is None:
            new_soup.append(new_tag)
        else:
            current.append(new_tag)
        current = new_tag
    match_clone = BeautifulSoup(str(match), parser).find()
    if current:
        current.append(match_clone)
    else:
        new_soup.append(match_clone)  # pragma: no cover
    return new_soup


def keep_css_selectors(soup, selector):
    """
    Returns a new BeautifulSoup object containing only the outermost elements matching the given CSS selector, with
    their parent hierarchy rebuilt (without extraneous text).
    Raises ExtractionError if no match or selector fails to parse.
    """
    try:
        matched_elements = soup.select(selector)
    except Exception as e:
        raise ExtractionError(
            code="CSS_OPERATION_FAILED",
            stage="PARSING",
            message=f"Failed to keep elements using selector '{selector}'",
            original_error=e,
            source_info=None,
        )

    if not matched_elements:
        raise ExtractionError(
            code="INVALID_CSS_SELECTOR_NO_MATCH",
            stage="PARSING",
            message=f"No elements match the CSS selector: '{selector}'",
            original_error=ValueError(f"No elements match the CSS selector: '{selector}'"),
            source_info=None,
        )

    outer_matches = []
    for element in matched_elements:
        if not any(element is not other and element in other.descendants for other in matched_elements):
            outer_matches.append(element)

    rebuilt_fragments = [str(rebuild_structure(match)) for match in outer_matches]
    container_html = "<div>" + "".join(rebuilt_fragments) + "</div>"
    return BeautifulSoup(container_html, "html.parser")
