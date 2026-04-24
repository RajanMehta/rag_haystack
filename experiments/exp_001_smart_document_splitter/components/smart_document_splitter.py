from typing import Any, Dict, List, Optional

from haystack import Document, component, logging
from haystack.components.preprocessors import (
    DocumentSplitter,
    MarkdownHeaderSplitter,
    RecursiveDocumentSplitter,
)

logger = logging.getLogger(__name__)

VALID_STRATEGIES = {"markdown_header", "recursive", "simple"}

VALID_SECONDARY_SPLITS = {"word", "passage", "period", "line"}

VALID_SPLIT_BY = {"function", "page", "passage", "period", "word", "line", "sentence"}


@component
class SmartDocumentSplitter:
    """
    A unified document splitter that delegates to one of three splitting strategies at runtime.

    Strategies:
    - "markdown_header" (default): Splits at markdown headers (#, ##, etc.) with optional secondary splitting.
      Best for documents converted to markdown via MarkItDown.
    - "recursive": Applies separators hierarchically (e.g., paragraph → sentence → line → word).
      Best when custom separators are needed.
    - "simple": Legacy fixed-size splitting by word/sentence/passage/page/line/function.
      Backward compatible with the original DocumentSplitter.
    """

    def __init__(
        self,
        default_strategy: str = "markdown_header",
        split_length: int = 200,
        split_overlap: int = 0,
    ):
        if default_strategy not in VALID_STRATEGIES:
            raise ValueError(f"Invalid default_strategy '{default_strategy}'. Must be one of {VALID_STRATEGIES}")

        self.default_strategy = default_strategy
        self.split_length = split_length
        self.split_overlap = split_overlap

    @component.output_types(documents=List[Document])
    def run(
        self,
        documents: List[Document],
        chunking_strategy: Optional[str] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        # MarkdownHeaderSplitter params
        secondary_split: Optional[str] = None,
        keep_headers: Optional[bool] = None,
        split_threshold: Optional[int] = None,
        # RecursiveDocumentSplitter params
        separators: Optional[List[str]] = None,
        split_unit: Optional[str] = None,
        # DocumentSplitter params
        split_by: Optional[str] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
    ) -> Dict[str, List[Document]]:
        strategy = chunking_strategy or self.default_strategy
        length = split_length if split_length is not None else self.split_length
        overlap = split_overlap if split_overlap is not None else self.split_overlap

        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"Invalid chunking_strategy '{strategy}'. Must be one of {VALID_STRATEGIES}")

        if strategy == "markdown_header":
            return self._run_markdown_header(documents, length, overlap, secondary_split, keep_headers, split_threshold)
        elif strategy == "recursive":
            return self._run_recursive(documents, length, overlap, separators, split_unit)
        else:
            return self._run_simple(documents, length, overlap, split_by, split_respect_sentence_boundary)

    def _run_markdown_header(
        self,
        documents: List[Document],
        split_length: int,
        split_overlap: int,
        secondary_split: Optional[str],
        keep_headers: Optional[bool],
        split_threshold: Optional[int],
    ) -> Dict[str, List[Document]]:
        kwargs: Dict[str, Any] = {
            "split_length": split_length,
            "split_overlap": split_overlap,
        }
        if secondary_split is not None:
            if secondary_split not in VALID_SECONDARY_SPLITS:
                raise ValueError(
                    f"Invalid secondary_split '{secondary_split}'. Must be one of {VALID_SECONDARY_SPLITS}"
                )
            kwargs["secondary_split"] = secondary_split
        if keep_headers is not None:
            kwargs["keep_headers"] = keep_headers
        if split_threshold is not None:
            kwargs["split_threshold"] = split_threshold

        splitter = MarkdownHeaderSplitter(**kwargs)
        return splitter.run(documents=documents)

    def _run_recursive(
        self,
        documents: List[Document],
        split_length: int,
        split_overlap: int,
        separators: Optional[List[str]],
        split_unit: Optional[str],
    ) -> Dict[str, List[Document]]:
        kwargs: Dict[str, Any] = {
            "split_length": split_length,
            "split_overlap": split_overlap,
        }
        if separators is not None:
            kwargs["separators"] = separators
        if split_unit is not None:
            kwargs["split_unit"] = split_unit

        splitter = RecursiveDocumentSplitter(**kwargs)
        return splitter.run(documents=documents)

    def _run_simple(
        self,
        documents: List[Document],
        split_length: int,
        split_overlap: int,
        split_by: Optional[str],
        split_respect_sentence_boundary: Optional[bool],
    ) -> Dict[str, List[Document]]:
        kwargs: Dict[str, Any] = {
            "split_by": split_by or "word",
            "split_length": split_length,
            "split_overlap": split_overlap,
        }
        if split_by is not None and split_by not in VALID_SPLIT_BY:
            raise ValueError(f"Invalid split_by '{split_by}'. Must be one of {VALID_SPLIT_BY}")
        if split_respect_sentence_boundary is not None:
            kwargs["respect_sentence_boundary"] = split_respect_sentence_boundary

        splitter = DocumentSplitter(**kwargs)
        return splitter.run(documents=documents)
