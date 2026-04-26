from typing import Dict, List, Optional

from haystack import Document, component, logging

from experiments.exp_002_gliner_biomedical_metadata.components._gliner_loader import (
    DEFAULT_LABELS,
    get_model,
    normalize_entities,
)

logger = logging.getLogger(__name__)


@component
class GLiNERMetadataExtractor:
    """
    Extract biomedical entities from each Document's content using GLiNER2 and write
    them into doc.meta as list-valued fields keyed by entity label.

    Place this BEFORE the document splitter so the extracted meta propagates to every
    chunk produced from the same document. That keeps filter recall intact when a
    user query mentions an entity that only appears in some chunks.

    Init params:
        model: HF repo id of the GLiNER2 model (default: fastino/gliner2-base-v1, ~205M).
        labels: entity labels to extract. Defaults to a generic biomedical schema.
        max_input_chars: GLiNER has a finite context window. Long docs are truncated
            before extraction; chunking still happens downstream on the full content.
    """

    def __init__(
        self,
        model: str = "fastino/gliner2-base-v1",
        labels: Optional[List[str]] = None,
        max_input_chars: int = 4000,
    ):
        self.model = model
        self.labels = labels or list(DEFAULT_LABELS)
        self.max_input_chars = max_input_chars

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        if not documents:
            return {"documents": documents}

        extractor = get_model(self.model)
        for doc in documents:
            content = doc.content or ""
            if not content.strip():
                continue
            try:
                result = extractor.extract_entities(content[: self.max_input_chars], self.labels)
            except Exception as e:
                logger.warning("GLiNER extraction failed for doc %s: %s", doc.id, e)
                continue

            for label, values in normalize_entities(result, self.labels).items():
                doc.meta[label] = values

        return {"documents": documents}
