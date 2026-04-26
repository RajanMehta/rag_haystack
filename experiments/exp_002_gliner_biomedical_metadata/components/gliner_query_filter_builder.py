from typing import Any, Dict, List, Optional

from haystack import component, logging

from experiments.exp_002_gliner_biomedical_metadata.components._gliner_loader import (
    DEFAULT_LABELS,
    get_model,
    normalize_entities,
)

logger = logging.getLogger(__name__)


@component
class GLiNERQueryFilterBuilder:
    """
    Extract biomedical entities from a free-text query and translate them into a
    Qdrant-compatible metadata filter dict that the existing CustomQdrantRetriever
    accepts unchanged.

    Filter shape:
        - AND across labels   ("metformin for diabetes" → drug AND disease must match)
        - OR within a label   ("aspirin or ibuprofen"  → match either drug)

    Behavior:
        - Returns filters=None when GLiNER finds no entities, so the retriever falls
          back to pure semantic search (no recall loss vs. base behavior).
        - user_filters (optional, supplied by the controller from request params) are
          AND-ed on top of GLiNER-extracted filters so callers can still constrain the
          search by uuid / tags / etc.
    """

    def __init__(
        self,
        model: str = "fastino/gliner2-base-v1",
        labels: Optional[List[str]] = None,
        meta_prefix: str = "meta.",
    ):
        self.model = model
        self.labels = labels or list(DEFAULT_LABELS)
        self.meta_prefix = meta_prefix

    @component.output_types(filters=Optional[Dict[str, Any]], query=str)
    def run(
        self,
        query: str,
        user_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        gliner_filter = self._extract_filter(query)
        merged = self._merge(gliner_filter, user_filters)
        return {"filters": merged, "query": query}

    def _extract_filter(self, query: str) -> Optional[Dict[str, Any]]:
        if not query or not query.strip():
            return None
        try:
            result = get_model(self.model).extract_entities(query, self.labels)
        except Exception as e:
            logger.warning("GLiNER query extraction failed: %s", e)
            return None

        conditions = []
        for label, values in normalize_entities(result, self.labels).items():
            conditions.append(
                {
                    "field": f"{self.meta_prefix}{label}",
                    "operator": "in",
                    "value": values,
                }
            )
        if not conditions:
            return None
        return {"operator": "AND", "conditions": conditions}

    @staticmethod
    def _merge(
        a: Optional[Dict[str, Any]],
        b: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not a and not b:
            return None
        if not a:
            return b
        if not b:
            return a
        return {"operator": "AND", "conditions": [a, b]}
