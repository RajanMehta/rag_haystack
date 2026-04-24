from typing import Any, Dict, List, Optional, Union

from haystack import Document, component
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from qdrant_client.http import models


@component
class CustomQdrantRetriever(QdrantEmbeddingRetriever):
    """
    Extended version of QdrantEmbeddingRetriever that allows overriding index at runtime.
    (thread-safe approach)
    """

    @component.output_types(documents=List[Document])
    async def run_async(
        self,
        query_embedding: List[float],
        filters: Optional[Union[Dict[str, Any], models.Filter]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        group_by: Optional[str] = None,
        group_size: Optional[int] = None,
        index: Optional[str] = None,
    ):
        # Store original index
        original_index = self._document_store.index

        try:
            # Temporarily set index if provided
            if index:
                self._document_store.index = index

            # Call the parent class's run method with the temporary index
            return await super(CustomQdrantRetriever, self).run_async(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k,
                scale_score=scale_score,
                return_embedding=return_embedding,
                score_threshold=score_threshold,
                group_by=group_by,
                group_size=group_size,
            )
        finally:
            # Always restore original index
            if index:
                self._document_store.index = original_index

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Union[Dict[str, Any], models.Filter]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        group_by: Optional[str] = None,
        group_size: Optional[int] = None,
        index: Optional[str] = None,
    ):
        # Store original index
        original_index = self._document_store.index

        try:
            # Temporarily set index if provided
            if index:
                self._document_store.index = index

            # Call the parent class's run method with the temporary index
            return super(CustomQdrantRetriever, self).run(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k,
                scale_score=scale_score,
                return_embedding=return_embedding,
                score_threshold=score_threshold,
                group_by=group_by,
                group_size=group_size,
            )
        finally:
            # Always restore original index
            if index:
                self._document_store.index = original_index
