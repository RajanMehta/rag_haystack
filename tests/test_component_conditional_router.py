from typing import Any, Dict
from unittest.mock import patch

from haystack import Document
from haystack.components.routers import ConditionalRouter

from haystack_api.pipeline.custom_components.conditional_router import (
    DocumentSerializingRouter,
    documents_to_json,
)


class TestDocumentsToJson:
    def test_empty_documents_list(self):
        """Test that an empty list returns an empty list."""
        result = documents_to_json([])
        assert result == []

    def test_basic_document_conversion(self):
        """Test conversion of a basic Document with minimal attributes."""
        doc = Document(content="Test content", id="doc1")
        result = documents_to_json([doc])

        assert len(result) == 1
        assert result[0]["id"] == "doc1"
        assert result[0]["content"] == "Test content"
        assert "meta" in result[0]
        assert isinstance(result[0]["meta"], dict)

    def test_document_with_meta(self):
        """Test conversion of a Document with metadata."""
        doc = Document(content="Test with meta", id="doc2", meta={"source": "test", "date": "2025-04-18"})
        result = documents_to_json([doc])

        assert result[0]["meta"]["source"] == "test"
        assert result[0]["meta"]["date"] == "2025-04-18"

    def test_document_with_score(self):
        """Test conversion of a Document with a score attribute."""
        doc = Document(content="Test with score", id="doc3")
        setattr(doc, "score", 0.95)

        result = documents_to_json([doc])
        assert result[0]["score"] == 0.95

    def test_multiple_documents(self):
        """Test conversion of multiple documents."""
        docs = [
            Document(content="First doc", id="doc1"),
            Document(content="Second doc", id="doc2", meta={"key": "value"}),
        ]
        setattr(docs[1], "score", 0.85)

        result = documents_to_json(docs)

        assert len(result) == 2
        assert result[0]["id"] == "doc1"
        assert result[1]["id"] == "doc2"
        assert result[1]["meta"]["key"] == "value"
        assert result[1]["score"] == 0.85


class TestDocumentSerializingRouter:
    def setup_method(self):
        """Setup method to create sample routes for all tests."""
        self.sample_routes = [
            {
                "condition": "{{ generate == true }}",
                "output": '{{ {"documents": documents, "query": query} }}',
                "output_name": "true_route",
                "output_type": Dict[str, Any],
            },
            {
                "condition": "{{ generate == false }}",
                "output": '{{ {"documents": documents, "query": query} }}',
                "output_name": "false_route",
                "output_type": Dict[str, Any],
            },
        ]

    def test_initialization(self):
        """Test that the router initializes correctly."""
        router = DocumentSerializingRouter(routes=self.sample_routes)
        assert isinstance(router, ConditionalRouter)

    def test_run_without_documents(self):
        """Test that the router works with inputs that don't contain documents."""
        router = DocumentSerializingRouter(routes=self.sample_routes)

        with patch.object(ConditionalRouter, "run", return_value={"output": "test"}) as mock_run:
            result = router.run(query="test query", generate=True)

            mock_run.assert_called_once_with(query="test query", generate=True)
            assert result == {"output": "test"}

    def test_run_with_empty_documents(self):
        """Test that the router handles empty document lists."""
        router = DocumentSerializingRouter(routes=self.sample_routes)

        with patch.object(ConditionalRouter, "run", return_value={"output": "test"}) as mock_run:
            result = router.run(documents=[], generate=True)

            mock_run.assert_called_once_with(documents=[], generate=True)
            assert result == {"output": "test"}

    def test_run_with_documents(self):
        """Test that the router serializes documents before routing."""
        router = DocumentSerializingRouter(routes=self.sample_routes)
        docs = [Document(content="Test doc", id="doc1", meta={"source": "test"})]

        with patch.object(ConditionalRouter, "run", return_value={"documents": "serialized"}) as mock_run:
            result = router.run(documents=docs, generate=True)

            # Check that documents were serialized
            expected_serialized = [{"id": "doc1", "content": "Test doc", "meta": {"source": "test"}, "score": None}]
            mock_run.assert_called_once()
            call_args = mock_run.call_args[1]
            assert "documents" in call_args
            assert call_args["documents"] == expected_serialized
            assert call_args["generate"] is True
            assert result == {"documents": "serialized"}

    def test_run_with_non_document_objects(self):
        """Test that the router doesn't modify non-Document objects."""
        router = DocumentSerializingRouter(routes=self.sample_routes)
        non_docs = ["string1", "string2"]

        with patch.object(ConditionalRouter, "run", return_value={"output": "test"}) as mock_run:
            result = router.run(documents=non_docs, generate=False)

            # Should pass through unchanged
            mock_run.assert_called_once_with(documents=non_docs, generate=False)
            assert result == {"output": "test"}

    def test_run_with_mixed_input(self):
        """Test that the router handles mixed input correctly."""
        router = DocumentSerializingRouter(routes=self.sample_routes)
        docs = [Document(content="Test doc", id="doc1")]

        with patch.object(ConditionalRouter, "run", return_value={"output": "test"}) as mock_run:
            result = router.run(documents=docs, query="test query", generate=True, other_param=123)

            # Documents should be serialized, other params passed through
            expected_serialized = [{"id": "doc1", "content": "Test doc", "meta": {}, "score": None}]
            mock_run.assert_called_once()
            call_args = mock_run.call_args[1]
            assert call_args["documents"] == expected_serialized
            assert call_args["query"] == "test query"
            assert call_args["generate"] is True
            assert call_args["other_param"] == 123
            assert result == {"output": "test"}
