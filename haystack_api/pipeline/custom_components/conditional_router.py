from haystack import Document, component
from haystack.components.routers import ConditionalRouter


def documents_to_json(documents):
    """Convert Haystack Document objects to JSON-serializable dictionaries."""
    json_docs = []

    for doc in documents:
        # Create a basic dictionary with common Document attributes
        doc_dict = {
            "id": getattr(doc, "id", None),
            "content": getattr(doc, "content", ""),
        }

        # Add meta dictionary if it exists
        if hasattr(doc, "meta"):
            # Ensure meta is also serializable
            doc_dict["meta"] = {k: v for k, v in doc.meta.items()}

        if hasattr(doc, "score"):
            doc_dict["score"] = getattr(doc, "score")

        json_docs.append(doc_dict)

    return json_docs


@component
class DocumentSerializingRouter(ConditionalRouter):
    """
    Extended version of ConditionalRouter that allows document serialization before routing.
    """

    def run(self, **kwargs):
        if "documents" in kwargs:
            if (
                isinstance(kwargs.get("documents"), list)
                and kwargs["documents"]
                and all(isinstance(doc, Document) for doc in kwargs["documents"])
            ):
                kwargs["documents"] = documents_to_json(kwargs["documents"])

        # Call the parent class's run method
        return super(DocumentSerializingRouter, self).run(**kwargs)
