from typing import Any, Dict, Optional

from haystack.core.component import component
from haystack_integrations.components.generators.ollama import OllamaGenerator
from ollama import Client


@component
class CustomOllamaGenerator(OllamaGenerator):
    """
    Extended version of OllamaGenerator that allows overriding model and API key at runtime.
    """

    def run(
        self,
        prompt: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # == 0 is important; only possible if no "documents" were retrieved
        # == None if this were a clf data generation request
        if generation_kwargs.get("total_documents") == 0:
            return {"replies": [], "meta": []}

        # Extract model and api_key if provided
        model_override = generation_kwargs.pop("model", None)
        api_key = generation_kwargs.pop("api_key", None)

        # Store original values to restore later
        original_model = self.model
        original_client = self._client

        try:
            # Update model if provided
            if model_override:
                self.model = model_override

            # Update client with API key if provided
            if api_key:
                # Create a new client with the API key in headers
                headers = {"Authorization": f"Bearer {api_key}"}
                # Recreate the client with the same URL but with headers
                self._client = Client(host=self.url, timeout=self.timeout, headers=headers)

            # Call the parent class's run method
            return super(CustomOllamaGenerator, self).run(
                prompt=prompt,
                generation_kwargs=generation_kwargs,
            )

        finally:
            # Restore original values
            self.model = original_model
            if api_key:
                self._client = original_client
