from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException
from haystack.components.generators import OpenAIGenerator
from haystack.core.component import component
from haystack.dataclasses import StreamingChunk
from openai import OpenAI


def _resolve_api_key(api_key) -> Optional[str]:
    """Return the string value of a Secret-like object or a raw string, or None."""
    if api_key is None:
        return None
    if hasattr(api_key, "resolve_value"):
        try:
            return api_key.resolve_value()
        except Exception:
            return None
    return api_key or None


@component
class CustomOpenAIGenerator(OpenAIGenerator):
    """
    Extended version of OpenAIGenerator that allows overriding model and API key at runtime.

    Tolerates a missing OPENAI_API_KEY at boot: when the configured api_key cannot be
    resolved to a real value, we skip the upstream client construction and defer the
    check to `run()`, which raises HTTP 400 with a clear message. This lets pipelines
    that don't exercise OpenAI boot cleanly on machines with no key configured.
    """

    def __init__(self, *args, **kwargs):
        api_key = kwargs.get("api_key") if "api_key" in kwargs else (args[0] if args else None)
        resolved = _resolve_api_key(api_key)
        if resolved:
            super(CustomOpenAIGenerator, self).__init__(*args, **kwargs)
            return

        # Deferred-init path: mirror the subset of state that `run()` and `to_dict()`
        # expect, without constructing an OpenAI() client (which would raise without a key).
        self.api_key = api_key
        self.model = kwargs.get("model", "gpt-5-mini")
        self.generation_kwargs = kwargs.get("generation_kwargs") or {}
        self.system_prompt = kwargs.get("system_prompt")
        self.streaming_callback = kwargs.get("streaming_callback")
        self.api_base_url = kwargs.get("api_base_url")
        self.organization = kwargs.get("organization")
        self.http_client_kwargs = kwargs.get("http_client_kwargs")
        self.client = None

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Extract model and api_key if provided
        model_override = generation_kwargs.pop("model", None)
        api_key = generation_kwargs.pop("api_key", None)
        generation_kwargs.pop("total_documents", None)

        # Fail-at-request: if no key was configured at boot and none was supplied
        # per-request, raise a clear 400 instead of letting it crash inside OpenAI().
        if not _resolve_api_key(api_key) and self.client is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "OPENAI_API_KEY is not configured. Set the OPENAI_API_KEY environment "
                    "variable on the service, or pass `api_key` in the request params."
                ),
            )

        # Store original values to restore later
        original_model = self.model
        original_api_key = self.api_key
        original_client = self.client

        try:
            # Update model if provided
            if model_override:
                self.model = model_override

            # Update API key if provided
            if api_key:
                self.api_key = api_key

                # Recreate client with new API key
                self.client = OpenAI(
                    api_key=api_key.resolve_value() if hasattr(api_key, "resolve_value") else api_key,
                    organization=self.organization,
                    base_url=self.api_base_url,
                    timeout=self.client.timeout,
                    max_retries=self.client.max_retries,
                )

            # Call the parent class's run method
            return super(CustomOpenAIGenerator, self).run(
                prompt=prompt,
                system_prompt=system_prompt,
                streaming_callback=streaming_callback,
                generation_kwargs=generation_kwargs,
            )
        finally:
            # Restore original values
            self.model = original_model
            self.api_key = original_api_key
            self.client = original_client
