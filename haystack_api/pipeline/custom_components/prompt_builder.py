from typing import Any, Dict, Optional

from haystack.components.builders import PromptBuilder
from haystack.core.component import component

from haystack_api.config import OLLAMA_API_KEY, OLLAMA_MODEL


@component
class CustomPromptBuilder(PromptBuilder):
    """
    Extended version of PromptBuilder that additionally returns `generation_kwargs`
    """

    @component.output_types(prompt=str, generation_kwargs=Optional[Dict[str, Any]])
    def run(self, template: Optional[str] = None, template_variables: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Renders the prompt template with the provided variables.

        It applies the template variables to render the final prompt. You can provide variables via pipeline kwargs.
        In order to overwrite the default template, you can set the `template` parameter.
        In order to overwrite pipeline kwargs, you can set the `template_variables` parameter.

        :param template:
            An optional string template to overwrite PromptBuilder's default template. If None, the default template
            provided at initialization is used.
        :param template_variables:
            An optional dictionary of template variables to overwrite the pipeline variables.
        :param kwargs:
            Pipeline variables used for rendering the prompt.

        :returns: A dictionary with the following keys:
            - `prompt`: The updated prompt text after rendering the prompt template.

        :raises ValueError:
            If any of the required template variables is not provided.
        """
        kwargs = kwargs or {}
        template_variables = template_variables or {}
        template_variables_combined = {**kwargs, **template_variables}
        self._validate_variables(set(template_variables_combined.keys()))

        compiled_template = self.template
        if template is not None:
            compiled_template = self._env.from_string(template)

        result = compiled_template.render(template_variables_combined)

        if "generation_kwargs" not in template_variables:
            template_variables["generation_kwargs"] = {}
        template_variables["generation_kwargs"]["model"] = template_variables.get("model")
        template_variables["generation_kwargs"]["api_key"] = template_variables.get("api_key")

        if "documents" in template_variables:
            # true for search_pipeline
            template_variables["generation_kwargs"] = {
                "model": OLLAMA_MODEL,
                "api_key": OLLAMA_API_KEY,
                "temperature": 0.1,
                "top_k": 40,
                "top_p": 0.9,
                "total_documents": len(template_variables["documents"]),
            }
        else:
            # false for gen_pipeline
            template_variables["generation_kwargs"]["total_documents"] = None

        return {"prompt": result, "generation_kwargs": template_variables["generation_kwargs"]}
