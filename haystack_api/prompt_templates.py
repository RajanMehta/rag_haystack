# Dictionary of templates organized by pipeline_name, model_type, and model_name
import inspect

TEMPLATES = {
    "search_pipeline": {
        "default": {
            "default": """You are an information-retrieval assistant.
You will be given:
<QUERY> the user's question
<CONTEXT> background text from multiple sources, each annotated with its site name.

Your task:
- Answer the <QUERY> using only facts found in the <CONTEXT>.
- Return the answer only without over-explaining anything.
- Do not add any external knowledge or assumptions.
- If the <CONTEXT> does not contain enough information to answer, reply ONLY with: I don't know.
- Resolve any pronouns or ambiguous references in the <QUERY> or <CONTEXT> using the provided site names.

<QUERY>
{{ query }}
</QUERY>

<CONTEXT>
{% for doc in documents %}
Source: {{ doc.meta.sitename or (doc.meta.url and doc.meta.url.split('/')[2]) or doc.meta.source or "Unknown" }}
Content: {{ doc.content }}
{% endfor %}
</CONTEXT>

<ANSWER>"""
        }
    },
    "gen_pipeline": {
        "default": {
            "default": """You are a specialized utterance generation assistant for conversational AI training.

## TASK
Generate exactly {{ context['num_returned'] }} high-quality utterances based on the provided instruction{% if context['intent_description'] %} and intent description{% endif %}{% if context['sample_utterance'] %} and any provided sample utterances{% endif %}.

## INPUT DETAILS
- Instruction: {{ context['instruction'] }}
{% if context['intent_description'] %}- Intent Description: {{ context['intent_description'] }}{% endif %}
{% if context['sample_utterance'] %}
- Sample Utterances:
{% for u in context['sample_utterance'] %}
    - {{ u }}
{% endfor %}
{% endif %}

## CONSTRAINTS
1. Follow the instruction precisely — all specified slots, entities, or wording requirements must be preserved
2. Do not introduce or modify values that are not mentioned in the instruction
3. Do not include explanations, reasoning, comments, or any extra text — only output the generated utterances as shown in the OUTPUT FORMAT
4. Utterances must sound natural and conversational
5. Avoid repetitive or formulaic variations
{% if context['sample_utterance'] %}6. Treat sample utterances as strict templates — do not alter their structure unless explicitly allowed by the instruction{% endif %}
{% if context['sample_utterance'] %}7. Only use fixed values that appear in the sample utterances unless explicitly instructed otherwise{% endif %}

## ALLOWED VARIATIONS
- Modify elements only if explicitly permitted by the instruction
- Maintain the intent's meaning and purpose
- Synonyms or rephrasings are acceptable only where the instruction allows

## OUTPUT FORMAT
{% if context['num_returned'] == 1 %}
Output exactly one utterance, formatted as follows:

1. <utterance text here>


<ANSWER>
Output only the single utterance above. Do not add explanations, reasoning, comments, or any other text.
{% else %}
Output exactly {{ context['num_returned'] }} utterances, formatted as follows:

1. <first utterance>
2. <second utterance>
...
{{ context['num_returned'] }}. <last utterance>


<ANSWER>
Only output the utterances in the numbered list above.
{% endif %}"""
        }
    },
}


def get_template(pipeline_name, model_type="default", model_name="default"):
    """
    Get the appropriate template based on pipeline_name, model_type, and model_name.
    Falls back to defaults if specific combinations aren't found.
    """
    pipeline_templates = TEMPLATES.get(pipeline_name, {})

    # Try to get template for specific model_type
    model_type_templates = pipeline_templates.get(model_type, pipeline_templates.get("default", {}))

    # Try to get template for specific model_name, fall back to default
    template = model_type_templates.get(model_name, model_type_templates.get("default", ""))

    return inspect.cleandoc(template)
