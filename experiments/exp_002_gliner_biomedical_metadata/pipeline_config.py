"""
exp_002 — GLiNER2 biomedical metadata extraction.

Diff vs. base (en_gen_config):
  * Indexing pipelines insert `metadata_extractor` between `documentcleaner` and
    `documentsplitter`. GLiNER runs once per document; chunking propagates meta to
    every chunk so filter recall is preserved across split boundaries.
  * Search pipeline inserts `query_filter_builder` upstream of `embedding_retriever`.
    The component extracts entities from the query and emits a Qdrant filter dict
    that the existing CustomQdrantRetriever consumes unchanged.

Everything else (embedder, generator, gen_pipeline, gen helpers) is identical to base.
"""

from haystack_api.config import (
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_HTTP_PORT,
)
from haystack_api.pipeline.pipeline_configs.pipeline_utils import get_model_path

GLINER_MODEL = "fastino/gliner2-base-v1"
GLINER_LABELS = ["disease", "drug", "gene", "symptom", "anatomy", "treatment"]


def _qdrant_store_params():
    return {
        "host": QDRANT_HOST,
        "port": QDRANT_HTTP_PORT,
        "grpc_port": QDRANT_GRPC_PORT,
        "recreate_index": False,
        "embedding_dim": 768,
        "similarity": "cosine",
        "prefer_grpc": True,
        "return_embedding": False,
        "timeout": 60,
        "index": "init_collection",
        "hnsw_config": {"ef_construct": 100, "m": 16},
        "https": None,
        "api_key": None,
        "prefix": None,
        "path": None,
        "force_disable_check_same_thread": False,
        "on_disk": False,
        "use_sparse_embeddings": False,
        "sparse_idf": False,
        "progress_bar": True,
        "shard_number": None,
        "replication_factor": None,
        "write_consistency_factor": None,
        "on_disk_payload": None,
        "optimizers_config": None,
        "wal_config": None,
        "quantization_config": None,
        "wait_result_from_api": True,
        "metadata": {},
        "write_batch_size": 100,
        "scroll_size": 10000,
        "payload_fields_to_index": None,
    }


def _document_writer():
    return {
        "type": "haystack.components.writers.document_writer.DocumentWriter",
        "init_parameters": {
            "document_store": {
                "type": "haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore",
                "init_parameters": _qdrant_store_params(),
            },
            "policy": "NONE",
        },
    }


def _document_cleaner():
    return {
        "type": "haystack.components.preprocessors.document_cleaner.DocumentCleaner",
        "init_parameters": {
            "remove_empty_lines": True,
            "remove_extra_whitespaces": True,
            "remove_repeated_substrings": False,
            "keep_id": False,
            "remove_substrings": None,
            "remove_regex": None,
            "unicode_normalization": None,
            "ascii_only": False,
        },
    }


def _document_splitter():
    return {
        "type": "haystack.components.preprocessors.document_splitter.DocumentSplitter",
        "init_parameters": {
            "split_by": "word",
            "split_length": 200,
            "split_overlap": 0,
            "split_threshold": 0,
            "respect_sentence_boundary": False,
            "language": "en",
            "use_split_rules": True,
            "extend_abbreviations": True,
        },
    }


def _metadata_extractor():
    return {
        "type": "experiments.exp_002_gliner_biomedical_metadata.components.gliner_metadata_extractor.GLiNERMetadataExtractor",
        "init_parameters": {
            "model": GLINER_MODEL,
            "labels": GLINER_LABELS,
            "max_input_chars": 4000,
        },
    }


def _query_filter_builder():
    return {
        "type": "experiments.exp_002_gliner_biomedical_metadata.components.gliner_query_filter_builder.GLiNERQueryFilterBuilder",
        "init_parameters": {
            "model": GLINER_MODEL,
            "labels": GLINER_LABELS,
            "meta_prefix": "meta.",
        },
    }


pipelines = [
    # --------------------------------------------------------------------------
    # Indexing pipeline (file-upload path)
    # GLiNER runs once on the full cleaned doc, before splitting.
    # --------------------------------------------------------------------------
    {
        "name": "indexing_pipeline",
        "type": "sync",
        "configs": {
            "metadata": {},
            "max_runs_per_component": 100,
            "components": {
                "filetyperouter": {
                    "type": "haystack.components.routers.file_type_router.FileTypeRouter",
                    "init_parameters": {
                        "mime_types": [
                            "application/json",
                            "application/pdf",
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            "text/plain",
                        ],
                        "additional_mimetypes": {
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"
                        },
                    },
                },
                "jsonconverter": {
                    "type": "haystack.components.converters.json.JSONConverter",
                    "init_parameters": {
                        "jq_schema": ".[]",
                        "content_key": "content",
                        "extra_meta_fields": "*",
                        "store_full_path": False,
                    },
                },
                "pypdftodocument": {
                    "type": "haystack.components.converters.pypdf.PyPDFToDocument",
                    "init_parameters": {
                        "extraction_mode": "plain",
                        "plain_mode_orientations": [0, 90, 180, 270],
                        "plain_mode_space_width": 200.0,
                        "layout_mode_space_vertically": True,
                        "layout_mode_scale_weight": 1.25,
                        "layout_mode_strip_rotated": True,
                        "layout_mode_font_height_weight": 1.0,
                        "store_full_path": False,
                    },
                },
                "docxtodocument": {
                    "type": "haystack.components.converters.docx.DOCXToDocument",
                    "init_parameters": {"table_format": "csv", "store_full_path": False},
                },
                "textfiletodocument": {
                    "type": "haystack.components.converters.txt.TextFileToDocument",
                    "init_parameters": {"encoding": "utf-8", "store_full_path": False},
                },
                "documentjoiner": {
                    "type": "haystack.components.joiners.document_joiner.DocumentJoiner",
                    "init_parameters": {
                        "join_mode": "concatenate",
                        "weights": None,
                        "top_k": None,
                        "sort_by_score": True,
                    },
                },
                "documentcleaner": _document_cleaner(),
                "metadata_extractor": _metadata_extractor(),
                "documentsplitter": _document_splitter(),
                "documentwriter": _document_writer(),
            },
            "connections": [
                {"sender": "filetyperouter.application/json", "receiver": "jsonconverter.sources"},
                {"sender": "filetyperouter.application/pdf", "receiver": "pypdftodocument.sources"},
                {
                    "sender": "filetyperouter.application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "receiver": "docxtodocument.sources",
                },
                {"sender": "filetyperouter.text/plain", "receiver": "textfiletodocument.sources"},
                {"sender": "jsonconverter.documents", "receiver": "documentjoiner.documents"},
                {"sender": "pypdftodocument.documents", "receiver": "documentjoiner.documents"},
                {"sender": "docxtodocument.documents", "receiver": "documentjoiner.documents"},
                {"sender": "textfiletodocument.documents", "receiver": "documentjoiner.documents"},
                {"sender": "documentjoiner.documents", "receiver": "documentcleaner.documents"},
                {"sender": "documentcleaner.documents", "receiver": "metadata_extractor.documents"},
                {"sender": "metadata_extractor.documents", "receiver": "documentsplitter.documents"},
                {"sender": "documentsplitter.documents", "receiver": "documentwriter.documents"},
            ],
        },
    },
    # --------------------------------------------------------------------------
    # Document indexing pipeline (web-scrape path)
    # --------------------------------------------------------------------------
    {
        "name": "document_indexing_pipeline",
        "type": "sync",
        "configs": {
            "metadata": {},
            "max_runs_per_component": 100,
            "components": {
                "documentcleaner": _document_cleaner(),
                "metadata_extractor": _metadata_extractor(),
                "documentsplitter": _document_splitter(),
                "documentwriter": _document_writer(),
            },
            "connections": [
                {"sender": "documentcleaner.documents", "receiver": "metadata_extractor.documents"},
                {"sender": "metadata_extractor.documents", "receiver": "documentsplitter.documents"},
                {"sender": "documentsplitter.documents", "receiver": "documentwriter.documents"},
            ],
        },
    },
    # --------------------------------------------------------------------------
    # Embedding pipeline (unchanged from base)
    # --------------------------------------------------------------------------
    {
        "name": "embedding_pipeline",
        "type": "sync",
        "configs": {
            "metadata": {},
            "max_runs_per_component": 100,
            "components": {
                "documentembedder": {
                    "type": "haystack.components.embedders.sentence_transformers_document_embedder.SentenceTransformersDocumentEmbedder",
                    "init_parameters": {
                        "model": get_model_path("sentence-transformers/all-mpnet-base-v2"),
                        "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
                        "prefix": "",
                        "suffix": "",
                        "batch_size": 32,
                        "progress_bar": True,
                        "normalize_embeddings": True,
                        "meta_fields_to_embed": [],
                        "embedding_separator": "\n",
                        "trust_remote_code": False,
                        "truncate_dim": None,
                        "model_kwargs": None,
                        "tokenizer_kwargs": None,
                        "config_kwargs": None,
                        "precision": "float32",
                        "encode_kwargs": None,
                    },
                },
                "documentwriter": _document_writer(),
            },
            "connections": [{"sender": "documentembedder.documents", "receiver": "documentwriter.documents"}],
        },
    },
    # --------------------------------------------------------------------------
    # Search pipeline
    # query_filter_builder runs alongside text_embedder. Its filters output is wired
    # into embedding_retriever.filters; controller passes any user-provided filters
    # through query_filter_builder.user_filters so the two get AND-merged.
    # --------------------------------------------------------------------------
    {
        "name": "search_pipeline",
        "type": "async",
        "configs": {
            "metadata": {},
            "max_runs_per_component": 100,
            "components": {
                "query_filter_builder": _query_filter_builder(),
                "text_embedder": {
                    "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
                    "init_parameters": {
                        "model": get_model_path("sentence-transformers/all-mpnet-base-v2"),
                        "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
                        "prefix": "",
                        "suffix": "",
                        "batch_size": 32,
                        "progress_bar": True,
                        "normalize_embeddings": True,
                        "trust_remote_code": False,
                        "truncate_dim": None,
                        "model_kwargs": None,
                        "tokenizer_kwargs": None,
                        "config_kwargs": None,
                        "precision": "float32",
                        "encode_kwargs": None,
                        "backend": "torch",
                    },
                },
                "embedding_retriever": {
                    "type": "haystack_api.pipeline.custom_components.qdrant_retriever.CustomQdrantRetriever",
                    "init_parameters": {
                        "document_store": {
                            "type": "haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore",
                            "init_parameters": _qdrant_store_params(),
                        },
                        "filters": None,
                        "top_k": 10,
                        "filter_policy": "replace",
                        "scale_score": False,
                        "return_embedding": False,
                        "score_threshold": None,
                        "group_by": None,
                        "group_size": None,
                    },
                },
                "document_router": {
                    "type": "haystack_api.pipeline.custom_components.conditional_router.DocumentSerializingRouter",
                    "init_parameters": {
                        "routes": [
                            {
                                "condition": "{{ generate == true }}",
                                "output": '{{ {"documents": documents, "query": query} }}',
                                "output_name": "generation_enabled_route",
                                "output_type": "typing.Dict[str, typing.Any]",
                            },
                            {
                                "condition": "{{ generate == false }}",
                                "output": '{{ {"documents": documents, "query": query} }}',
                                "output_name": "generation_disabled_route",
                                "output_type": "typing.Dict[str, typing.Any]",
                            },
                        ],
                        "custom_filters": {},
                        "unsafe": False,
                        "validate_output_type": False,
                        "optional_variables": [],
                    },
                },
                "prompt_builder": {
                    "type": "haystack_api.pipeline.custom_components.prompt_builder.CustomPromptBuilder",
                    "init_parameters": {"template": "", "variables": None, "required_variables": None},
                },
                "local_generator": {
                    "type": "haystack_api.pipeline.custom_components.ollama_generator.CustomOllamaGenerator",
                    "init_parameters": {
                        "timeout": 120,
                        "raw": False,
                        "template": None,
                        "system_prompt": None,
                        "model": OLLAMA_MODEL,
                        "url": OLLAMA_API_BASE,
                        "keep_alive": None,
                        "generation_kwargs": {},
                        "streaming_callback": None,
                    },
                },
            },
            "connections": [
                {"sender": "query_filter_builder.filters", "receiver": "embedding_retriever.filters"},
                {"sender": "text_embedder.embedding", "receiver": "embedding_retriever.query_embedding"},
                {"sender": "embedding_retriever.documents", "receiver": "document_router.documents"},
                {"sender": "document_router.generation_enabled_route", "receiver": "prompt_builder.template_variables"},
                {"sender": "prompt_builder.prompt", "receiver": "local_generator.prompt"},
                {"sender": "prompt_builder.generation_kwargs", "receiver": "local_generator.generation_kwargs"},
            ],
            "connection_type_validation": False,
        },
    },
    # --------------------------------------------------------------------------
    # Gen pipeline (unchanged from base)
    # --------------------------------------------------------------------------
    {
        "name": "gen_pipeline",
        "type": "async",
        "configs": {
            "metadata": {},
            "max_runs_per_component": 100,
            "components": {
                "model_type_router": {
                    "type": "haystack.components.routers.conditional_router.ConditionalRouter",
                    "init_parameters": {
                        "routes": [
                            {
                                "condition": "{{ params['model_type'] == 'openai' }}",
                                "output": "{{ {'context': params['invocation_context'], 'model': params['model_name'],  'api_key': params['api_key'], 'generation_kwargs': params['generation_kwargs']} }}",
                                "output_name": "openai_route",
                                "output_type": "typing.Dict[str, typing.Any]",
                            },
                            {
                                "condition": "{{ params['model_type'] == 'local' }}",
                                "output": "{{ {'context': params['invocation_context'], 'model': params['model_name'], 'api_key': params['api_key'], 'generation_kwargs': params['generation_kwargs']} }}",
                                "output_name": "local_model_route",
                                "output_type": "typing.Dict[str, typing.Any]",
                            },
                        ],
                        "custom_filters": {},
                        "unsafe": False,
                        "validate_output_type": False,
                        "optional_variables": [],
                    },
                },
                "prompt_builder_openai": {
                    "type": "haystack_api.pipeline.custom_components.prompt_builder.CustomPromptBuilder",
                    "init_parameters": {"template": "", "variables": None, "required_variables": None},
                },
                "prompt_builder_local": {
                    "type": "haystack_api.pipeline.custom_components.prompt_builder.CustomPromptBuilder",
                    "init_parameters": {"template": "", "variables": None, "required_variables": None},
                },
                "openai_generator": {
                    "type": "haystack_api.pipeline.custom_components.openai_generator.CustomOpenAIGenerator",
                    "init_parameters": {
                        "model": "test",
                        "streaming_callback": None,
                        "api_base_url": None,
                        "organization": None,
                        "generation_kwargs": {},
                        "system_prompt": None,
                        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": False},
                    },
                },
                "local_generator": {
                    "type": "haystack_api.pipeline.custom_components.ollama_generator.CustomOllamaGenerator",
                    "init_parameters": {
                        "timeout": 120,
                        "raw": False,
                        "template": None,
                        "system_prompt": None,
                        "model": OLLAMA_MODEL,
                        "url": OLLAMA_API_BASE,
                        "keep_alive": None,
                        "generation_kwargs": {},
                        "streaming_callback": None,
                    },
                },
            },
            "connections": [
                {"sender": "model_type_router.openai_route", "receiver": "prompt_builder_openai.template_variables"},
                {"sender": "model_type_router.local_model_route", "receiver": "prompt_builder_local.template_variables"},
                {"sender": "prompt_builder_openai.prompt", "receiver": "openai_generator.prompt"},
                {"sender": "prompt_builder_openai.generation_kwargs", "receiver": "openai_generator.generation_kwargs"},
                {"sender": "prompt_builder_local.prompt", "receiver": "local_generator.prompt"},
                {"sender": "prompt_builder_local.generation_kwargs", "receiver": "local_generator.generation_kwargs"},
            ],
            "connection_type_validation": False,
        },
    },
]
