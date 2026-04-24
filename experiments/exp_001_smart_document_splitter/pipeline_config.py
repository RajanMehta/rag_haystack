from haystack_api.config import (
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_HTTP_PORT,
)
from haystack_api.pipeline.pipeline_configs.pipeline_utils import get_model_path


def _qdrant_store_params():
    """Shared Qdrant document store init parameters."""
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
    """Shared document writer component config."""
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


pipelines = [
    # --------------------------------------------------------------------------
    # Indexing Pipeline (file upload)
    # Uses MarkItDown for PDF/DOCX/TXT → Markdown conversion,
    # keeps JSONConverter for JSON files,
    # and SmartDocumentSplitter for flexible chunking strategies.
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
                "markitdown_pdf": {
                    "type": "haystack_integrations.components.converters.markitdown.markitdown_converter.MarkItDownConverter",
                    "init_parameters": {"store_full_path": False},
                },
                "markitdown_docx": {
                    "type": "haystack_integrations.components.converters.markitdown.markitdown_converter.MarkItDownConverter",
                    "init_parameters": {"store_full_path": False},
                },
                "markitdown_txt": {
                    "type": "haystack_integrations.components.converters.markitdown.markitdown_converter.MarkItDownConverter",
                    "init_parameters": {"store_full_path": False},
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
                "documentcleaner": {
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
                },
                "documentsplitter": {
                    "type": "experiments.exp_001_smart_document_splitter.components.smart_document_splitter.SmartDocumentSplitter",
                    "init_parameters": {
                        "default_strategy": "markdown_header",
                        "split_length": 200,
                        "split_overlap": 0,
                    },
                },
                "documentwriter": _document_writer(),
            },
            "connections": [
                {"sender": "filetyperouter.application/json", "receiver": "jsonconverter.sources"},
                {"sender": "filetyperouter.application/pdf", "receiver": "markitdown_pdf.sources"},
                {
                    "sender": "filetyperouter.application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "receiver": "markitdown_docx.sources",
                },
                {"sender": "filetyperouter.text/plain", "receiver": "markitdown_txt.sources"},
                {"sender": "jsonconverter.documents", "receiver": "documentjoiner.documents"},
                {"sender": "markitdown_pdf.documents", "receiver": "documentjoiner.documents"},
                {"sender": "markitdown_docx.documents", "receiver": "documentjoiner.documents"},
                {"sender": "markitdown_txt.documents", "receiver": "documentjoiner.documents"},
                {"sender": "documentjoiner.documents", "receiver": "documentcleaner.documents"},
                {"sender": "documentcleaner.documents", "receiver": "documentsplitter.documents"},
                {"sender": "documentsplitter.documents", "receiver": "documentwriter.documents"},
            ],
        },
    },
    # --------------------------------------------------------------------------
    # Document Indexing Pipeline (web scrape / raw document ingestion)
    # --------------------------------------------------------------------------
    {
        "name": "document_indexing_pipeline",
        "type": "sync",
        "configs": {
            "metadata": {},
            "max_runs_per_component": 100,
            "components": {
                "documentcleaner": {
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
                },
                "documentsplitter": {
                    "type": "experiments.exp_001_smart_document_splitter.components.smart_document_splitter.SmartDocumentSplitter",
                    "init_parameters": {
                        "default_strategy": "markdown_header",
                        "split_length": 200,
                        "split_overlap": 0,
                    },
                },
                "documentwriter": _document_writer(),
            },
            "connections": [
                {"sender": "documentcleaner.documents", "receiver": "documentsplitter.documents"},
                {"sender": "documentsplitter.documents", "receiver": "documentwriter.documents"},
            ],
        },
    },
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
    {
        "name": "search_pipeline",
        "type": "async",
        "configs": {
            "metadata": {},
            "max_runs_per_component": 100,
            "components": {
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
                {
                    "sender": "text_embedder.embedding",
                    "receiver": "embedding_retriever.query_embedding",
                },
                {
                    "sender": "embedding_retriever.documents",
                    "receiver": "document_router.documents",
                },
                {
                    "sender": "document_router.generation_enabled_route",
                    "receiver": "prompt_builder.template_variables",
                },
                {
                    "sender": "prompt_builder.prompt",
                    "receiver": "local_generator.prompt",
                },
                {"sender": "prompt_builder.generation_kwargs", "receiver": "local_generator.generation_kwargs"},
            ],
            "connection_type_validation": False,
        },
    },
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
                {
                    "sender": "model_type_router.local_model_route",
                    "receiver": "prompt_builder_local.template_variables",
                },
                {"sender": "prompt_builder_openai.prompt", "receiver": "openai_generator.prompt"},
                {"sender": "prompt_builder_openai.generation_kwargs", "receiver": "openai_generator.generation_kwargs"},
                {"sender": "prompt_builder_local.prompt", "receiver": "local_generator.prompt"},
                {"sender": "prompt_builder_local.generation_kwargs", "receiver": "local_generator.generation_kwargs"},
            ],
            "connection_type_validation": False,
        },
    },
]
