# Experiments

Each experiment is a self-contained folder built on top of the base `haystack_api/` service. Bring one up with `PIPELINE_CONFIG=exp_NNN_<slug> make up`.

## Index

| # | Name | Idea |
|---|------|------|
| 001 | [Smart Document Splitter](exp_001_smart_document_splitter/README.md) | Strategy-pluggable chunking (markdown-header / recursive / simple) selectable per request instead of hardcoded at pipeline-build time |
| 002 | [GLiNER Metadata extraction](exp_002_gliner_biomedical_metadata/README.md) | Extract biomedical entities at index-time and at query-time using GLiNER2; query entities become Qdrant filters that narrow the search space before vector retrieval |
