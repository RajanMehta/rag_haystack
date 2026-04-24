# Experiments

Exploration code layered on top of the base `haystack_api/` service. One folder per experiment. Select with `PIPELINE_CONFIG=exp_NNN_<slug> make up`.

## Index

| # | Name | Idea |
|---|------|------|
| 001 | [Smart Document Splitter](exp_001_smart_document_splitter/README.md) | Strategy-pluggable chunking (markdown-header / recursive / simple) selectable per request instead of hardcoded at pipeline-build time |
