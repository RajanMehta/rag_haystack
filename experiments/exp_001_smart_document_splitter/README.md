# Experiment 001 — Smart Document Splitter

**PIPELINE_CONFIG value:** `exp_001_smart_document_splitter`

## The idea

Haystack's stock `DocumentSplitter` is rigid: you pick `split_by` (`word` / `sentence` / `passage` / `page` / `line`) and `split_length` **at pipeline-build time**, and every document gets chopped the same way. That's fine for uniform text but poor for documents where structure matters — markdown with headers, code blocks, nested lists, headings that imply semantic boundaries.

This experiment adds `SmartDocumentSplitter`: a single Haystack `@component` that wraps three chunking strategies and lets the caller pick one **per request**, not per pipeline.

## What's new vs. base

- **New component** — [`components/smart_document_splitter.py`](components/smart_document_splitter.py): dispatcher over three strategies
  - `markdown_header` (default) — wraps Haystack's `MarkdownHeaderSplitter`; splits at `#`, `##`, …, with an optional secondary split (`word` / `passage` / `period` / `line`) for sections exceeding `split_threshold`
  - `recursive` — wraps `RecursiveDocumentSplitter`; hierarchical separators (e.g. `["\n\n", "\n", ". ", " "]`)
  - `simple` — wraps the stock `DocumentSplitter`; backward-compatible rigid behavior so you can A/B inside the same service
- **Swapped converters** — [`pipeline_config.py`](pipeline_config.py) replaces `PyPDFToDocument` / `DOCXToDocument` / `TextFileToDocument` with `MarkItDownConverter` for PDF/DOCX/TXT. Structure-aware chunking only works when upstream preserves structure, so this is part of the experiment, not an orthogonal change.
- **Runtime strategy selection** — the `chunking_strategy` field on [`PreprocessorParams`](../../haystack_api/schema.py) flows HTTP → controller → Celery task → `run_input["documentsplitter"]`. The base dispatcher uses duck-typing (`_splitter_accepts_runtime_chunking` in [`haystack_api/tasks.py`](../../haystack_api/tasks.py)), so the same running service can chunk doc A as markdown-aware and doc B recursively.

## How to run

```bash
PIPELINE_CONFIG=exp_001_smart_document_splitter make up
```

Same PDF, three strategies — diff the resulting chunks:

```bash
# markdown-aware (respects headers)
curl -F "files=@sample.pdf" \
     -F 'meta=[{"uuid":"doc-md"}]' \
     -F 'params={"index":"exp001","preprocessor_params":{"chunking_strategy":"markdown_header","split_length":200}}' \
     http://localhost:31415/file-upload

# recursive (hierarchical separators)
curl -F "files=@sample.pdf" \
     -F 'meta=[{"uuid":"doc-rec"}]' \
     -F 'params={"index":"exp001","preprocessor_params":{"chunking_strategy":"recursive","split_length":200,"separators":["\n\n","\n",". "," "]}}' \
     http://localhost:31415/file-upload

# simple (stock, for comparison)
curl -F "files=@sample.pdf" \
     -F 'meta=[{"uuid":"doc-simple"}]' \
     -F 'params={"index":"exp001","preprocessor_params":{"chunking_strategy":"simple","split_by":"word","split_length":200}}' \
     http://localhost:31415/file-upload
```

Then inspect Qdrant (points by `meta.uuid`) to compare chunk counts, boundaries, and semantic coherence.

## Reading order

1. [`pipeline_config.py`](pipeline_config.py) — the indexing DAG: file-type router → MarkItDown converters → joiner → cleaner → **`SmartDocumentSplitter`** → writer.
2. [`components/smart_document_splitter.py`](components/smart_document_splitter.py) — the dispatcher: `__init__` sets defaults; `run()` picks a strategy per call; `_run_markdown_header` / `_run_recursive` / `_run_simple` delegate to the underlying Haystack splitter.
3. [`haystack_api/schema.py`](../../haystack_api/schema.py) — `PreprocessorParams.chunking_strategy` (field + enum validator).
4. [`haystack_api/tasks.py`](../../haystack_api/tasks.py) — `_splitter_accepts_runtime_chunking` and the dispatch in `index_url` / `index_file`.

## Gotchas

- `secondary_split` only applies to `markdown_header`. Passing it with other strategies is silently ignored.
- `recursive` is **not** token-aware — `split_length` counts the `split_unit` (default: words), not tokens.
- Strategy validation lives in two places on purpose: the component raises `ValueError` for defense-in-depth, the schema rejects invalid values at request time so bad requests fail fast with a 422.
- Base (`PIPELINE_CONFIG=en_gen`) accepts the `chunking_strategy` field in requests but ignores it. The stock splitter doesn't consume it, and that's by design: the HTTP contract stays uniform across pipelines.

## Files

```
experiments/exp_001_smart_document_splitter/
├── README.md                               # this file
├── pipeline_config.py                      # indexing + search + embedding + gen pipelines
├── components/
│   └── smart_document_splitter.py          # the dispatcher
└── assets/                                 # diagrams, sample chunked outputs for the blog
```
