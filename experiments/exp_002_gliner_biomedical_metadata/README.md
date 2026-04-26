# Experiment 002 — GLiNER Metadata Extraction

**PIPELINE_CONFIG value:** `exp_002_gliner_biomedical_metadata`

## The idea

Pure-vector RAG retrieval is blind to entity structure. A query like *"metformin for type 2 diabetes"* will pull anything semantically close — including chunks about insulin, statins, or weight loss — and rely on the LLM to filter. That's wasteful, slow, and a recall-vs-precision liability when the corpus is large.

Haystack's [`LLMMetadataExtractor`](https://docs.haystack.deepset.ai/docs/llmmetadataextractor) solves this by asking an LLM to tag entities at indexing time, then [extracting them again from the query](https://haystack.deepset.ai/blog/extracting-metadata-filter) and translating to a metadata filter. But the LLM is the bottleneck: a 70B model per chunk during ingest is expensive, slow, and adds an external API dependency.

This experiment replaces the LLM with [**GLiNER2**](https://github.com/fastino-ai/GLiNER2) (Fastino, 205M params) — one CPU-friendly model that handles NER, classification, and structured extraction in a unified interface. We use it for NER only here: extract `disease / drug / gene / symptom / anatomy / treatment` from each document at ingest, extract the same labels from the query at search time, and translate query entities into a Qdrant filter that the existing `CustomQdrantRetriever` consumes unchanged.

## What's new vs. base (`en_gen`)

- **New components** in [`components/`](components):
  - [`GLiNERMetadataExtractor`](components/gliner_metadata_extractor.py) — index-time. Runs over `documents`, calls `extract_entities(content, labels)`, writes results into `doc.meta` as list-valued fields (`meta.disease`, `meta.drug`, …). Placed **before** the splitter so meta propagates to every chunk of the same document — keeps filter recall intact when entities are mentioned in only one chunk.
  - [`GLiNERQueryFilterBuilder`](components/gliner_query_filter_builder.py) — query-time. Extracts entities from the query string and emits a Qdrant filter dict: AND across labels (drug AND disease must both match), OR within a label (any of the listed drugs match). Returns `filters=None` when nothing is found, gracefully degrading to pure semantic search.
  - [`_gliner_loader.py`](components/_gliner_loader.py) — module-level singleton cache so indexing-time and query-time components share the same loaded GLiNER2 weights instead of doubling memory.
- **Pipeline diff** in [`pipeline_config.py`](pipeline_config.py):
  - `indexing_pipeline`: `documentcleaner → metadata_extractor → documentsplitter → documentwriter` (extractor inserted between cleaner and splitter).
  - `document_indexing_pipeline`: same insertion.
  - `search_pipeline`: `query_filter_builder` runs in parallel with `text_embedder`. Its `filters` output wires into `embedding_retriever.filters`. The user-supplied `filters` from the request body now flow through `query_filter_builder.user_filters` and get AND-merged with the GLiNER-derived filter.
- **Controller diff** in [`haystack_api/controller/query.py`](../../haystack_api/controller/query.py): tiny switch — if the active pipeline contains `query_filter_builder`, redirect the request's `filters` param to it instead of the retriever. Base pipelines are unchanged.
- **New dependency** in [`pyproject.toml`](../../pyproject.toml): `gliner2>=1.3.0,<2.0.0`. After updating the lockfile (`make lock`), the model `fastino/gliner2-base-v1` (~205M) downloads from HF on first use and caches under `HF_HOME`.

## How to run

```bash
# refresh lockfile + rebuild (gliner2 is a new dep)
make lock
make build

# bring up the stack with the experiment pipeline
make exp-up EXP=002_gliner_biomedical_metadata
```

End-to-end demo:

```bash
INST=demo
COLL=biomed
INDEX="${INST}_${COLL}"

# 1. Create the collection
curl -s -X POST localhost:31415/collection/create \
     -H "content-type: application/json" \
     -d "{\"institution_id\":\"$INST\",\"collection_name\":\"$COLL\"}"

# 2. Upload the sample biomedical dataset (JSON list — one Document per item).
#    GLiNER tags each doc with disease / drug / gene / symptom / anatomy / treatment
#    BEFORE chunking, so every chunk inherits the same meta.
curl -s -X POST localhost:31415/file-upload \
     -F "files=@experiments/exp_002_gliner_biomedical_metadata/assets/biomedical_articles.json;type=application/json" \
     -F "institution_id=$INST" \
     -F "collection_name=$COLL" \
     -F 'uuids=["bio-batch-1"]' \
     -F 'tags=["biomed","exp002"]' \
     -F "split_by=word" \
     -F "split_length=200" \
     -F "split_overlap=0"

# 3. Inspect indexed meta — confirm GLiNER populated entity fields per chunk.
curl -s -X POST localhost:31415/documents/get_by_filters \
     -H "content-type: application/json" \
     -d "{\"institution_id\":\"$INST\",\"collection_name\":\"$COLL\"}" | jq '.[0].meta'

# 4. Query — entities in the prompt narrow the search space via Qdrant filter.
#    "metformin for type 2 diabetes" → filter {meta.drug in [metformin], meta.disease in [type 2 diabetes]}
curl -s -X POST localhost:31415/query \
     -H "content-type: application/json" \
     -d "{
       \"query\": \"How does metformin help with type 2 diabetes?\",
       \"pipeline_name\": \"search_pipeline\",
       \"institution_id\": \"$INST\",
       \"collection_name\": \"$COLL\",
       \"params\": {\"top_k\": 5, \"threshold\": 0.0, \"generate\": false},
       \"debug\": true
     }" | jq '.raw_results.query_filter_builder.filters, .results.documents | keys'

# 5. Negative control — query without biomedical entities. filters=None,
#    behaves identically to base search_pipeline (pure semantic).
curl -s -X POST localhost:31415/query \
     -H "content-type: application/json" \
     -d "{
       \"query\": \"What is this dataset about?\",
       \"pipeline_name\": \"search_pipeline\",
       \"institution_id\": \"$INST\",
       \"collection_name\": \"$COLL\",
       \"params\": {\"top_k\": 5, \"threshold\": 0.0, \"generate\": false},
       \"debug\": true
     }" | jq '.raw_results.query_filter_builder.filters'
```

## Reading order

1. [`pipeline_config.py`](pipeline_config.py) — see where the two new components slot into the indexing and search DAGs. Helpers (`_metadata_extractor`, `_query_filter_builder`) make the diff against base obvious.
2. [`components/_gliner_loader.py`](components/_gliner_loader.py) — singleton model cache + the `normalize_entities` helper that flattens GLiNER's `{entities: {label: [str|dict, …]}}` shape to `{label: [str, …]}`.
3. [`components/gliner_metadata_extractor.py`](components/gliner_metadata_extractor.py) — index-time pass: one GLiNER call per doc, writes `doc.meta[label] = [...]`.
4. [`components/gliner_query_filter_builder.py`](components/gliner_query_filter_builder.py) — query-time pass: builds the Qdrant filter, AND-merges with `user_filters`.
5. [`haystack_api/controller/query.py`](../../haystack_api/controller/query.py) — the small dispatch switch that routes the request's `filters` to the builder when the pipeline includes one.
6. [`assets/biomedical_articles.json`](assets/biomedical_articles.json) — sample dataset used in the demo above.

## Gotchas

- **First request is slow.** The GLiNER model is lazy-loaded on first `run()`, downloading from HF (~205M) if not yet cached. Bake into the image (à la `sentence-transformers/all-mpnet-base-v2` in [`Dockerfile`](../../Dockerfile)) if cold-start matters.
- **Truncation.** `max_input_chars` (default 4000) caps GLiNER's input. Very long documents are truncated *before* extraction, but chunking still happens on the full content. Increase if your corpus has long-form articles where entities live near the end.
- **Filter strictness.** AND-across-labels can over-filter when GLiNER picks up an entity the corpus doesn't tag for. Mitigation if you hit this: relax to OR-across-labels in `_extract_filter`, or move to a "soft filter" pattern (rerank with filter signal instead of hard-filter). Not implemented here.
- **Label set is global.** Same labels are used for indexing and query extraction by design. Different label sets would mean the query extractor picks fields the corpus never indexed.
- **Base pipelines (`PIPELINE_CONFIG=en_gen`) are unaffected.** The controller switch only fires when `query_filter_builder` exists in the active pipeline. Base requests with `filters` continue to flow straight to the retriever.

## Files

```
experiments/exp_002_gliner_biomedical_metadata/
├── README.md                                   # this file
├── pipeline_config.py                          # indexing / search / embedding / gen DAGs
├── components/
│   ├── _gliner_loader.py                       # shared model cache + entity normalization
│   ├── gliner_metadata_extractor.py            # index-time component
│   └── gliner_query_filter_builder.py          # query-time component
└── assets/
    └── biomedical_articles.json                # 12-doc sample corpus for the demo
```
