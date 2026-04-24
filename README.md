## rag_haystack

**rag_haystack** is a RAG stack that provides APIs to consume customized [haystack](https://docs.haystack.deepset.ai/docs)-pipelines. It uses [Qdrant](https://qdrant.tech/) as a document store and APIs to interact with it are also provided. This repo also holds self-contained experiments with various RAG techniques that are not part of the default pipelines.

## launch everything

1. Run `make help` to get the list of available commands through the `Makefile`
2. Ensure the file-upload directory exists with correct permissions (required for the file-upload endpoint):
```
mkdir -p haystack_api/file-upload && chmod 777 haystack_api/file-upload
```
3. Build `haystack_api` container (production image — no dev dependencies):
```
make build
```
   To build a dev-flavored image (includes pytest, ruff, etc., used by `make test`/`make coverage`):
```
make build-dev
```
4. Stand up the stack:
```
make up logs
```
5. The server should start running at port 31415. Expose it with ngrok. Open `<ngrok-url>/docs` to see Swagger API Documentation.


## Models

The pipelines rely on one HuggingFace model:

- `sentence-transformers/all-mpnet-base-v2` — document / query embedder (768-dim)

The model is **pre-downloaded during `make build`** and baked into the image at `/opt/hf_cache` (`HF_HOME`). No network traffic at startup. Rebuild the image to pick up a new revision/model.

For air-gapped or custom-model deployments, set `LOCAL_MODEL_DIR=/abs/path` in your `.env` and mount that path in a compose override. The model should live at `<LOCAL_MODEL_DIR>/sentence-transformers/all-mpnet-base-v2/`. If `LOCAL_MODEL_DIR` is set but the subdirectory is missing, the service logs a warning and falls back to the baked-in HuggingFace cache. Set `HF_TOKEN` if you hit rate limits at build time or need gated models.


## local setup

To run different development tasks like testing, code-formatting and debugging it's best to have a non-docker setup too. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

1. This repository requires `python 3.11` and `uv`. Install uv with:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Install dependencies. uv will automatically create a `.venv` in the project root and pick a compatible Python interpreter (downloading one if needed).

   For a runtime-only install (no dev tools):
```
make install
```
   For a dev install (pytest, ruff, black, isort, commitizen, etc.):
```
make install-dev
```
3. Activate the environment (optional — `uv run` works without activation):
```
source .venv/bin/activate
```
4. You should now be able to run the standalone server. Either of the following works:
```
uv run uvicorn --app-dir=haystack_api/ application:app --reload --host 0.0.0.0 --port 31415
```
```
uvicorn --app-dir=haystack_api/ application:app --reload --host 0.0.0.0 --port 31415  # if .venv is activated
```
5. You should also be able to run tests and check formatting. Run `make help` to find more.

### Updating dependencies

Edit `pyproject.toml`, then refresh the lockfile:
```
make lock
```
Commit both `pyproject.toml` and `uv.lock`.


## GPU Utilization

Haystack nodes can run on a GPU if your machine has one.
The following commands will only work if [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed on your machine.

1. Stands up the stack
```
make up-gpu
```

2. Stands up the stack in dev mode
```
make dev-gpu
```


## Experiments

The base service in `haystack_api/` is the reference RAG stack. On top of it, [`experiments/`](experiments/README.md) holds self-contained experiments — each one folder, each one idea. Pick one with `PIPELINE_CONFIG`:

```
PIPELINE_CONFIG=exp_001_smart_document_splitter make up
# or
make exp-up EXP=001_smart_document_splitter
make exp-list
```

Current experiments:

- **001 — Smart Document Splitter** ([experiments/exp_001_smart_document_splitter/](experiments/exp_001_smart_document_splitter/README.md)) — strategy-pluggable chunking (markdown-header / recursive / simple) selectable per request.


## Useful Resources

1. Haystack API Reference: https://docs.haystack.deepset.ai/reference/agent-api
2. Qdrant Documentation: https://qdrant.tech/documentation/
