"""End-to-end tests against a running haystack_api stack.

Assumes the stack is reachable at E2E_BASE_URL (default: http://localhost:31415).
The Makefile target `make e2e` builds the dev image, brings up the stack,
runs this suite, and tears it down.
"""
import json
import uuid
from pathlib import Path

import pytest

from .conftest import wait_for_task

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
PDF_SAMPLE = SAMPLES_DIR / "Sick leave.pdf"


@pytest.fixture(scope="module")
def uploaded_uuid() -> str:
    return f"u_pdf_{uuid.uuid4().hex[:8]}"


# --- Health & introspection ---


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "version" in body
    assert "memory" in body
    assert "cpu" in body


def test_initialized(client):
    r = client.get("/initialized")
    assert r.status_code == 200
    assert r.json() == {"success": True}


def test_openapi_lists_expected_routes(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    for expected in [
        "/health",
        "/initialized",
        "/query",
        "/generate",
        "/file-upload",
        "/web-scrape",
        "/documents/count",
        "/documents/get_by_filters",
        "/documents/delete_by_filters",
        "/documents/update_tags",
        "/collection/create",
        "/info/collection/{institution_id}/{collection_name}",
        "/collection/{institution_id}/{collection_name}",
        "/evaluators/information-retrieval",
        "/tasks",
        "/tasks/{task_id}",
    ]:
        assert expected in paths, f"missing route {expected}"


# --- Collection lifecycle ---


def test_collection_create(client, institution_id, collection_name):
    r = client.post(
        "/collection/create",
        json={"institution_id": institution_id, "collection_name": collection_name},
    )
    assert r.status_code == 200, r.text
    assert r.json()["result"] is True


def test_collection_create_rejects_init_from_collection(client, institution_id, collection_name):
    r = client.post(
        "/collection/create",
        json={
            "institution_id": institution_id,
            "collection_name": collection_name,
            "init_from_collection": "anything",
        },
    )
    assert r.status_code == 422
    assert "init_from_collection" in r.text


def test_collection_create_missing_institution_returns_422(client, collection_name):
    r = client.post("/collection/create", json={"collection_name": collection_name})
    assert r.status_code == 422


def test_collection_info(client, institution_id, collection_name):
    r = client.get(f"/info/collection/{institution_id}/{collection_name}")
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["status"] in {"Green", "Yellow"}
    assert body["result"]["config"]["params"]["vectorsConfig"]["params"]["size"] == "768"


def test_collection_info_missing_collection_returns_500(client, institution_id):
    r = client.get(f"/info/collection/{institution_id}")
    assert r.status_code == 500
    assert "collection_name" in r.json()["errors"][0]


def test_collection_info_nonexistent_returns_500(client, institution_id):
    r = client.get(f"/info/collection/{institution_id}/nope_{uuid.uuid4().hex[:6]}")
    assert r.status_code == 500
    assert "not found" in r.json()["errors"][0].lower()


# --- Documents (empty collection) ---


def test_documents_count_empty(client, institution_id, collection_name):
    r = client.post(
        "/documents/count",
        json={"institution_id": institution_id, "collection_name": collection_name},
    )
    assert r.status_code == 200
    assert r.json() == {"total_docs": 0, "embedded_docs": 0}


def test_documents_get_by_filters_empty(client, institution_id, collection_name):
    r = client.post(
        "/documents/get_by_filters",
        json={"institution_id": institution_id, "collection_name": collection_name},
    )
    assert r.status_code == 200
    assert r.json() == []


# --- File upload (PDF) ---


def test_file_upload_pdf_indexes_successfully(client, institution_id, collection_name, uploaded_uuid):
    assert PDF_SAMPLE.exists(), f"sample fixture missing: {PDF_SAMPLE}"
    with PDF_SAMPLE.open("rb") as fh:
        r = client.post(
            "/file-upload",
            files={"files": (PDF_SAMPLE.name, fh, "application/pdf")},
            data={
                "institution_id": institution_id,
                "collection_name": collection_name,
                "uuids": json.dumps([uploaded_uuid]),
                "tags": json.dumps(["e2e", "policy"]),
                "chunking_strategy": "simple",
                "split_by": "word",
                "split_length": "200",
                "split_overlap": "20",
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    task_id = body["task_id"]

    task = wait_for_task(client, task_id)
    assert task["status"] == "SUCCESS", task
    sub = task["result"][uploaded_uuid]
    assert sub["status"] == "SUCCESS"
    assert sub["result"]["indexing"]["embedded"] >= 1


def test_documents_count_after_upload(client, institution_id, collection_name):
    r = client.post(
        "/documents/count",
        json={"institution_id": institution_id, "collection_name": collection_name},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total_docs"] >= 1
    assert body["embedded_docs"] >= 1


def test_documents_get_by_filters_returns_uploaded_doc(client, institution_id, collection_name, uploaded_uuid):
    r = client.post(
        "/documents/get_by_filters",
        json={"institution_id": institution_id, "collection_name": collection_name},
    )
    assert r.status_code == 200
    docs = r.json()
    assert any(d["meta"].get("uuid") == uploaded_uuid for d in docs)


def test_documents_update_tags(client, institution_id, collection_name, uploaded_uuid):
    r = client.post(
        "/documents/update_tags",
        json={
            "institution_id": institution_id,
            "collection_name": collection_name,
            "filters": {
                "operator": "AND",
                "conditions": [{"field": "meta.uuid", "operator": "==", "value": uploaded_uuid}],
            },
            "tags": ["updated", "e2e"],
        },
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"success": True}

    docs = client.post(
        "/documents/get_by_filters",
        json={"institution_id": institution_id, "collection_name": collection_name},
    ).json()
    matching = [d for d in docs if d["meta"].get("uuid") == uploaded_uuid]
    assert matching, "expected doc not present"
    assert all("updated" in d["meta"]["tags"] for d in matching)


# --- Search pipeline ---


def test_query_search_pipeline_returns_documents(client, institution_id, collection_name):
    r = client.post(
        "/query",
        json={
            "query": "sick leave policy",
            "pipeline_name": "search_pipeline",
            "institution_id": institution_id,
            "collection_name": collection_name,
            "params": {"threshold": 0.0, "top_k": 3, "generate": False},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "results" in body
    docs = body["results"]["documents"]
    assert any(docs.values()), "expected at least one document group with hits"


def test_query_rejects_unknown_pipeline(client, institution_id, collection_name):
    r = client.post(
        "/query",
        json={
            "query": "test",
            "pipeline_name": "does_not_exist",
            "institution_id": institution_id,
            "collection_name": collection_name,
            "params": {"threshold": 0.0, "top_k": 1, "generate": False},
        },
    )
    assert r.status_code == 501
    assert "Pipeline" in r.json()["errors"][0]


def test_query_rejects_extra_field(client):
    r = client.post(
        "/query",
        json={
            "query": "x",
            "pipeline_name": "search_pipeline",
            "totally_unknown_field": True,
        },
    )
    assert r.status_code == 422


# --- Web scrape (async) ---


def test_web_scrape_validates_uuids_required(client, institution_id, collection_name):
    r = client.post(
        "/web-scrape",
        json={
            "urls": ["https://example.com"],
            "institution_id": institution_id,
            "collection_name": collection_name,
        },
    )
    assert r.status_code == 422
    assert "urls" in r.text and "uuids" in r.text


def test_web_scrape_submits_task(client, institution_id, collection_name):
    r = client.post(
        "/web-scrape",
        json={
            "urls": ["https://example.com"],
            "uuids": [f"u_web_{uuid.uuid4().hex[:6]}"],
            "tags": ["scraped"],
            "institution_id": institution_id,
            "collection_name": collection_name,
            "preprocessor_params": {
                "chunking_strategy": "simple",
                "split_by": "word",
                "split_length": 100,
                "split_overlap": 10,
            },
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    task = wait_for_task(client, body["task_id"], timeout=120)
    # Outcome depends on outbound network; assert deterministic terminal state.
    assert task["status"] in {"SUCCESS", "FAILED"}


# --- Tasks ---


def test_tasks_list(client):
    r = client.get("/tasks")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_task_get_unknown_returns_payload(client):
    r = client.get(f"/tasks/{uuid.uuid4()}")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body


def test_task_delete(client, institution_id, collection_name):
    r = client.post(
        "/web-scrape",
        json={
            "urls": ["https://example.com"],
            "uuids": [f"u_del_{uuid.uuid4().hex[:6]}"],
            "institution_id": institution_id,
            "collection_name": collection_name,
            "preprocessor_params": {
                "chunking_strategy": "simple",
                "split_by": "word",
                "split_length": 100,
                "split_overlap": 10,
            },
        },
    )
    task_id = r.json()["task_id"]
    d = client.delete(f"/tasks/{task_id}")
    assert d.status_code == 204


# --- Evaluator ---


def test_evaluator_information_retrieval(client):
    r = client.post(
        "/evaluators/information-retrieval",
        json={
            "ground_truth_documents": ["doc1", "doc2", "doc3"],
            "retrieved_documents": [
                {"id": "doc1", "score": 0.9},
                {"id": "doc4", "score": 0.5},
                {"id": "doc2", "score": 0.3},
            ],
            "k": 3,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, dict) and len(body) > 0


# --- Generate (only when an LLM key is configured) ---


def test_generate_requires_configuration(client):
    """Generate requires a real model + api key, so we only assert the schema validates.

    We send a payload without an API key; the endpoint must respond (200 with an error
    in the body, or 4xx/5xx) — never crash the server.
    """
    r = client.post(
        "/generate",
        json={
            "pipeline_name": "gen_pipeline",
            "params": {
                "model_type": "openai",
                "model_name": "gpt-4o-mini",
                "api_key": "sk-invalid",
                "generation_kwargs": {},
                "invocation_context": {"query": "hello"},
            },
        },
    )
    assert r.status_code in {200, 400, 401, 422, 500, 501}


# --- Cleanup (kept as the final tests so destructive ops run after queries) ---


def test_documents_delete_by_filters_clears_uploaded_doc(client, institution_id, collection_name, uploaded_uuid):
    r = client.post(
        "/documents/delete_by_filters",
        json={
            "institution_id": institution_id,
            "collection_name": collection_name,
            "filters": {
                "operator": "AND",
                "conditions": [{"field": "meta.uuid", "operator": "==", "value": uploaded_uuid}],
            },
        },
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"success": True}

    # Filter result should no longer include the deleted uuid
    docs = client.post(
        "/documents/get_by_filters",
        json={"institution_id": institution_id, "collection_name": collection_name},
    ).json()
    assert not any(d["meta"].get("uuid") == uploaded_uuid for d in docs)


def test_collection_delete(client, institution_id, collection_name):
    r = client.delete(f"/collection/{institution_id}/{collection_name}")
    assert r.status_code == 200, r.text
    assert r.json()["result"] is True

    info = client.get(f"/info/collection/{institution_id}/{collection_name}")
    assert info.status_code == 500
