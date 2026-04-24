import os
import time
import uuid

import httpx
import pytest

BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:31415")
HEALTH_TIMEOUT_S = int(os.environ.get("E2E_HEALTH_TIMEOUT", "180"))


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session")
def institution_id() -> str:
    # Unique per-run so reruns don't collide on leftover state
    return f"e2e_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def collection_name() -> str:
    return "e2e_col"


@pytest.fixture(scope="session")
def client(base_url, institution_id, collection_name):
    transport = httpx.HTTPTransport(retries=2)
    with httpx.Client(base_url=base_url, timeout=120.0, transport=transport) as c:
        deadline = time.time() + HEALTH_TIMEOUT_S
        last_err = None
        while time.time() < deadline:
            try:
                if c.get("/health").status_code == 200:
                    break
            except Exception as e:
                last_err = e
            time.sleep(2)
        else:
            pytest.fail(f"haystack-api never became healthy at {base_url}: {last_err}")
        yield c
        # Best-effort cleanup
        try:
            c.delete(f"/collection/{institution_id}/{collection_name}")
        except Exception:
            pass


def wait_for_task(client: httpx.Client, task_id: str, timeout: int = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/tasks/{task_id}")
        assert r.status_code == 200, r.text
        data = r.json()
        if data.get("status") in {"SUCCESS", "FAILED", "REVOKED"}:
            return data
        time.sleep(2)
    raise AssertionError(f"task {task_id} did not finish within {timeout}s")
