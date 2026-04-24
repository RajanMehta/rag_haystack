import asyncio
import uuid

import pytest
import uvicorn
from httpx import AsyncClient

# Test data
TEST_INSTITUTION_ID = "test_institution"
TEST_COLLECTION_NAME = f"test_collection_{uuid.uuid4().hex[:8]}"
TEST_SPECIAL_COLLECTION = f"special-collection.with_chars_{uuid.uuid4().hex[:5]}"

# Base URL where your app is running during tests
BASE_URL = "http://localhost:31415"
uvicorn_process = None


@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module", autouse=True)
async def start_stop_app():
    """Starts and stops the FastAPI application for testing."""
    global uvicorn_process
    config = uvicorn.Config(app="haystack_api.application:app", host="0.0.0.0", port=31416, log_level="info")
    server = uvicorn.Server(config)
    uvicorn_process = asyncio.create_task(server.serve())
    await asyncio.sleep(0.1)  # Give the app a moment to start
    yield
    uvicorn_process.cancel()
    try:
        await uvicorn_process
    except asyncio.CancelledError:
        pass


@pytest.fixture(scope="module")
async def async_client():
    """Async client for testing"""
    async with AsyncClient(base_url=BASE_URL) as client:
        yield client


@pytest.fixture(scope="module", autouse=True)
async def cleanup_collections(async_client):
    """Fixture to clean up any test collections before and after tests"""
    # Clean up before tests (in case previous tests failed)
    collections_to_cleanup = [TEST_COLLECTION_NAME, TEST_SPECIAL_COLLECTION]

    for collection in collections_to_cleanup:
        try:
            await async_client.delete(f"/collection/{TEST_INSTITUTION_ID}/{collection}")
        except Exception:
            pass

    # Run the tests
    yield

    # Clean up after tests
    for collection in collections_to_cleanup:
        try:
            await async_client.delete(f"/collection/{TEST_INSTITUTION_ID}/{collection}")
        except Exception:
            pass


# Tests for /collection/create endpoint
@pytest.mark.asyncio
async def test_create_collection(async_client):
    """Test creating a new collection"""
    response = await async_client.post(
        "/collection/create",
        json={
            "institution_id": TEST_INSTITUTION_ID,
            "collection_name": TEST_COLLECTION_NAME,
        },
    )

    assert response.status_code == 200
    assert "result" in response.json()
    assert response.json()["result"] is True


@pytest.mark.asyncio
async def test_create_collection_missing_institution_id(async_client):
    """Test creating a collection with missing institution_id"""
    response = await async_client.post("/collection/create", json={"collection_name": TEST_COLLECTION_NAME})
    assert response.status_code == 422

    response = await async_client.post(
        "/collection/create",
        json={"institution_id": "", "collection_name": TEST_COLLECTION_NAME},
    )
    assert response.status_code == 500
    assert "institution_id" in response.json()["errors"][0]


@pytest.mark.asyncio
async def test_create_collection_missing_collection_name(async_client):
    """Test creating a collection with missing collection_name"""
    response = await async_client.post("/collection/create", json={"institution_id": TEST_INSTITUTION_ID})

    assert response.status_code == 500
    assert "collection_name" in response.json()["errors"][0]


@pytest.mark.asyncio
async def test_create_collection_with_special_chars(async_client):
    """Test creating a collection with special characters in name"""
    response = await async_client.post(
        "/collection/create",
        json={
            "institution_id": TEST_INSTITUTION_ID,
            "collection_name": TEST_SPECIAL_COLLECTION,
        },
    )

    assert response.status_code == 200
    assert "result" in response.json()
    assert response.json()["result"] is True


# Tests for /info/collection/{institution_id}/{collection_name} endpoint
@pytest.mark.asyncio
async def test_get_collection_info(async_client):
    """Test getting collection info"""
    # First ensure collection exists
    await async_client.post(
        "/collection/create",
        json={
            "institution_id": TEST_INSTITUTION_ID,
            "collection_name": TEST_COLLECTION_NAME,
        },
    )

    # Get collection info
    response = await async_client.get(f"/info/collection/{TEST_INSTITUTION_ID}/{TEST_COLLECTION_NAME}")

    assert response.status_code == 200
    collection_info = response.json()
    assert "result" in collection_info
    assert collection_info["result"]["status"] == "Green"


@pytest.mark.asyncio
async def test_get_collection_info_missing_collection_name(async_client):
    """Test getting collection info with missing collection_name"""
    response = await async_client.get(f"/info/collection/{TEST_INSTITUTION_ID}")

    assert response.status_code == 500
    assert "collection_name" in response.json()["errors"][0]


@pytest.mark.asyncio
async def test_get_collection_info_nonexistent_collection(async_client):
    """Test getting info for a collection that doesn't exist"""
    nonexistent_collection = f"nonexistent_{uuid.uuid4().hex[:8]}"
    response = await async_client.get(f"/info/collection/{TEST_INSTITUTION_ID}/{nonexistent_collection}")

    assert response.status_code == 500
    # The error message should indicate the collection doesn't exist
    assert "not found" in response.json()["errors"][0].lower()


# Tests for /collection/{institution_id}/{collection_name} DELETE endpoint
@pytest.mark.asyncio
async def test_delete_collection(async_client):
    """Test deleting a collection"""
    # First ensure collection exists
    await async_client.post(
        "/collection/create",
        json={
            "institution_id": TEST_INSTITUTION_ID,
            "collection_name": TEST_COLLECTION_NAME,
        },
    )

    # Delete the collection
    response = await async_client.delete(f"/collection/{TEST_INSTITUTION_ID}/{TEST_COLLECTION_NAME}")

    assert response.status_code == 200
    assert "result" in response.json()
    assert response.json()["result"] is True

    # Verify collection no longer exists
    info_response = await async_client.get(f"/info/collection/{TEST_INSTITUTION_ID}/{TEST_COLLECTION_NAME}")
    assert info_response.status_code == 500


@pytest.mark.asyncio
async def test_delete_nonexistent_collection(async_client):
    """Test deleting a collection that doesn't exist"""
    nonexistent_collection = f"nonexistent_{uuid.uuid4().hex[:8]}"
    response = await async_client.delete(f"/collection/{TEST_INSTITUTION_ID}/{nonexistent_collection}")

    assert response.status_code == 200
    assert "result" not in response.json()


@pytest.mark.asyncio
async def test_delete_collection_with_special_chars(async_client):
    """Test deleting a collection with special characters in name"""
    # First ensure collection exists
    await async_client.post(
        "/collection/create",
        json={
            "institution_id": TEST_INSTITUTION_ID,
            "collection_name": TEST_SPECIAL_COLLECTION,
        },
    )

    # Delete the collection
    response = await async_client.delete(f"/collection/{TEST_INSTITUTION_ID}/{TEST_SPECIAL_COLLECTION}")

    assert response.status_code == 200
    assert "result" in response.json()
    assert response.json()["result"] is True
