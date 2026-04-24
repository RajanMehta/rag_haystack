import json
from unittest.mock import MagicMock, mock_open, patch

import pytest
from fastapi import UploadFile
from fastapi.responses import JSONResponse

from haystack_api.controller.file_upload import PreprocessorParams, upload_file


@pytest.fixture
def mock_upload_file():
    file_mock = MagicMock(spec=UploadFile)
    file_mock.filename = "test_file.txt"
    file_mock.file = MagicMock()
    # Configure the read method to return bytes
    file_mock.file.read.side_effect = [b"test content", b""]  # Content for first call, empty for second call
    return file_mock


@pytest.fixture
def valid_preprocessor_params():
    return PreprocessorParams(
        remove_substrings="[]",
        split_by="word",
        split_length=100,
        split_overlap=10,
        split_respect_sentence_boundary=True,
    )


@patch("haystack_api.controller.file_upload.index_files")
@patch("haystack_api.controller.file_upload.uuid.uuid4")
@patch("haystack_api.controller.file_upload.Path")
@patch("builtins.open", new_callable=mock_open)
@patch("haystack_api.controller.file_upload.shutil.copyfileobj")
async def test_upload_file_success(
    mock_copyfileobj,
    mock_file_open,
    mock_path,
    mock_uuid,
    mock_index_files,
    mock_upload_file,
    valid_preprocessor_params,
):
    # Setup
    mock_uuid.return_value.hex = "test_uuid"
    mock_path_instance = MagicMock()
    mock_path.return_value.__truediv__.return_value = mock_path_instance
    mock_path_instance.open.return_value.__enter__.return_value = mock_file_open

    mock_task = MagicMock()
    mock_task.task_id = "test_task_id"
    mock_index_files.delay.return_value = mock_task

    # Execute
    response = await upload_file(
        files=[mock_upload_file],
        institution_id="test_institution",
        collection_name="test_collection",
        tags=json.dumps(["tag1", "tag2"]),
        meta=json.dumps({"key": "value"}),
        uuids=json.dumps(["file_uuid"]),
        preprocessor_params=valid_preprocessor_params,
    )

    # Assert
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    assert json.loads(response.body) == {"success": True, "task_id": "test_task_id"}

    # Verify index_files was called with correct parameters
    mock_index_files.delay.assert_called_once()
    args, kwargs = mock_index_files.delay.call_args

    # Check file paths
    assert len(args[0]) == 1  # file_paths
    assert args[0][0] == mock_path_instance  # The file path

    # Check file metas
    assert len(args[1]) == 1  # file_metas
    assert args[1][0]["tags"] == ["tag1", "tag2"]
    assert args[1][0]["source"] == "test_file.txt"
    assert args[1][0]["embedded"] is False
    assert args[1][0]["uuid"] == "file_uuid"
    assert args[1][0]["key"] == "value"

    # Check params
    assert args[2]["index"] == "test_institution_test_collection"
    assert "preprocessor_params" in args[2]


@patch("haystack_api.controller.file_upload.index_files")
async def test_upload_file_missing_institution_or_collection(
    mock_index_files, mock_upload_file, valid_preprocessor_params
):
    # Test missing institution_id
    response = await upload_file(
        files=[mock_upload_file],
        institution_id="",
        collection_name="test_collection",
        tags="[]",
        meta="{}",
        uuids="[]",
        preprocessor_params=valid_preprocessor_params,
    )
    assert response.status_code == 400
    json_data = json.loads(response.body)
    assert "institution_id and/or collection_name are absent" in json_data["error"]["details"]["original_error_message"]

    # Test missing collection_name
    response = await upload_file(
        files=[mock_upload_file],
        institution_id="test_institution",
        collection_name="",
        tags="[]",
        meta="{}",
        uuids="[]",
        preprocessor_params=valid_preprocessor_params,
    )
    assert response.status_code == 400
    json_data = json.loads(response.body)
    assert "institution_id and/or collection_name are absent" in json_data["error"]["details"]["original_error_message"]


@patch("haystack_api.controller.file_upload.index_files")
async def test_upload_file_invalid_meta_json(mock_index_files, mock_upload_file, valid_preprocessor_params):
    # Test invalid meta JSON
    response = await upload_file(
        files=[mock_upload_file],
        institution_id="test_institution",
        collection_name="test_collection",
        tags="[]",
        meta="not_json",
        uuids=json.dumps(["file_uuid"]),
        preprocessor_params=valid_preprocessor_params,
    )
    assert response.status_code == 400


@patch("haystack_api.controller.file_upload.index_files")
async def test_upload_file_invalid_tags_json(mock_index_files, mock_upload_file, valid_preprocessor_params):
    response = await upload_file(
        files=[mock_upload_file],
        institution_id="test_institution",
        collection_name="test_collection",
        tags="not_json",
        meta="{}",
        uuids="[]",
        preprocessor_params=valid_preprocessor_params,
    )
    assert response.status_code == 400
    json_data = json.loads(response.body)
    assert (
        "tags or uuids field must be a valid JSON-stringified list"
        in json_data["error"]["details"]["original_error_message"]
    )


@patch("haystack_api.controller.file_upload.index_files")
async def test_upload_file_invalid_uuids_json(mock_index_files, mock_upload_file, valid_preprocessor_params):
    response = await upload_file(
        files=[mock_upload_file],
        institution_id="test_institution",
        collection_name="test_collection",
        tags="[]",
        meta="{}",
        uuids="not_json",
        preprocessor_params=valid_preprocessor_params,
    )
    assert response.status_code == 400
    json_data = json.loads(response.body)
    assert (
        "tags or uuids field must be a valid JSON-stringified list"
        in json_data["error"]["details"]["original_error_message"]
    )


@patch("haystack_api.controller.file_upload.index_files")
async def test_upload_file_uuids_files_mismatch(mock_index_files, mock_upload_file, valid_preprocessor_params):
    response = await upload_file(
        files=[mock_upload_file, mock_upload_file],  # Two files
        institution_id="test_institution",
        collection_name="test_collection",
        tags="[]",
        meta="{}",
        uuids='["single_uuid"]',  # But only one UUID
        preprocessor_params=valid_preprocessor_params,
    )
    assert response.status_code == 400
    json_data = json.loads(response.body)
    assert (
        "number of UUIDs provided does not match the number of files"
        in json_data["error"]["details"]["original_error_message"]
    )


@patch("haystack_api.controller.file_upload.index_files")
@patch("haystack_api.controller.file_upload.uuid.uuid4")
@patch("haystack_api.controller.file_upload.Path")
@patch("builtins.open", new_callable=mock_open)
@patch("haystack_api.controller.file_upload.shutil.copyfileobj")
async def test_upload_file_with_remove_substrings(
    mock_copyfileobj, mock_file_open, mock_path, mock_uuid, mock_index_files, mock_upload_file
):
    # Setup
    mock_uuid.return_value.hex = "test_uuid"
    mock_path_instance = MagicMock()
    mock_path.return_value.__truediv__.return_value = mock_path_instance
    mock_path_instance.open.return_value.__enter__.return_value = mock_file_open

    mock_task = MagicMock()
    mock_task.task_id = "test_task_id"
    mock_index_files.delay.return_value = mock_task

    # Create preprocessor params with remove_substrings
    preprocessor_params = PreprocessorParams(
        remove_substrings=json.dumps(["remove1", "remove2"]),
        split_by="word",
        split_length=100,
        split_overlap=10,
        split_respect_sentence_boundary=True,
    )

    # Execute
    response = await upload_file(
        files=[mock_upload_file],
        institution_id="test_institution",
        collection_name="test_collection",
        tags="[]",
        meta="{}",
        uuids=json.dumps(["file_uuid"]),
        preprocessor_params=preprocessor_params,
    )

    # Assert
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200

    # Verify preprocessor_params were correctly processed
    args, kwargs = mock_index_files.delay.call_args
    assert args[2]["preprocessor_params"]["remove_substrings"] == ["remove1", "remove2"]


@patch("haystack_api.controller.file_upload.index_files")
async def test_upload_file_invalid_remove_substrings(mock_index_files, mock_upload_file):
    # Create preprocessor params with invalid remove_substrings
    preprocessor_params = PreprocessorParams(
        remove_substrings="not_json",
        split_by="word",
        split_length=100,
        split_overlap=10,
        split_respect_sentence_boundary=True,
    )

    response = await upload_file(
        files=[mock_upload_file],
        institution_id="test_institution",
        collection_name="test_collection",
        tags="[]",
        meta="{}",
        uuids=json.dumps(["file_uuid"]),
        preprocessor_params=preprocessor_params,
    )
    assert response.status_code == 400
    json_data = json.loads(response.body)
    assert (
        "remove_substrings field must be a valid JSON-stringified list"
        in json_data["error"]["details"]["original_error_message"]
    )


@patch("haystack_api.controller.file_upload.index_files")
@patch("haystack_api.controller.file_upload.uuid.uuid4")
@patch("haystack_api.controller.file_upload.Path")
@patch("builtins.open", new_callable=mock_open)
@patch("haystack_api.controller.file_upload.shutil.copyfileobj")
@patch("haystack_api.controller.file_upload.structlog_config.get_headers")
async def test_upload_file_with_structlog_headers(
    mock_get_headers,
    mock_copyfileobj,
    mock_file_open,
    mock_path,
    mock_uuid,
    mock_index_files,
    mock_upload_file,
    valid_preprocessor_params,
):
    # Setup
    mock_uuid.return_value.hex = "test_uuid"
    mock_path_instance = MagicMock()
    mock_path.return_value.__truediv__.return_value = mock_path_instance
    mock_path_instance.open.return_value.__enter__.return_value = mock_file_open

    mock_task = MagicMock()
    mock_task.task_id = "test_task_id"
    mock_index_files.delay.return_value = mock_task

    mock_get_headers.return_value = {"request_id": "test-request-id"}

    # Execute
    response = await upload_file(
        files=[mock_upload_file],
        institution_id="test_institution",
        collection_name="test_collection",
        tags="[]",
        meta="{}",
        uuids=json.dumps(["file_uuid"]),
        preprocessor_params=valid_preprocessor_params,
    )

    # Assert
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200

    # Verify structlog headers were passed
    mock_get_headers.assert_called_once()
    args, kwargs = mock_index_files.delay.call_args
    assert kwargs["structlog_contextvars"] == {"request_id": "test-request-id"}


@patch("haystack_api.controller.file_upload.index_files")
@patch("haystack_api.controller.file_upload.uuid.uuid4")
@patch("haystack_api.controller.file_upload.Path")
@patch("builtins.open", new_callable=mock_open)
@patch("haystack_api.controller.file_upload.shutil.copyfileobj")
async def test_upload_file_multiple_files(
    mock_copyfileobj, mock_file_open, mock_path, mock_uuid, mock_index_files, valid_preprocessor_params
):
    # Setup
    mock_uuid.return_value.hex = "test_uuid"
    mock_path_instance1 = MagicMock()
    mock_path_instance2 = MagicMock()
    mock_path.return_value.__truediv__.side_effect = [mock_path_instance1, mock_path_instance2]

    mock_task = MagicMock()
    mock_task.task_id = "test_task_id"
    mock_index_files.delay.return_value = mock_task

    # Create two mock files
    file1 = MagicMock(spec=UploadFile)
    file1.filename = "test_file1.txt"
    file1.file = MagicMock()

    file2 = MagicMock(spec=UploadFile)
    file2.filename = "test_file2.txt"
    file2.file = MagicMock()

    # Execute
    response = await upload_file(
        files=[file1, file2],
        institution_id="test_institution",
        collection_name="test_collection",
        tags=json.dumps(["tag1", "tag2"]),
        meta=json.dumps({"key": "value"}),
        uuids=json.dumps(["uuid1", "uuid2"]),
        preprocessor_params=valid_preprocessor_params,
    )

    # Assert
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200

    # Verify index_files was called with correct parameters
    args, kwargs = mock_index_files.delay.call_args

    # Check file paths - should have two files
    assert len(args[0]) == 2
    assert args[0][0] == mock_path_instance1
    assert args[0][1] == mock_path_instance2

    # Check file metas - should have two entries
    assert len(args[1]) == 2
    assert args[1][0]["source"] == "test_file1.txt"
    assert args[1][0]["uuid"] == "uuid1"
    assert args[1][1]["source"] == "test_file2.txt"
    assert args[1][1]["uuid"] == "uuid2"
