import json

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from haystack_api.controller.errors.http_error import http_error_handler


@pytest.mark.asyncio
async def test_http_error_handler():
    # Arrange
    mock_request = Request({"type": "http"})
    test_detail = "Test error message"
    test_status_code = 404
    http_exception = HTTPException(status_code=test_status_code, detail=test_detail)

    # Act
    response = await http_error_handler(mock_request, http_exception)

    # Assert
    assert isinstance(response, JSONResponse)
    assert response.status_code == test_status_code

    # Parse the JSON content to verify the structure
    response_body = response.body.decode("utf-8")
    response_json = json.loads(response_body)

    assert "errors" in response_json
    assert isinstance(response_json["errors"], list)
    assert response_json["errors"][0] == test_detail


@pytest.mark.asyncio
async def test_http_error_handler_with_different_status_code():
    # Arrange
    mock_request = Request({"type": "http"})
    test_detail = "Unauthorized access"
    test_status_code = 401
    http_exception = HTTPException(status_code=test_status_code, detail=test_detail)

    # Act
    response = await http_error_handler(mock_request, http_exception)

    # Assert
    assert response.status_code == test_status_code

    response_body = response.body.decode("utf-8")
    response_json = json.loads(response_body)

    assert response_json["errors"][0] == test_detail


@pytest.mark.asyncio
async def test_http_error_handler_with_list_detail():
    # Arrange
    mock_request = Request({"type": "http"})
    test_detail = ["Multiple", "Error", "Messages"]
    test_status_code = 400
    http_exception = HTTPException(status_code=test_status_code, detail=test_detail)

    # Act
    response = await http_error_handler(mock_request, http_exception)

    # Assert
    assert response.status_code == test_status_code

    response_body = response.body.decode("utf-8")
    response_json = json.loads(response_body)

    assert response_json["errors"][0] == test_detail
