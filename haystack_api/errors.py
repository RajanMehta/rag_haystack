from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any


class IngestionError(Exception):
    def __init__(
        self,
        code: str,
        error_type: str,
        message: str,
        original_error: Any = None,
        stage: str = None,
        source_info: str = None,
        http_status_code: int = None,
    ):
        self.code = code
        self.error_type = error_type
        self.message = message
        self.original_error = original_error
        self.stage = stage
        self.source_info = source_info
        self.http_status_code = http_status_code

        if isinstance(original_error, BaseException):
            original_error_message = str(original_error)
        elif original_error is not None:
            original_error_message = str(original_error)
        else:
            original_error_message = message

        self.details = {
            "original_error_message": original_error_message,
            "stage": stage,
            "source_info": source_info,
            "http_status_code": http_status_code,
        }
        self.timestamp = datetime.now(timezone.utc).isoformat()
        super().__init__(message)

    def to_error_payload(self, document_id: str = None) -> dict:
        return {
            "document_id": document_id,
            "error": {
                "code": self.code,
                "type": self.error_type,
                "message": self.message,
                "details": self.details,
                "timestamp": self.timestamp,
            },
        }


class RequestValidationError(IngestionError):
    def __init__(
        self,
        message: str = "Request validation failed.",
        original_error: Any = None,
        source_info: str = None,
    ):
        super().__init__(
            code="INVALID_INPUT",
            error_type="VALIDATION_ERROR",
            message=message,
            original_error=original_error,
            stage="REQUEST_VALIDATION",
            source_info=source_info,
            http_status_code=400,
        )


class FetchingError(IngestionError):
    def __init__(
        self,
        source_info: str,
        message: str = "Failed to fetch URL content",
        original_error: Any = None,
        http_status_code: int = None,
    ):
        try:
            code = f"HTTP_{http_status_code}_{HTTPStatus(http_status_code).phrase.upper().replace(' ', '_')}"
        except (ValueError, KeyError):
            code = "HTTP_500_INTERNAL_SERVER_ERROR"
        super().__init__(
            code=code,
            error_type="FETCHING_ERROR",
            message=message,
            original_error=original_error,
            stage="FETCHING",
            source_info=source_info,
            http_status_code=http_status_code,
        )


class ExtractionError(IngestionError):
    def __init__(
        self,
        code: str,
        stage: str,
        message: str,
        source_info: str,
        original_error: Any = None,
    ):
        super().__init__(
            code=code,
            stage=stage,
            error_type=f"{stage}_ERROR",
            message=message,
            original_error=original_error,
            source_info=source_info,
            http_status_code=404,
        )


class EmbeddingError(IngestionError):
    def __init__(self, code: str, message: str = "Failed to generate embeddings.", **kwargs):
        super().__init__(
            code=code,
            error_type=kwargs.pop("error_type", "EMBEDDING_ERROR"),
            message=message,
            stage=kwargs.pop("stage", "EMBEDDING_GENERATION"),
            http_status_code=kwargs.pop("http_status_code", 500),
            **kwargs,
        )
