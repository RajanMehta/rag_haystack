import logging
from typing import Any, Optional

from celery.result import AsyncResult, GroupResult
from fastapi import APIRouter, FastAPI, Response
from pydantic import BaseModel, Field

from haystack_api.config import LOG_LEVEL
from haystack_api.tasks import app as celery_app
from haystack_api.utils import get_app

logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


router = APIRouter()
app: FastAPI = get_app()


class TaskResponse(BaseModel):
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    total: Optional[int] = Field(None, description="Number of total tasks")
    completed: Optional[int] = Field(None, description="Number of completed tasks")
    successful: Optional[int] = Field(None, description="Number of successful tasks")
    failed: Optional[int] = Field(None, description="Number of failed tasks")
    completion_percent: Optional[float] = Field(None, description="Percent completion")
    result: Optional[Any] = Field(None, description="Task results")


def _is_app_failed(result):
    return isinstance(result, dict) and "error" in result


@router.get("/tasks", response_model=list[TaskResponse], response_model_exclude_none=True, status_code=200)
def get_tasks(return_all: bool = False):
    """
    This endpoint allows external systems to monitor the health of the Haystack REST API.
    """
    response = []

    keys = celery_app.backend.client.keys("celery-task-*")
    for key in keys:
        a_res = AsyncResult(key.decode().removeprefix("celery-task-meta-"), app=celery_app)
        if isinstance(a_res.result, GroupResult):
            g_res = GroupResult.restore(a_res.result.id, app=celery_app)
            total = len(g_res.results)
            failed = 0
            successful = 0
            result_dict = {}

            for r in g_res.results:
                result_data = _format_result(r)
                inner_status = "FAILED" if r.failed() or _is_app_failed(result_data) else r.status

                if inner_status == "FAILED":
                    failed += 1
                elif inner_status == "SUCCESS":
                    successful += 1

                result_dict[str(r).removeprefix(f"{a_res.task_id}:")] = TaskResponse(
                    task_id=str(r), status=inner_status, result=result_data
                )

            completed = failed + successful
            completion_percent = (completed / total) * 100 if total else 0

            status = "PENDING"
            if completed != 0 and completed != total:
                status = "RUNNING"  # pragma: no cover
            elif completed == total:
                if failed > 0:
                    status = "FAILED"
                else:
                    status = "SUCCESS"

            response.append(
                TaskResponse(
                    task_id=a_res.task_id,
                    total=total,
                    completed=completed,
                    successful=successful,
                    failed=failed,
                    completion_percent=completion_percent,
                    status=status,
                    result=result_dict,
                )
            )
        # Check if this task is a chord (a_res.result is a dict)
        elif isinstance(a_res.result, dict) and "embeddings" in a_res.result:
            # Handle the group part of the chord
            group_result_ids = a_res.result["embeddings"]
            group_results = [AsyncResult(res_id, app=celery_app) for res_id in group_result_ids]
            total = len(group_results)
            failed = 0
            successful = 0
            result_dict = {}

            for r in group_results:
                result_data = _format_result(r)
                inner_status = "FAILED" if r.failed() or _is_app_failed(result_data) else r.status

                if inner_status == "FAILED":
                    failed += 1
                elif inner_status == "SUCCESS":
                    successful += 1

                result_dict[str(r.id)] = TaskResponse(task_id=str(r.id), status=inner_status, result=result_data)

            completed = failed + successful
            completion_percent = (completed / total) * 100 if total else 0

            if completed == 0:
                status = "PENDING"  # pragma: no cover
            elif completed < total:
                status = "RUNNING"  # pragma: no cover
            elif failed > 0:
                status = "FAILED"  # pragma: no cover
            else:
                status = "SUCCESS"

            response.append(
                TaskResponse(
                    task_id=a_res.task_id,
                    total=total,
                    completed=completed,
                    successful=successful,
                    failed=failed,
                    completion_percent=completion_percent,
                    status=status,
                    result=result_dict,
                )
            )

        elif return_all and ":" not in a_res.task_id:
            result_data = _format_result(a_res)
            inner_status = "FAILED" if _is_app_failed(result_data) else a_res.status
            response.append(
                TaskResponse(
                    task_id=a_res.task_id,
                    status=inner_status,
                    result=result_data,
                )
            )

    return response


@router.delete("/tasks/{task_id}", status_code=204)
def delete_tasks(task_id: str):
    """
    This endpoint allows external systems to monitor the health of the Haystack REST API.
    """
    a_res = AsyncResult(task_id, app=celery_app)
    if isinstance(a_res.result, GroupResult):
        g_res = GroupResult.restore(a_res.result.id, app=celery_app)
        g_res.revoke(terminate=True)
        g_res.forget()

    a_res.revoke(terminate=True)
    a_res.forget()

    return Response(status_code=204)


@router.get("/tasks/{task_id}", response_model=TaskResponse, response_model_exclude_none=True, status_code=200)
def get_task(task_id: str):
    """
    This endpoint allows external systems to monitor the health of the Haystack REST API.
    """
    a_res = AsyncResult(task_id, app=celery_app)
    # Check if this task is a chord (a_res.result is a dict)
    if isinstance(a_res.result, dict) and "embeddings" in a_res.result:
        # Handle the group part of the chord
        group_result_ids = a_res.result["embeddings"]
        group_results = [AsyncResult(res_id, app=celery_app) for res_id in group_result_ids]
        total = len(group_results)
        failed = 0
        successful = 0
        result_dict = {}

        for r in group_results:
            result_data = _format_result(r)
            inner_status = "FAILED" if r.failed() or _is_app_failed(result_data) else r.status

            if inner_status == "FAILED":
                failed += 1
            elif inner_status == "SUCCESS":
                successful += 1

            result_dict[str(r.id)] = TaskResponse(task_id=str(r.id), status=inner_status, result=result_data)

        completed = failed + successful
        completion_percent = (completed / total) * 100 if total else 0

        if completed == 0:
            status = "PENDING"  # pragma: no cover
        elif completed < total:
            status = "RUNNING"  # pragma: no cover
        elif failed > 0:
            status = "FAILED"  # pragma: no cover
        else:
            status = "SUCCESS"

        return TaskResponse(
            task_id=task_id,
            total=total,
            completed=completed,
            successful=successful,
            failed=failed,
            completion_percent=completion_percent,
            status=status,
            result=result_dict,
        )

    elif isinstance(a_res.result, GroupResult):
        g_res = GroupResult.restore(a_res.result.id, app=celery_app)
        total = len(g_res.results)
        failed = 0
        successful = 0
        result_dict = {}

        for r in g_res.results:
            result_data = _format_result(r)
            inner_status = "FAILED" if r.failed() or _is_app_failed(result_data) else r.status

            if inner_status == "FAILED":
                failed += 1
            elif inner_status == "SUCCESS":
                successful += 1

            result_dict[str(r).removeprefix(f"{a_res.task_id}:")] = TaskResponse(
                task_id=str(r), status=inner_status, result=result_data
            )

        completed = failed + successful
        completion_percent = (completed / total) * 100 if total else 0

        status = "PENDING"
        if completed != 0 and completed != total:
            status = "RUNNING"  # pragma: no cover
        elif completed == total:
            if failed > 0:
                status = "FAILED"
            else:
                status = "SUCCESS"

        return TaskResponse(
            task_id=a_res.task_id,
            total=total,
            completed=completed,
            successful=successful,
            failed=failed,
            completion_percent=completion_percent,
            status=status,
            result=result_dict,
        )
    # Default response for a single task
    return TaskResponse(
        task_id=task_id,
        status=a_res.state,
        result=_format_result(a_res),
    )


def _format_result(res: AsyncResult) -> Any:
    if isinstance(res.result, Exception):
        return res.traceback

    if isinstance(res.result, dict) or isinstance(res.result, list):
        return res.result

    return str(res.result)
