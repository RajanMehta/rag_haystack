from unittest.mock import MagicMock, patch

import pytest

from haystack_api.controller.tasks import (
    _format_result,
    delete_tasks,
    get_task,
    get_tasks,
)


@pytest.fixture
def mock_celery_app():
    with patch("haystack_api.controller.tasks.celery_app") as mock_app:
        yield mock_app


@pytest.fixture
def mock_async_result():
    with patch("haystack_api.controller.tasks.AsyncResult") as mock_result:
        yield mock_result


class TestFormatResult:
    def test_format_exception_result(self):
        mock_result = MagicMock()
        mock_result.result = Exception("Test exception")
        mock_result.traceback = "Traceback info"

        result = _format_result(mock_result)
        assert result == "Traceback info"

    def test_format_dict_result(self):
        mock_result = MagicMock()
        mock_result.result = {"key": "value"}

        result = _format_result(mock_result)
        assert result == {"key": "value"}

    def test_format_list_result(self):
        mock_result = MagicMock()
        mock_result.result = ["item1", "item2"]

        result = _format_result(mock_result)
        assert result == ["item1", "item2"]

    def test_format_other_result(self):
        mock_result = MagicMock()
        mock_result.result = 123

        result = _format_result(mock_result)
        assert result == "123"


class TestGetTasks:
    def test_get_tasks_empty(self, mock_celery_app):
        mock_celery_app.backend.client.keys.return_value = []

        result = get_tasks()
        assert result == []

    def test_get_tasks_with_group_result(self, mock_celery_app, mock_async_result):
        # Setup mock for keys
        mock_celery_app.backend.client.keys.return_value = [b"celery-task-meta-task1"]

        # Setup AsyncResult mock
        async_result_instance = MagicMock()
        async_result_instance.task_id = "task1"
        group_result = MagicMock()
        group_result.id = "group1"
        async_result_instance.result = group_result
        mock_async_result.return_value = async_result_instance

        # Setup GroupResult.restore
        mock_restored_group = MagicMock()
        mock_restored_group.results = [MagicMock(), MagicMock()]
        mock_restored_group.completed_count.return_value = 2
        mock_restored_group.successful.return_value = True
        mock_restored_group.failed.return_value = False

        # Patch isinstance to return True for our GroupResult check
        original_isinstance = isinstance

        def patched_isinstance(obj, class_or_tuple):
            if obj is group_result and class_or_tuple.__name__ == "GroupResult":
                return True
            return original_isinstance(obj, class_or_tuple)

        with patch("builtins.isinstance", patched_isinstance):
            with patch("haystack_api.controller.tasks.GroupResult.restore", return_value=mock_restored_group):
                # Mock individual results in the group
                for i, result in enumerate(mock_restored_group.results):
                    result.__str__.return_value = f"task1:subtask{i}"
                    result.status = "SUCCESS"
                    result.failed.return_value = False
                    result.result = f"Result {i}"

                result = get_tasks()

                assert len(result) == 1
                assert result[0].task_id == "task1"
                assert result[0].status == "SUCCESS"
                assert result[0].total == 2
                assert result[0].completed == 2
                assert result[0].successful == 2
                assert result[0].failed == 0
                assert result[0].completion_percent == 100.0
                assert "subtask0" in result[0].result
                assert "subtask1" in result[0].result

    def test_get_tasks_with_chord(self, mock_celery_app, mock_async_result):
        # Setup mock for keys
        mock_celery_app.backend.client.keys.return_value = [b"celery-task-meta-chord1"]

        # Setup AsyncResult mock for chord
        chord_result = MagicMock()
        chord_result.task_id = "chord1"
        chord_result.result = {"embeddings": ["subtask1", "subtask2"]}

        # Setup subtask results
        subtask1 = MagicMock(
            id="subtask1", status="SUCCESS", successful=lambda: True, failed=lambda: False, result="Result 1"
        )
        subtask2 = MagicMock(
            id="subtask2", status="SUCCESS", successful=lambda: True, failed=lambda: False, result="Result 2"
        )

        # Patch isinstance to return False for GroupResult check
        original_isinstance = isinstance

        def patched_isinstance(obj, class_or_tuple):
            if obj is chord_result.result and class_or_tuple.__name__ == "GroupResult":
                return False
            return original_isinstance(obj, class_or_tuple)

        with patch("builtins.isinstance", patched_isinstance):
            mock_async_result.side_effect = lambda task_id, app: {
                "chord1": chord_result,
                "subtask1": subtask1,
                "subtask2": subtask2,
            }[task_id]

            result = get_tasks()

            assert len(result) == 1
            assert result[0].task_id == "chord1"
            assert result[0].status == "SUCCESS"
            assert result[0].total == 2
            assert result[0].completed == 2
            assert result[0].successful == 2
            assert result[0].failed == 0
            assert result[0].completion_percent == 100.0
            assert "subtask1" in result[0].result
            assert "subtask2" in result[0].result

    def test_get_tasks_with_return_all(self, mock_celery_app, mock_async_result):
        # Setup mock for keys
        mock_celery_app.backend.client.keys.return_value = [b"celery-task-meta-task1"]

        # Setup AsyncResult mock
        async_result_instance = MagicMock()
        async_result_instance.task_id = "task1"
        async_result_instance.status = "SUCCESS"
        async_result_instance.result = "Simple result"
        mock_async_result.return_value = async_result_instance

        # Patch isinstance to return False for GroupResult check
        original_isinstance = isinstance

        def patched_isinstance(obj, class_or_tuple):
            if obj is async_result_instance.result and class_or_tuple.__name__ == "GroupResult":
                return False
            return original_isinstance(obj, class_or_tuple)

        with patch("builtins.isinstance", patched_isinstance):
            result = get_tasks(return_all=True)

            assert len(result) == 1
            assert result[0].task_id == "task1"
            assert result[0].status == "SUCCESS"
            assert result[0].result == "Simple result"

    def test_get_tasks_with_failed_group_result(self, mock_celery_app, mock_async_result):
        # Setup mock for keys
        mock_celery_app.backend.client.keys.return_value = [b"celery-task-meta-task1"]

        # Setup AsyncResult mock
        async_result_instance = MagicMock()
        async_result_instance.task_id = "task1"
        group_result = MagicMock()
        group_result.id = "group1"
        async_result_instance.result = group_result
        mock_async_result.return_value = async_result_instance

        # Setup GroupResult.restore with failed tasks
        mock_restored_group = MagicMock()
        mock_restored_group.results = [MagicMock(), MagicMock()]
        mock_restored_group.completed_count.return_value = 1
        mock_restored_group.successful.return_value = False
        mock_restored_group.failed.return_value = True

        # Patch isinstance to return True for our GroupResult check
        original_isinstance = isinstance

        def patched_isinstance(obj, class_or_tuple):
            if obj is group_result and class_or_tuple.__name__ == "GroupResult":
                return True
            return original_isinstance(obj, class_or_tuple)

        with patch("builtins.isinstance", patched_isinstance):
            with patch("haystack_api.controller.tasks.GroupResult.restore", return_value=mock_restored_group):
                # Mock individual results in the group - one success, one failure
                mock_restored_group.results[0].__str__.return_value = "task1:subtask0"
                mock_restored_group.results[0].status = "SUCCESS"
                mock_restored_group.results[0].failed.return_value = False
                mock_restored_group.results[0].result = "Result 0"

                mock_restored_group.results[1].__str__.return_value = "task1:subtask1"
                mock_restored_group.results[1].status = "FAILED"
                mock_restored_group.results[1].failed.return_value = True
                mock_restored_group.results[1].result = Exception("Task failed")
                mock_restored_group.results[1].traceback = "Error traceback"

                result = get_tasks()

                assert len(result) == 1
                assert result[0].task_id == "task1"
                assert result[0].status == "FAILED"
                assert result[0].total == 2
                assert result[0].completed == 2
                assert result[0].successful == 1
                assert result[0].failed == 1
                assert result[0].completion_percent == 100.0

    def test_get_tasks_with_failed_chord(self, mock_celery_app, mock_async_result):
        # Setup mock for keys
        mock_celery_app.backend.client.keys.return_value = [b"celery-task-meta-chord1"]

        # Setup AsyncResult mock for chord
        chord_result = MagicMock()
        chord_result.task_id = "chord1"
        chord_result.result = {"embeddings": ["subtask1", "subtask2"]}

        # Setup subtask results - one success, one failure
        subtask1 = MagicMock(
            id="subtask1", status="SUCCESS", successful=lambda: True, failed=lambda: False, result="Result 1"
        )
        subtask2 = MagicMock(
            id="subtask2",
            status="FAILED",
            successful=lambda: False,
            failed=lambda: True,
            result=Exception("Task failed"),
            traceback="Error traceback",
        )

        # Patch isinstance to handle both checks correctly
        original_isinstance = isinstance

        def patched_isinstance(obj, class_or_tuple):
            if obj is chord_result.result:
                if class_or_tuple.__name__ == "GroupResult":
                    return False
                elif class_or_tuple.__name__ == "dict":
                    return True
            return original_isinstance(obj, class_or_tuple)

        with patch("builtins.isinstance", patched_isinstance):
            mock_async_result.side_effect = lambda task_id, app: {
                "chord1": chord_result,
                "subtask1": subtask1,
                "subtask2": subtask2,
            }[task_id]

            result = get_tasks()

            assert len(result) == 1
            assert result[0].task_id == "chord1"
            assert result[0].status == "FAILED"
            assert result[0].total == 2
            assert result[0].completed == 2
            assert result[0].successful == 1
            assert result[0].failed == 1
            assert result[0].completion_percent == 100.0


class TestGetTask:
    def test_get_task_simple(self, mock_async_result):
        # Setup AsyncResult mock
        async_result_instance = MagicMock()
        async_result_instance.task_id = "task1"
        async_result_instance.state = "SUCCESS"
        async_result_instance.result = "Simple result"
        mock_async_result.return_value = async_result_instance

        # Patch isinstance to return False for GroupResult check
        original_isinstance = isinstance

        def patched_isinstance(obj, class_or_tuple):
            if obj is async_result_instance.result and class_or_tuple.__name__ == "GroupResult":
                return False
            return original_isinstance(obj, class_or_tuple)

        with patch("builtins.isinstance", patched_isinstance):
            result = get_task("task1")

            assert result.task_id == "task1"
            assert result.status == "SUCCESS"
            assert result.result == "Simple result"

    def test_get_task_group_result(self, mock_async_result):
        # Setup AsyncResult mock
        async_result_instance = MagicMock()
        async_result_instance.task_id = "task1"
        group_result = MagicMock()
        group_result.id = "group1"
        async_result_instance.result = group_result
        mock_async_result.return_value = async_result_instance

        # Setup GroupResult.restore
        mock_restored_group = MagicMock()
        mock_restored_group.results = [MagicMock(), MagicMock()]
        mock_restored_group.completed_count.return_value = 2
        mock_restored_group.successful.return_value = True
        mock_restored_group.failed.return_value = False

        # Patch isinstance to return True for our GroupResult check
        original_isinstance = isinstance

        def patched_isinstance(obj, class_or_tuple):
            if obj is group_result and class_or_tuple.__name__ == "GroupResult":
                return True
            return original_isinstance(obj, class_or_tuple)

        with patch("builtins.isinstance", patched_isinstance):
            with patch("haystack_api.controller.tasks.GroupResult.restore", return_value=mock_restored_group):
                # Mock individual results in the group
                for i, result in enumerate(mock_restored_group.results):
                    result.__str__.return_value = f"task1:subtask{i}"
                    result.status = "SUCCESS"
                    result.failed.return_value = False
                    result.result = f"Result {i}"

                result = get_task("task1")

                assert result.task_id == "task1"
                assert result.status == "SUCCESS"
                assert result.total == 2
                assert result.completed == 2
                assert result.successful == 2
                assert result.failed == 0
                assert result.completion_percent == 100.0
                assert "subtask0" in result.result
                assert "subtask1" in result.result

    def test_get_task_chord(self, mock_async_result):
        # Setup AsyncResult mock for chord
        chord_result = MagicMock()
        chord_result.task_id = "chord1"
        chord_result.result = {"embeddings": ["subtask1", "subtask2"]}

        # Setup subtask results
        subtask1 = MagicMock(
            id="subtask1", status="SUCCESS", successful=lambda: True, failed=lambda: False, result="Result 1"
        )
        subtask2 = MagicMock(
            id="subtask2", status="SUCCESS", successful=lambda: True, failed=lambda: False, result="Result 2"
        )

        # Patch isinstance to handle both checks correctly
        original_isinstance = isinstance

        def patched_isinstance(obj, class_or_tuple):
            if obj is chord_result.result:
                if class_or_tuple.__name__ == "GroupResult":
                    return False
                elif class_or_tuple.__name__ == "dict":
                    return True
            return original_isinstance(obj, class_or_tuple)

        with patch("builtins.isinstance", patched_isinstance):
            mock_async_result.side_effect = lambda task_id, app: {
                "chord1": chord_result,
                "subtask1": subtask1,
                "subtask2": subtask2,
            }[task_id]

            result = get_task("chord1")

            assert result.task_id == "chord1"
            assert result.status == "SUCCESS"
            assert result.total == 2
            assert result.completed == 2
            assert result.successful == 2
            assert result.failed == 0
            assert result.completion_percent == 100.0
            assert "subtask1" in result.result
            assert "subtask2" in result.result

    def test_get_task_failed_group_result(self, mock_async_result):
        # Setup AsyncResult mock
        async_result_instance = MagicMock()
        async_result_instance.task_id = "task1"
        group_result = MagicMock()
        group_result.id = "group1"
        async_result_instance.result = group_result
        mock_async_result.return_value = async_result_instance

        # Setup GroupResult.restore with failed tasks
        mock_restored_group = MagicMock()
        mock_restored_group.results = [MagicMock(), MagicMock()]
        mock_restored_group.completed_count.return_value = 1
        mock_restored_group.successful.return_value = False
        mock_restored_group.failed.return_value = True

        # Patch isinstance to return True for our GroupResult check
        original_isinstance = isinstance

        def patched_isinstance(obj, class_or_tuple):
            if obj is group_result and class_or_tuple.__name__ == "GroupResult":
                return True
            return original_isinstance(obj, class_or_tuple)

        with patch("builtins.isinstance", patched_isinstance):
            with patch("haystack_api.controller.tasks.GroupResult.restore", return_value=mock_restored_group):
                # Mock individual results in the group - one success, one failure
                mock_restored_group.results[0].__str__.return_value = "task1:subtask0"
                mock_restored_group.results[0].status = "SUCCESS"
                mock_restored_group.results[0].failed.return_value = False
                mock_restored_group.results[0].result = "Result 0"

                mock_restored_group.results[1].__str__.return_value = "task1:subtask1"
                mock_restored_group.results[1].status = "FAILED"
                mock_restored_group.results[1].failed.return_value = True
                mock_restored_group.results[1].result = Exception("Task failed")
                mock_restored_group.results[1].traceback = "Error traceback"

                result = get_task("task1")

                assert result.task_id == "task1"
                assert result.status == "FAILED"
                assert result.total == 2
                assert result.completed == 2
                assert result.successful == 1
                assert result.failed == 1
                assert result.completion_percent == 100.0

    def test_get_task_failed_chord(self, mock_async_result):
        # Setup AsyncResult mock for chord
        chord_result = MagicMock()
        chord_result.task_id = "chord1"
        chord_result.result = {"embeddings": ["subtask1", "subtask2"]}

        # Setup subtask results - one success, one failure
        subtask1 = MagicMock(
            id="subtask1", status="SUCCESS", successful=lambda: True, failed=lambda: False, result="Result 1"
        )
        subtask2 = MagicMock(
            id="subtask2",
            status="FAILED",
            successful=lambda: False,
            failed=lambda: True,
            result=Exception("Task failed"),
            traceback="Error traceback",
        )

        # Patch isinstance to handle both checks correctly
        original_isinstance = isinstance

        def patched_isinstance(obj, class_or_tuple):
            if obj is chord_result.result:
                if class_or_tuple.__name__ == "GroupResult":
                    return False
                elif class_or_tuple.__name__ == "dict":
                    return True
            return original_isinstance(obj, class_or_tuple)

        with patch("builtins.isinstance", patched_isinstance):
            mock_async_result.side_effect = lambda task_id, app: {
                "chord1": chord_result,
                "subtask1": subtask1,
                "subtask2": subtask2,
            }[task_id]

            result = get_task("chord1")

            assert result.task_id == "chord1"
            assert result.status == "FAILED"
            assert result.total == 2
            assert result.completed == 2
            assert result.successful == 1
            assert result.failed == 1
            assert result.completion_percent == 100.0


class TestDeleteTasks:
    def test_delete_task_simple(self, mock_async_result):
        # Setup AsyncResult mock
        async_result_instance = MagicMock()
        async_result_instance.result = "Simple result"
        mock_async_result.return_value = async_result_instance

        # Patch isinstance to return False for GroupResult check
        original_isinstance = isinstance

        def patched_isinstance(obj, class_or_tuple):
            if obj is async_result_instance.result and class_or_tuple.__name__ == "GroupResult":
                return False
            return original_isinstance(obj, class_or_tuple)

        with patch("builtins.isinstance", patched_isinstance):
            response = delete_tasks("task1")

            # Verify the task was revoked and forgotten
            async_result_instance.revoke.assert_called_once_with(terminate=True)
            async_result_instance.forget.assert_called_once()
            assert response.status_code == 204

    def test_delete_task_group_result(self, mock_async_result):
        # Setup AsyncResult mock
        async_result_instance = MagicMock()
        group_result = MagicMock()
        group_result.id = "group1"
        async_result_instance.result = group_result
        mock_async_result.return_value = async_result_instance

        # Setup GroupResult.restore
        mock_restored_group = MagicMock()

        # Patch isinstance to return True for our GroupResult check
        original_isinstance = isinstance

        def patched_isinstance(obj, class_or_tuple):
            if obj is group_result and class_or_tuple.__name__ == "GroupResult":
                return True
            return original_isinstance(obj, class_or_tuple)

        with patch("builtins.isinstance", patched_isinstance):
            with patch("haystack_api.controller.tasks.GroupResult.restore", return_value=mock_restored_group):
                response = delete_tasks("task1")

                # Verify the group was revoked and forgotten
                mock_restored_group.revoke.assert_called_once_with(terminate=True)
                mock_restored_group.forget.assert_called_once()

                # Verify the task was revoked and forgotten
                async_result_instance.revoke.assert_called_once_with(terminate=True)
                async_result_instance.forget.assert_called_once()

                assert response.status_code == 204
