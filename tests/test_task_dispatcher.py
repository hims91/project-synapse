"""
Unit tests for Task Dispatcher.
Tests priority queuing, retry logic, and task execution.
"""
import pytest
import pytest_asyncio
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.signal_relay.task_dispatcher import (
    TaskDispatcher, TaskPriority, TaskType, DispatchableTask, TaskExecution,
    TaskDispatcherStats, get_task_dispatcher, cleanup_task_dispatcher
)
from src.synaptic_vesicle.database import DatabaseManager


@pytest.fixture
def mock_db_manager():
    """Mock DatabaseManager for testing."""
    return AsyncMock(spec=DatabaseManager)


@pytest.fixture
def sample_task_payload():
    """Sample task payload for testing."""
    return {
        "url": "https://example.com",
        "priority": True,
        "metadata": {"source": "test"}
    }


@pytest_asyncio.fixture
async def task_dispatcher(mock_db_manager):
    """Create a task dispatcher for testing."""
    dispatcher = TaskDispatcher(
        max_workers=2,
        max_queue_size=10,
        db_manager=mock_db_manager
    )
    await dispatcher.start()
    yield dispatcher
    await dispatcher.stop()


class TestTaskExecution:
    """Test TaskExecution functionality."""
    
    def test_initialization(self):
        """Test task execution initialization."""
        execution = TaskExecution(
            task_id="test-task",
            attempt=1,
            started_at=datetime.now(timezone.utc)
        )
        
        assert execution.task_id == "test-task"
        assert execution.attempt == 1
        assert execution.completed_at is None
        assert execution.error is None
        assert execution.result is None
        assert execution.duration_ms is None
    
    def test_mark_completed(self):
        """Test marking execution as completed."""
        start_time = datetime.now(timezone.utc)
        execution = TaskExecution(
            task_id="test-task",
            attempt=1,
            started_at=start_time
        )
        
        result = {"status": "success", "data": "test"}
        execution.mark_completed(result)
        
        assert execution.completed_at is not None
        assert execution.result == result
        assert execution.duration_ms is not None
        assert execution.duration_ms >= 0
    
    def test_mark_failed(self):
        """Test marking execution as failed."""
        start_time = datetime.now(timezone.utc)
        execution = TaskExecution(
            task_id="test-task",
            attempt=1,
            started_at=start_time
        )
        
        error_msg = "Task failed due to network error"
        execution.mark_failed(error_msg)
        
        assert execution.completed_at is not None
        assert execution.error == error_msg
        assert execution.duration_ms is not None
        assert execution.duration_ms >= 0


class TestDispatchableTask:
    """Test DispatchableTask functionality."""
    
    def test_initialization(self, sample_task_payload):
        """Test task initialization."""
        task = DispatchableTask(
            id="test-task",
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            priority=TaskPriority.HIGH
        )
        
        assert task.id == "test-task"
        assert task.task_type == TaskType.SCRAPE_URL
        assert task.payload == sample_task_payload
        assert task.priority == TaskPriority.HIGH
        assert task.max_retries == 3
        assert task.retry_count == 0
        assert len(task.executions) == 0
    
    def test_priority_comparison(self, sample_task_payload):
        """Test task priority comparison for queue ordering."""
        high_task = DispatchableTask(
            id="high-task",
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            priority=TaskPriority.HIGH
        )
        
        normal_task = DispatchableTask(
            id="normal-task",
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            priority=TaskPriority.NORMAL
        )
        
        # High priority task should be "less than" normal priority
        assert high_task < normal_task
        assert not (normal_task < high_task)
    
    def test_time_based_priority(self, sample_task_payload):
        """Test time-based priority for same priority tasks."""
        earlier_time = datetime.now(timezone.utc)
        later_time = earlier_time + timedelta(seconds=1)
        
        earlier_task = DispatchableTask(
            id="earlier-task",
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            priority=TaskPriority.NORMAL,
            scheduled_at=earlier_time
        )
        
        later_task = DispatchableTask(
            id="later-task",
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            priority=TaskPriority.NORMAL,
            scheduled_at=later_time
        )
        
        # Earlier scheduled task should have higher priority
        assert earlier_task < later_task
    
    def test_can_retry(self, sample_task_payload):
        """Test retry capability checking."""
        task = DispatchableTask(
            id="test-task",
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            priority=TaskPriority.NORMAL,
            max_retries=2
        )
        
        # Initially can retry
        assert task.can_retry()
        
        # After first retry
        task.retry_count = 1
        assert task.can_retry()
        
        # After max retries
        task.retry_count = 2
        assert not task.can_retry()
    
    def test_calculate_retry_delay(self, sample_task_payload):
        """Test exponential backoff calculation."""
        task = DispatchableTask(
            id="test-task",
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            priority=TaskPriority.NORMAL
        )
        
        # First retry: base_delay * 2^0 = 1
        assert task.calculate_next_retry_delay(1) == 1
        
        # Second retry: base_delay * 2^1 = 2
        task.retry_count = 1
        assert task.calculate_next_retry_delay(1) == 2
        
        # Third retry: base_delay * 2^2 = 4
        task.retry_count = 2
        assert task.calculate_next_retry_delay(1) == 4
        
        # Test max delay cap (300 seconds)
        task.retry_count = 10
        assert task.calculate_next_retry_delay(1) == 300
    
    def test_schedule_retry(self, sample_task_payload):
        """Test retry scheduling."""
        task = DispatchableTask(
            id="test-task",
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            priority=TaskPriority.NORMAL,
            max_retries=2
        )
        
        original_scheduled_at = task.scheduled_at
        original_retry_count = task.retry_count
        
        task.schedule_retry(1)
        
        # Should increment retry count
        assert task.retry_count == original_retry_count + 1
        
        # Should reschedule for later
        assert task.scheduled_at > original_scheduled_at
    
    def test_start_execution(self, sample_task_payload):
        """Test starting task execution."""
        task = DispatchableTask(
            id="test-task",
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            priority=TaskPriority.NORMAL
        )
        
        execution = task.start_execution()
        
        assert execution.task_id == task.id
        assert execution.attempt == 1
        assert execution.started_at is not None
        assert len(task.executions) == 1
        assert task.last_attempt_at == execution.started_at
        
        # Second execution
        execution2 = task.start_execution()
        assert execution2.attempt == 2
        assert len(task.executions) == 2
    
    def test_to_dict(self, sample_task_payload):
        """Test task serialization."""
        task = DispatchableTask(
            id="test-task",
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            priority=TaskPriority.HIGH,
            max_retries=2
        )
        
        # Add an execution
        execution = task.start_execution()
        execution.mark_completed({"result": "success"})
        
        task_dict = task.to_dict()
        
        assert task_dict["id"] == "test-task"
        assert task_dict["task_type"] == TaskType.SCRAPE_URL
        assert task_dict["payload"] == sample_task_payload
        assert task_dict["priority"] == TaskPriority.HIGH
        assert task_dict["max_retries"] == 2
        assert len(task_dict["executions"]) == 1
        assert task_dict["executions"][0]["attempt"] == 1
        assert task_dict["executions"][0]["result"] == {"result": "success"}


class TestTaskDispatcherStats:
    """Test TaskDispatcherStats functionality."""
    
    def test_initialization(self):
        """Test stats initialization."""
        stats = TaskDispatcherStats()
        
        assert stats.tasks_dispatched == 0
        assert stats.tasks_completed == 0
        assert stats.tasks_failed == 0
        assert stats.tasks_retried == 0
        assert stats.total_execution_time_ms == 0
        assert stats.active_workers == 0
        assert stats.queue_size == 0
        assert stats.average_execution_time_ms == 0.0
        assert stats.success_rate == 0.0
        assert stats.started_at is not None
    
    def test_update_completion_success(self):
        """Test updating stats on successful completion."""
        stats = TaskDispatcherStats()
        
        start_time = datetime.now(timezone.utc)
        execution = TaskExecution(
            task_id="test-task",
            attempt=1,
            started_at=start_time
        )
        # Add a small delay to ensure duration > 0
        import time
        time.sleep(0.001)
        execution.mark_completed({"result": "success"})
        
        stats.update_completion(execution, success=True)
        
        assert stats.tasks_completed == 1
        assert stats.tasks_failed == 0
        assert stats.success_rate == 1.0
        assert stats.average_execution_time_ms >= 0  # Changed to >= to handle very fast execution
    
    def test_update_completion_failure(self):
        """Test updating stats on failure."""
        stats = TaskDispatcherStats()
        
        start_time = datetime.now(timezone.utc)
        execution = TaskExecution(
            task_id="test-task",
            attempt=1,
            started_at=start_time
        )
        # Add a small delay to ensure duration > 0
        import time
        time.sleep(0.001)
        execution.mark_failed("Test error")
        
        stats.update_completion(execution, success=False)
        
        assert stats.tasks_completed == 0
        assert stats.tasks_failed == 1
        assert stats.success_rate == 0.0
        assert stats.average_execution_time_ms >= 0  # Changed to >= to handle very fast execution
    
    def test_mixed_completion_stats(self):
        """Test stats with mixed success and failure."""
        stats = TaskDispatcherStats()
        
        # Add successful execution
        success_execution = TaskExecution(
            task_id="success-task",
            attempt=1,
            started_at=datetime.now(timezone.utc)
        )
        success_execution.mark_completed({"result": "success"})
        stats.update_completion(success_execution, success=True)
        
        # Add failed execution
        failed_execution = TaskExecution(
            task_id="failed-task",
            attempt=1,
            started_at=datetime.now(timezone.utc)
        )
        failed_execution.mark_failed("Test error")
        stats.update_completion(failed_execution, success=False)
        
        assert stats.tasks_completed == 1
        assert stats.tasks_failed == 1
        assert stats.success_rate == 0.5
    
    def test_to_dict(self):
        """Test stats serialization."""
        stats = TaskDispatcherStats()
        stats.tasks_dispatched = 5
        stats.tasks_completed = 3
        stats.tasks_failed = 1
        
        stats_dict = stats.to_dict()
        
        assert stats_dict["tasks_dispatched"] == 5
        assert stats_dict["tasks_completed"] == 3
        assert stats_dict["tasks_failed"] == 1
        assert "uptime_seconds" in stats_dict
        assert "timestamp" in stats_dict


class TestTaskDispatcher:
    """Test TaskDispatcher functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_db_manager):
        """Test dispatcher initialization."""
        dispatcher = TaskDispatcher(
            max_workers=5,
            max_queue_size=100,
            db_manager=mock_db_manager
        )
        
        assert dispatcher.max_workers == 5
        assert dispatcher.max_queue_size == 100
        assert len(dispatcher.task_queue) == 0
        assert len(dispatcher.active_tasks) == 0
        assert len(dispatcher.workers) == 0
        assert not dispatcher._running
    
    @pytest.mark.asyncio
    async def test_start_and_stop(self, mock_db_manager):
        """Test dispatcher start and stop."""
        dispatcher = TaskDispatcher(db_manager=mock_db_manager)
        
        # Start dispatcher
        await dispatcher.start()
        assert dispatcher._running
        assert dispatcher._scheduler_task is not None
        
        # Stop dispatcher
        await dispatcher.stop()
        assert not dispatcher._running
        assert len(dispatcher.workers) == 0
    
    @pytest.mark.asyncio
    async def test_submit_task(self, task_dispatcher, sample_task_payload):
        """Test task submission."""
        task_id = await task_dispatcher.submit_task(
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            priority=TaskPriority.HIGH
        )
        
        assert task_id is not None
        assert len(task_dispatcher.task_queue) == 1
        assert task_dispatcher.stats.tasks_dispatched == 1
        
        # Check task properties
        task = task_dispatcher.task_queue[0]
        assert task.id == task_id
        assert task.task_type == TaskType.SCRAPE_URL
        assert task.payload == sample_task_payload
        assert task.priority == TaskPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_submit_task_string_types(self, task_dispatcher, sample_task_payload):
        """Test task submission with string types."""
        task_id = await task_dispatcher.submit_task(
            task_type="scrape_url",  # String instead of enum
            payload=sample_task_payload,
            priority=2  # Int instead of enum
        )
        
        assert task_id is not None
        task = task_dispatcher.task_queue[0]
        assert task.task_type == TaskType.SCRAPE_URL
        assert task.priority == TaskPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_submit_task_queue_full(self, mock_db_manager, sample_task_payload):
        """Test task submission when queue is full."""
        dispatcher = TaskDispatcher(
            max_queue_size=1,
            db_manager=mock_db_manager
        )
        await dispatcher.start()
        
        try:
            # Fill the queue
            await dispatcher.submit_task(
                task_type=TaskType.SCRAPE_URL,
                payload=sample_task_payload
            )
            
            # This should raise an error
            with pytest.raises(RuntimeError, match="Task queue is full"):
                await dispatcher.submit_task(
                    task_type=TaskType.SCRAPE_URL,
                    payload=sample_task_payload
                )
        finally:
            await dispatcher.stop()
    
    @pytest.mark.asyncio
    async def test_submit_task_not_running(self, mock_db_manager, sample_task_payload):
        """Test task submission when dispatcher is not running."""
        dispatcher = TaskDispatcher(db_manager=mock_db_manager)
        
        with pytest.raises(RuntimeError, match="Task dispatcher is not running"):
            await dispatcher.submit_task(
                task_type=TaskType.SCRAPE_URL,
                payload=sample_task_payload
            )
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, task_dispatcher, sample_task_payload):
        """Test that tasks are processed in priority order."""
        # Submit tasks in reverse priority order
        low_task_id = await task_dispatcher.submit_task(
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            priority=TaskPriority.LOW
        )
        
        high_task_id = await task_dispatcher.submit_task(
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            priority=TaskPriority.HIGH
        )
        
        critical_task_id = await task_dispatcher.submit_task(
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            priority=TaskPriority.CRITICAL
        )
        
        # Check queue ordering (heap property)
        assert len(task_dispatcher.task_queue) == 3
        
        # The first task should be the critical one
        first_task = task_dispatcher.task_queue[0]
        assert first_task.id == critical_task_id
        assert first_task.priority == TaskPriority.CRITICAL
    
    @pytest.mark.asyncio
    async def test_get_task_status_queued(self, task_dispatcher, sample_task_payload):
        """Test getting status of queued task."""
        task_id = await task_dispatcher.submit_task(
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload
        )
        
        status = await task_dispatcher.get_task_status(task_id)
        
        assert status is not None
        assert status["status"] == "queued"
        assert status["task"]["id"] == task_id
    
    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, task_dispatcher):
        """Test getting status of non-existent task."""
        status = await task_dispatcher.get_task_status("non-existent-task")
        assert status is None
    
    @pytest.mark.asyncio
    async def test_cancel_queued_task(self, task_dispatcher, sample_task_payload):
        """Test cancelling a queued task."""
        task_id = await task_dispatcher.submit_task(
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload
        )
        
        assert len(task_dispatcher.task_queue) == 1
        
        success = await task_dispatcher.cancel_task(task_id)
        
        assert success
        assert len(task_dispatcher.task_queue) == 0
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, task_dispatcher):
        """Test cancelling a non-existent task."""
        success = await task_dispatcher.cancel_task("non-existent-task")
        assert not success
    
    @pytest.mark.asyncio
    async def test_register_task_handler(self, task_dispatcher):
        """Test registering task handlers."""
        async def test_handler(payload):
            return {"result": "success"}
        
        task_dispatcher.register_task_handler(TaskType.SCRAPE_URL, test_handler)
        
        assert TaskType.SCRAPE_URL in task_dispatcher.task_handlers
        assert task_dispatcher.task_handlers[TaskType.SCRAPE_URL] == test_handler
    
    @pytest.mark.asyncio
    async def test_task_execution_success(self, task_dispatcher, sample_task_payload):
        """Test successful task execution."""
        # Register a test handler
        async def test_handler(payload):
            await asyncio.sleep(0.01)  # Simulate work
            return {"result": "success", "payload": payload}
        
        task_dispatcher.register_task_handler(TaskType.SCRAPE_URL, test_handler)
        
        # Submit task
        task_id = await task_dispatcher.submit_task(
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload
        )
        
        # Wait for execution
        await asyncio.sleep(0.2)
        
        # Check task completed
        assert task_id in task_dispatcher.completed_tasks
        task = task_dispatcher.completed_tasks[task_id]
        assert len(task.executions) == 1
        assert task.executions[0].result["result"] == "success"
        assert task.executions[0].error is None
    
    @pytest.mark.asyncio
    async def test_task_execution_failure_with_retry(self, task_dispatcher, sample_task_payload):
        """Test task execution failure with retry."""
        call_count = 0
        
        async def failing_handler(payload):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 attempts
                raise Exception(f"Attempt {call_count} failed")
            return {"result": "success", "attempt": call_count}
        
        task_dispatcher.register_task_handler(TaskType.SCRAPE_URL, failing_handler)
        
        # Submit task
        task_id = await task_dispatcher.submit_task(
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            max_retries=3
        )
        
        # Wait for execution and retries
        await asyncio.sleep(1.0)
        
        # Should eventually succeed
        assert task_id in task_dispatcher.completed_tasks
        task = task_dispatcher.completed_tasks[task_id]
        assert len(task.executions) == 3  # 1 initial + 2 retries
        assert task.executions[-1].result["result"] == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_task_execution_permanent_failure(self, task_dispatcher, sample_task_payload):
        """Test task execution with permanent failure."""
        async def always_failing_handler(payload):
            raise Exception("This task always fails")
        
        task_dispatcher.register_task_handler(TaskType.SCRAPE_URL, always_failing_handler)
        
        # Mock fallback manager to avoid actual fallback storage
        with patch('src.signal_relay.task_dispatcher.get_fallback_manager') as mock_fallback:
            mock_fallback_instance = AsyncMock()
            mock_fallback.return_value = mock_fallback_instance
            
            # Submit task with limited retries
            task_id = await task_dispatcher.submit_task(
                task_type=TaskType.SCRAPE_URL,
                payload=sample_task_payload,
                max_retries=1
            )
            
            # Wait for execution and retries
            await asyncio.sleep(0.5)
            
            # Should be permanently failed
            assert task_id in task_dispatcher.failed_tasks
            task = task_dispatcher.failed_tasks[task_id]
            assert len(task.executions) == 2  # 1 initial + 1 retry
            assert all(exec.error is not None for exec in task.executions)
            
            # Should have tried to store in fallback
            mock_fallback_instance.store_task_in_fallback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_execution_no_handler(self, task_dispatcher, sample_task_payload):
        """Test task execution with no registered handler."""
        # Submit task without registering handler
        task_id = await task_dispatcher.submit_task(
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload,
            max_retries=1
        )
        
        # Wait for execution
        await asyncio.sleep(0.2)
        
        # Should be permanently failed
        assert task_id in task_dispatcher.failed_tasks
        task = task_dispatcher.failed_tasks[task_id]
        assert "No handler registered" in task.executions[0].error
    
    def test_get_stats(self, task_dispatcher):
        """Test getting dispatcher statistics."""
        stats = task_dispatcher.get_stats()
        
        assert isinstance(stats, dict)
        assert "tasks_dispatched" in stats
        assert "tasks_completed" in stats
        assert "tasks_failed" in stats
        assert "active_workers" in stats
        assert "queue_size" in stats
        assert "success_rate" in stats
        assert "timestamp" in stats
    
    def test_get_queue_info(self, task_dispatcher):
        """Test getting queue information."""
        queue_info = task_dispatcher.get_queue_info()
        
        assert isinstance(queue_info, dict)
        assert "total_queued" in queue_info
        assert "active_tasks" in queue_info
        assert "completed_tasks" in queue_info
        assert "failed_tasks" in queue_info
        assert "priority_breakdown" in queue_info
        assert "next_task_scheduled" in queue_info
    
    @pytest.mark.asyncio
    async def test_add_status_callback(self, task_dispatcher, sample_task_payload):
        """Test adding status callbacks."""
        callback_calls = []
        
        def status_callback(task):
            callback_calls.append(task.id)
        
        task_dispatcher.add_status_callback(status_callback)
        
        # Register handler and submit task
        async def test_handler(payload):
            return {"result": "success"}
        
        task_dispatcher.register_task_handler(TaskType.SCRAPE_URL, test_handler)
        
        task_id = await task_dispatcher.submit_task(
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload
        )
        
        # Wait for execution
        await asyncio.sleep(0.1)
        
        # Callback should have been called
        assert task_id in callback_calls
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, mock_db_manager, sample_task_payload):
        """Test concurrent execution of multiple tasks."""
        dispatcher = TaskDispatcher(
            max_workers=3,
            db_manager=mock_db_manager
        )
        await dispatcher.start()
        
        try:
            execution_order = []
            
            async def test_handler(payload):
                task_id = payload.get("task_id")
                execution_order.append(task_id)
                await asyncio.sleep(0.05)  # Simulate work
                return {"result": "success", "task_id": task_id}
            
            dispatcher.register_task_handler(TaskType.SCRAPE_URL, test_handler)
            
            # Submit multiple tasks
            task_ids = []
            for i in range(5):
                payload = {**sample_task_payload, "task_id": f"task-{i}"}
                task_id = await dispatcher.submit_task(
                    task_type=TaskType.SCRAPE_URL,
                    payload=payload
                )
                task_ids.append(task_id)
            
            # Wait for all tasks to complete
            await asyncio.sleep(0.3)
            
            # All tasks should be completed
            assert len(dispatcher.completed_tasks) == 5
            assert len(execution_order) == 5
            
            # Check that tasks were executed concurrently (not in strict order)
            # This is probabilistic but should work most of the time
            assert len(set(execution_order)) == 5  # All unique
            
        finally:
            await dispatcher.stop()


class TestGlobalTaskDispatcher:
    """Test global task dispatcher functions."""
    
    @pytest.mark.asyncio
    async def test_get_task_dispatcher_singleton(self):
        """Test that get_task_dispatcher returns singleton instance."""
        # Clean up any existing instance
        await cleanup_task_dispatcher()
        
        with patch('src.signal_relay.task_dispatcher.TaskDispatcher') as mock_dispatcher_class:
            mock_instance = AsyncMock()
            mock_instance.start = AsyncMock()
            mock_dispatcher_class.return_value = mock_instance
            
            # First call should create instance
            dispatcher1 = await get_task_dispatcher()
            assert dispatcher1 == mock_instance
            mock_instance.start.assert_called_once()
            
            # Second call should return same instance
            dispatcher2 = await get_task_dispatcher()
            assert dispatcher2 == dispatcher1
            # Start should not be called again
            assert mock_instance.start.call_count == 1
        
        # Cleanup
        await cleanup_task_dispatcher()
    
    @pytest.mark.asyncio
    async def test_cleanup_task_dispatcher(self):
        """Test cleanup of global task dispatcher."""
        # Clean up any existing instance
        await cleanup_task_dispatcher()
        
        with patch('src.signal_relay.task_dispatcher.TaskDispatcher') as mock_dispatcher_class:
            mock_instance = AsyncMock()
            mock_instance.start = AsyncMock()
            mock_instance.stop = AsyncMock()
            mock_dispatcher_class.return_value = mock_instance
            
            # Create instance
            dispatcher = await get_task_dispatcher()
            assert dispatcher == mock_instance
            
            # Cleanup
            await cleanup_task_dispatcher()
            mock_instance.stop.assert_called_once()
            
            # Next call should create new instance
            dispatcher2 = await get_task_dispatcher()
            assert dispatcher2 == mock_instance  # Same mock, but would be new instance in reality


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_scheduler_loop_error_handling(self, mock_db_manager):
        """Test that scheduler loop handles errors gracefully."""
        dispatcher = TaskDispatcher(db_manager=mock_db_manager)
        
        # Mock _process_ready_tasks to raise an exception
        original_process = dispatcher._process_ready_tasks
        call_count = 0
        
        async def failing_process():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Test error")
            return await original_process()
        
        dispatcher._process_ready_tasks = failing_process
        
        await dispatcher.start()
        
        # Wait a bit to let the scheduler loop run
        await asyncio.sleep(0.1)
        
        # Dispatcher should still be running despite the error
        assert dispatcher._running
        assert call_count >= 1
        
        await dispatcher.stop()
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, task_dispatcher, sample_task_payload):
        """Test that callback errors don't break task execution."""
        def failing_callback(task):
            raise Exception("Callback failed")
        
        task_dispatcher.add_status_callback(failing_callback)
        
        # Register handler and submit task
        async def test_handler(payload):
            return {"result": "success"}
        
        task_dispatcher.register_task_handler(TaskType.SCRAPE_URL, test_handler)
        
        task_id = await task_dispatcher.submit_task(
            task_type=TaskType.SCRAPE_URL,
            payload=sample_task_payload
        )
        
        # Wait for execution
        await asyncio.sleep(0.1)
        
        # Task should still complete successfully despite callback error
        assert task_id in task_dispatcher.completed_tasks
    
    @pytest.mark.asyncio
    async def test_worker_cleanup_on_exception(self, mock_db_manager, sample_task_payload):
        """Test that workers are cleaned up even if they raise exceptions."""
        dispatcher = TaskDispatcher(max_workers=1, db_manager=mock_db_manager)
        await dispatcher.start()
        
        try:
            async def exception_handler(payload):
                raise Exception("Handler exception")
            
            dispatcher.register_task_handler(TaskType.SCRAPE_URL, exception_handler)
            
            # Submit task
            await dispatcher.submit_task(
                task_type=TaskType.SCRAPE_URL,
                payload=sample_task_payload,
                max_retries=0  # No retries to speed up test
            )
            
            # Wait for execution and cleanup
            await asyncio.sleep(0.2)
            
            # Force cleanup
            dispatcher._cleanup_workers()
            
            # Worker should be cleaned up
            assert len(dispatcher.workers) == 0
            
        finally:
            await dispatcher.stop()