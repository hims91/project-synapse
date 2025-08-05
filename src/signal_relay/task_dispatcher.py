"""
Signal Relay - Task Dispatcher
Layer 2: Signal Network

This module implements the async task dispatcher with priority-based processing.
Handles task distribution, retry logic, and progress monitoring across the system.
"""
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable, Set, Union
from enum import Enum
import heapq
import structlog
from dataclasses import dataclass, field

from ..shared.config import get_settings
from ..synaptic_vesicle.database import DatabaseManager, db_manager
from ..synaptic_vesicle.repositories import RepositoryFactory, TaskQueueRepository
from ..synaptic_vesicle.models import TaskQueue
from ..shared.schemas import TaskQueueCreate, TaskStatus
from ..spinal_cord.fallback_manager import get_fallback_manager

logger = structlog.get_logger(__name__)


class TaskPriority(int, Enum):
    """Task priority levels (lower number = higher priority)."""
    CRITICAL = 1    # System-critical tasks (health checks, recovery)
    HIGH = 2        # User-facing tasks (API requests, real-time scraping)
    NORMAL = 3      # Regular processing (feed polling, batch scraping)
    LOW = 4         # Background tasks (cleanup, analytics)
    BULK = 5        # Bulk operations (data migration, batch processing)


class TaskType(str, Enum):
    """Types of tasks that can be dispatched."""
    SCRAPE_URL = "scrape_url"
    PROCESS_FEED = "process_feed"
    ANALYZE_CONTENT = "analyze_content"
    GENERATE_SUMMARY = "generate_summary"
    EXTRACT_ENTITIES = "extract_entities"
    MONITOR_KEYWORD = "monitor_keyword"
    SEND_WEBHOOK = "send_webhook"
    CLEANUP_DATA = "cleanup_data"
    HEALTH_CHECK = "health_check"
    SYSTEM_MAINTENANCE = "system_maintenance"


@dataclass
class TaskExecution:
    """Represents a task execution attempt."""
    task_id: str
    attempt: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None
    
    def mark_completed(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark the execution as completed."""
        self.completed_at = datetime.now(timezone.utc)
        self.result = result
        if self.started_at:
            self.duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
    
    def mark_failed(self, error: str) -> None:
        """Mark the execution as failed."""
        self.completed_at = datetime.now(timezone.utc)
        self.error = error
        if self.started_at:
            self.duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)


@dataclass
class DispatchableTask:
    """A task ready for dispatch with priority and retry information."""
    id: str
    task_type: TaskType
    payload: Dict[str, Any]
    priority: TaskPriority
    max_retries: int = 3
    retry_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_attempt_at: Optional[datetime] = None
    executions: List[TaskExecution] = field(default_factory=list)
    
    def __lt__(self, other: 'DispatchableTask') -> bool:
        """Priority queue comparison (lower priority number = higher priority)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        # If same priority, earlier scheduled time has higher priority
        return self.scheduled_at < other.scheduled_at
    
    def can_retry(self) -> bool:
        """Check if the task can be retried."""
        return self.retry_count < self.max_retries
    
    def calculate_next_retry_delay(self, base_delay: int = 1) -> int:
        """Calculate exponential backoff delay for next retry."""
        return min(base_delay * (2 ** self.retry_count), 300)  # Max 5 minutes
    
    def schedule_retry(self, base_delay: int = 1) -> None:
        """Schedule the task for retry with exponential backoff."""
        if self.can_retry():
            delay_seconds = self.calculate_next_retry_delay(base_delay)
            self.scheduled_at = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
            self.retry_count += 1
            logger.info("Task scheduled for retry",
                       task_id=self.id,
                       retry_count=self.retry_count,
                       delay_seconds=delay_seconds)
    
    def start_execution(self) -> TaskExecution:
        """Start a new execution attempt."""
        execution = TaskExecution(
            task_id=self.id,
            attempt=len(self.executions) + 1,
            started_at=datetime.now(timezone.utc)
        )
        self.executions.append(execution)
        self.last_attempt_at = execution.started_at
        return execution
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "payload": self.payload,
            "priority": self.priority,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat(),
            "last_attempt_at": self.last_attempt_at.isoformat() if self.last_attempt_at else None,
            "executions": [
                {
                    "attempt": exec.attempt,
                    "started_at": exec.started_at.isoformat(),
                    "completed_at": exec.completed_at.isoformat() if exec.completed_at else None,
                    "duration_ms": exec.duration_ms,
                    "error": exec.error,
                    "result": exec.result
                }
                for exec in self.executions
            ]
        }


class TaskDispatcherStats:
    """Statistics for task dispatcher operations."""
    
    def __init__(self):
        self.tasks_dispatched = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.tasks_retried = 0
        self.total_execution_time_ms = 0
        self.active_workers = 0
        self.queue_size = 0
        self.average_execution_time_ms = 0.0
        self.success_rate = 0.0
        self.started_at = datetime.now(timezone.utc)
    
    def update_completion(self, execution: TaskExecution, success: bool) -> None:
        """Update stats when a task completes."""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        
        if execution.duration_ms:
            self.total_execution_time_ms += execution.duration_ms
            completed_tasks = self.tasks_completed + self.tasks_failed
            if completed_tasks > 0:
                self.average_execution_time_ms = self.total_execution_time_ms / completed_tasks
        
        # Calculate success rate
        total_attempts = self.tasks_completed + self.tasks_failed
        if total_attempts > 0:
            self.success_rate = self.tasks_completed / total_attempts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for serialization."""
        uptime = datetime.now(timezone.utc) - self.started_at
        return {
            "tasks_dispatched": self.tasks_dispatched,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "tasks_retried": self.tasks_retried,
            "active_workers": self.active_workers,
            "queue_size": self.queue_size,
            "average_execution_time_ms": round(self.average_execution_time_ms, 2),
            "success_rate": round(self.success_rate, 4),
            "uptime_seconds": uptime.total_seconds(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class TaskDispatcher:
    """
    Async task dispatcher with priority-based processing and retry logic.
    
    Features:
    - Priority queue for task ordering
    - Exponential backoff retry logic
    - Concurrent worker management
    - Task progress monitoring
    - Integration with fallback system
    - Comprehensive statistics tracking
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        max_queue_size: int = 1000,
        base_retry_delay: int = 1,
        db_manager: Optional[DatabaseManager] = None,
        task_handlers: Optional[Dict[TaskType, Callable]] = None
    ):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.base_retry_delay = base_retry_delay
        self.db_manager = db_manager or db_manager
        self.task_handlers = task_handlers or {}
        
        # Task management
        self.task_queue: List[DispatchableTask] = []
        self.active_tasks: Dict[str, DispatchableTask] = {}
        self.completed_tasks: Dict[str, DispatchableTask] = {}
        self.failed_tasks: Dict[str, DispatchableTask] = {}
        
        # Worker management
        self.workers: Set[asyncio.Task] = set()
        self.worker_semaphore = asyncio.Semaphore(max_workers)
        self.shutdown_event = asyncio.Event()
        
        # Statistics and monitoring
        self.stats = TaskDispatcherStats()
        self.task_status_callbacks: List[Callable] = []
        
        # Internal state
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        logger.info("Task dispatcher initialized",
                   max_workers=max_workers,
                   max_queue_size=max_queue_size)
    
    async def start(self) -> None:
        """Start the task dispatcher."""
        if self._running:
            logger.warning("Task dispatcher already running")
            return
        
        try:
            self._running = True
            self.shutdown_event.clear()
            
            # Start the scheduler
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            
            logger.info("Task dispatcher started",
                       max_workers=self.max_workers,
                       max_queue_size=self.max_queue_size)
            
        except Exception as e:
            logger.error("Failed to start task dispatcher", error=str(e))
            self._running = False
            raise
    
    async def stop(self, timeout: int = 30) -> None:
        """Stop the task dispatcher gracefully."""
        if not self._running:
            return
        
        try:
            logger.info("Stopping task dispatcher...")
            
            # Signal shutdown
            self.shutdown_event.set()
            self._running = False
            
            # Stop scheduler
            if self._scheduler_task:
                self._scheduler_task.cancel()
                try:
                    await asyncio.wait_for(self._scheduler_task, timeout=5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Wait for active workers to complete
            if self.workers:
                logger.info("Waiting for active workers to complete",
                           active_workers=len(self.workers))
                
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.workers, return_exceptions=True),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for workers, cancelling remaining tasks")
                    for worker in self.workers:
                        worker.cancel()
            
            self.workers.clear()
            
            logger.info("Task dispatcher stopped",
                       completed_tasks=self.stats.tasks_completed,
                       failed_tasks=self.stats.tasks_failed)
            
        except Exception as e:
            logger.error("Error stopping task dispatcher", error=str(e))
    
    async def submit_task(
        self,
        task_type: Union[TaskType, str],
        payload: Dict[str, Any],
        priority: Union[TaskPriority, int] = TaskPriority.NORMAL,
        max_retries: int = 3,
        scheduled_at: Optional[datetime] = None
    ) -> str:
        """
        Submit a task for execution.
        
        Args:
            task_type: Type of task to execute
            payload: Task payload data
            priority: Task priority level
            max_retries: Maximum retry attempts
            scheduled_at: When to execute the task (default: now)
            
        Returns:
            Task ID
        """
        if not self._running:
            raise RuntimeError("Task dispatcher is not running")
        
        if len(self.task_queue) >= self.max_queue_size:
            raise RuntimeError(f"Task queue is full (max: {self.max_queue_size})")
        
        # Convert string types to enums
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        if isinstance(priority, int):
            priority = TaskPriority(priority)
        
        # Create task
        task = DispatchableTask(
            id=str(uuid.uuid4()),
            task_type=task_type,
            payload=payload,
            priority=priority,
            max_retries=max_retries,
            scheduled_at=scheduled_at or datetime.now(timezone.utc)
        )
        
        # Add to priority queue
        heapq.heappush(self.task_queue, task)
        self.stats.tasks_dispatched += 1
        self.stats.queue_size = len(self.task_queue)
        
        logger.info("Task submitted",
                   task_id=task.id,
                   task_type=task_type,
                   priority=priority,
                   queue_size=len(self.task_queue))
        
        return task.id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "status": "running",
                "task": task.to_dict()
            }
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                "status": "completed",
                "task": task.to_dict()
            }
        
        # Check failed tasks
        if task_id in self.failed_tasks:
            task = self.failed_tasks[task_id]
            return {
                "status": "failed",
                "task": task.to_dict()
            }
        
        # Check queue
        for task in self.task_queue:
            if task.id == task_id:
                return {
                    "status": "queued",
                    "task": task.to_dict()
                }
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued or active task."""
        # Remove from queue
        for i, task in enumerate(self.task_queue):
            if task.id == task_id:
                del self.task_queue[i]
                heapq.heapify(self.task_queue)  # Restore heap property
                self.stats.queue_size = len(self.task_queue)
                logger.info("Task cancelled from queue", task_id=task_id)
                return True
        
        # Cancel active task (this is more complex and depends on task implementation)
        if task_id in self.active_tasks:
            logger.warning("Cannot cancel active task", task_id=task_id)
            return False
        
        return False
    
    def register_task_handler(self, task_type: TaskType, handler: Callable) -> None:
        """Register a handler for a specific task type."""
        self.task_handlers[task_type] = handler
        logger.info("Task handler registered", task_type=task_type)
    
    def add_status_callback(self, callback: Callable) -> None:
        """Add a callback for task status updates."""
        self.task_status_callbacks.append(callback)
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop that processes the task queue."""
        while not self.shutdown_event.is_set():
            try:
                # Process ready tasks
                await self._process_ready_tasks()
                
                # Update stats
                self.stats.queue_size = len(self.task_queue)
                self.stats.active_workers = len(self.workers)
                
                # Clean up completed workers
                self._cleanup_workers()
                
                # Short sleep to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("Error in scheduler loop", error=str(e))
                await asyncio.sleep(1)  # Longer sleep on error
    
    async def _process_ready_tasks(self) -> None:
        """Process tasks that are ready for execution."""
        now = datetime.now(timezone.utc)
        
        while (self.task_queue and 
               len(self.workers) < self.max_workers and
               self.task_queue[0].scheduled_at <= now):
            
            # Get highest priority task
            task = heapq.heappop(self.task_queue)
            
            # Move to active tasks
            self.active_tasks[task.id] = task
            
            # Start worker
            worker = asyncio.create_task(self._execute_task(task))
            self.workers.add(worker)
            
            logger.debug("Task dispatched to worker",
                        task_id=task.id,
                        task_type=task.task_type,
                        active_workers=len(self.workers))
    
    async def _execute_task(self, task: DispatchableTask) -> None:
        """Execute a single task with error handling and retry logic."""
        async with self.worker_semaphore:
            execution = task.start_execution()
            
            try:
                # Get task handler
                handler = self.task_handlers.get(task.task_type)
                if not handler:
                    raise ValueError(f"No handler registered for task type: {task.task_type}")
                
                # Execute task
                logger.info("Executing task",
                           task_id=task.id,
                           task_type=task.task_type,
                           attempt=execution.attempt)
                
                result = await handler(task.payload)
                
                # Mark as completed
                execution.mark_completed(result)
                self._handle_task_success(task)
                
                logger.info("Task completed successfully",
                           task_id=task.id,
                           duration_ms=execution.duration_ms)
                
            except Exception as e:
                # Mark as failed
                execution.mark_failed(str(e))
                
                logger.error("Task execution failed",
                            task_id=task.id,
                            error=str(e),
                            attempt=execution.attempt)
                
                # Handle retry or failure
                await self._handle_task_failure(task, str(e))
            
            finally:
                # Remove from active tasks
                self.active_tasks.pop(task.id, None)
                
                # Notify callbacks
                await self._notify_status_callbacks(task)
    
    def _handle_task_success(self, task: DispatchableTask) -> None:
        """Handle successful task completion."""
        self.completed_tasks[task.id] = task
        self.stats.update_completion(task.executions[-1], success=True)
        
        logger.debug("Task marked as completed", task_id=task.id)
    
    async def _handle_task_failure(self, task: DispatchableTask, error: str) -> None:
        """Handle task failure with retry logic."""
        if task.can_retry():
            # Schedule retry
            task.schedule_retry(self.base_retry_delay)
            heapq.heappush(self.task_queue, task)
            self.stats.tasks_retried += 1
            
            logger.info("Task scheduled for retry",
                       task_id=task.id,
                       retry_count=task.retry_count,
                       next_attempt=task.scheduled_at.isoformat())
        else:
            # Mark as permanently failed
            self.failed_tasks[task.id] = task
            self.stats.update_completion(task.executions[-1], success=False)
            
            logger.error("Task permanently failed",
                        task_id=task.id,
                        max_retries=task.max_retries,
                        error=error)
            
            # Try to store in fallback system
            try:
                fallback_manager = await get_fallback_manager()
                await fallback_manager.store_task_in_fallback({
                    "id": task.id,
                    "task_type": task.task_type,
                    "payload": task.payload,
                    "priority": task.priority,
                    "error": error,
                    "failed_at": datetime.now(timezone.utc).isoformat()
                })
            except Exception as fallback_error:
                logger.error("Failed to store task in fallback",
                            task_id=task.id,
                            error=str(fallback_error))
    
    def _cleanup_workers(self) -> None:
        """Clean up completed worker tasks."""
        completed_workers = {worker for worker in self.workers if worker.done()}
        self.workers -= completed_workers
        
        # Log any worker exceptions
        for worker in completed_workers:
            try:
                worker.result()  # This will raise if the worker had an exception
            except Exception as e:
                logger.error("Worker task failed", error=str(e))
    
    async def _notify_status_callbacks(self, task: DispatchableTask) -> None:
        """Notify registered callbacks about task status changes."""
        for callback in self.task_status_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task)
                else:
                    callback(task)
            except Exception as e:
                logger.error("Error in status callback", error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current dispatcher statistics."""
        return self.stats.to_dict()
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Get information about the current queue state."""
        priority_counts = {}
        for task in self.task_queue:
            priority_counts[task.priority.name] = priority_counts.get(task.priority.name, 0) + 1
        
        return {
            "total_queued": len(self.task_queue),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "priority_breakdown": priority_counts,
            "next_task_scheduled": (
                self.task_queue[0].scheduled_at.isoformat() 
                if self.task_queue else None
            )
        }


# Global instance for dependency injection
_task_dispatcher: Optional[TaskDispatcher] = None


async def get_task_dispatcher() -> TaskDispatcher:
    """
    Get TaskDispatcher instance for dependency injection.
    
    Usage in FastAPI:
        @app.post("/tasks")
        async def create_task(
            task_data: dict,
            dispatcher: TaskDispatcher = Depends(get_task_dispatcher)
        ):
            task_id = await dispatcher.submit_task(
                task_type=TaskType.SCRAPE_URL,
                payload=task_data
            )
            return {"task_id": task_id}
    """
    global _task_dispatcher
    if _task_dispatcher is None:
        _task_dispatcher = TaskDispatcher()
        await _task_dispatcher.start()
    return _task_dispatcher


async def cleanup_task_dispatcher():
    """Cleanup the global task dispatcher instance."""
    global _task_dispatcher
    if _task_dispatcher:
        await _task_dispatcher.stop()
        _task_dispatcher = None