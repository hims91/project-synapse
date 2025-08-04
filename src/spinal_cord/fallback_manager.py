"""
Spinal Cord - Task Queue Fallback Manager
Layer 2: Signal Network

This module implements the automatic task serialization and recovery system.
Handles fallback to R2 storage during database outages and recovery when connectivity is restored.
"""
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Callable, Set
from enum import Enum
import structlog

from .r2_client import SpinalCordStorage, FallbackTask, R2StorageError
from ..synaptic_vesicle.database import DatabaseManager, db_manager
from ..synaptic_vesicle.repositories import TaskQueueRepository, RepositoryFactory
from ..synaptic_vesicle.models import TaskQueue
from ..shared.schemas import TaskQueueCreate, TaskStatus
from ..shared.config import get_settings

logger = structlog.get_logger(__name__)


class FallbackMode(str, Enum):
    """Fallback system operational modes."""
    NORMAL = "normal"           # Database available, normal operation
    FALLBACK = "fallback"       # Database unavailable, using R2 storage
    RECOVERY = "recovery"       # Recovering from fallback, re-injecting tasks
    MAINTENANCE = "maintenance" # Manual maintenance mode


class FallbackStats:
    """Statistics for fallback system operations."""
    
    def __init__(self):
        self.tasks_stored_in_fallback = 0
        self.tasks_recovered_from_fallback = 0
        self.fallback_activations = 0
        self.recovery_operations = 0
        self.last_fallback_activation = None
        self.last_recovery_completion = None
        self.current_mode = FallbackMode.NORMAL
        self.database_outage_duration = timedelta(0)
        self._outage_start_time = None
    
    def start_outage(self):
        """Mark the start of a database outage."""
        self._outage_start_time = datetime.now(timezone.utc)
        self.fallback_activations += 1
        self.last_fallback_activation = self._outage_start_time
        self.current_mode = FallbackMode.FALLBACK
    
    def end_outage(self):
        """Mark the end of a database outage."""
        if self._outage_start_time:
            outage_duration = datetime.now(timezone.utc) - self._outage_start_time
            self.database_outage_duration += outage_duration
            self._outage_start_time = None
        
        self.recovery_operations += 1
        self.last_recovery_completion = datetime.now(timezone.utc)
        self.current_mode = FallbackMode.NORMAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for serialization."""
        return {
            "tasks_stored_in_fallback": self.tasks_stored_in_fallback,
            "tasks_recovered_from_fallback": self.tasks_recovered_from_fallback,
            "fallback_activations": self.fallback_activations,
            "recovery_operations": self.recovery_operations,
            "last_fallback_activation": self.last_fallback_activation.isoformat() if self.last_fallback_activation else None,
            "last_recovery_completion": self.last_recovery_completion.isoformat() if self.last_recovery_completion else None,
            "current_mode": self.current_mode,
            "total_outage_duration_seconds": self.database_outage_duration.total_seconds(),
            "current_outage_duration_seconds": (
                (datetime.now(timezone.utc) - self._outage_start_time).total_seconds()
                if self._outage_start_time else 0
            )
        }


class TaskQueueFallbackManager:
    """
    Manages automatic fallback to R2 storage during database outages.
    
    Features:
    - Automatic detection of database connectivity issues
    - Seamless fallback to R2 storage for task queuing
    - Automatic recovery and task re-injection when database is restored
    - Monitoring and alerting for fallback system status
    - Batch processing for efficient recovery operations
    """
    
    def __init__(
        self,
        spinal_cord: Optional[SpinalCordStorage] = None,
        db_manager: Optional[DatabaseManager] = None,
        health_check_interval: int = 30,
        recovery_batch_size: int = 50,
        max_recovery_retries: int = 3
    ):
        self.spinal_cord = spinal_cord or SpinalCordStorage()
        self.db_manager = db_manager or db_manager
        self.health_check_interval = health_check_interval
        self.recovery_batch_size = recovery_batch_size
        self.max_recovery_retries = max_recovery_retries
        
        # State management
        self.current_mode = FallbackMode.NORMAL
        self.stats = FallbackStats()
        self._health_check_task: Optional[asyncio.Task] = None
        self._recovery_in_progress = False
        self._pending_tasks: Set[str] = set()  # Track tasks being processed
        
        # Callbacks for external notification
        self.on_fallback_activated: Optional[Callable] = None
        self.on_recovery_completed: Optional[Callable] = None
        self.on_task_stored: Optional[Callable] = None
        self.on_task_recovered: Optional[Callable] = None
    
    async def start(self) -> None:
        """Start the fallback manager and health monitoring."""
        try:
            # Initialize connections
            await self.spinal_cord.__aenter__()
            
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitor_loop())
            
            logger.info("Task queue fallback manager started", 
                       health_check_interval=self.health_check_interval)
            
        except Exception as e:
            logger.error("Failed to start fallback manager", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop the fallback manager and cleanup resources."""
        try:
            # Stop health monitoring
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup connections
            await self.spinal_cord.__aexit__(None, None, None)
            
            logger.info("Task queue fallback manager stopped")
            
        except Exception as e:
            logger.error("Error stopping fallback manager", error=str(e))
    
    async def _health_monitor_loop(self) -> None:
        """Continuous health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_database_health()
                
            except asyncio.CancelledError:
                logger.info("Health monitor loop cancelled")
                break
            except Exception as e:
                logger.error("Error in health monitor loop", error=str(e))
                # Continue monitoring despite errors
                await asyncio.sleep(5)
    
    async def _check_database_health(self) -> None:
        """Check database health and manage fallback state."""
        try:
            is_healthy = await self.db_manager.health_check()
            
            if is_healthy and self.current_mode == FallbackMode.FALLBACK:
                # Database recovered, start recovery process
                await self._initiate_recovery()
                
            elif not is_healthy and self.current_mode == FallbackMode.NORMAL:
                # Database failed, activate fallback
                await self._activate_fallback()
                
        except Exception as e:
            logger.error("Error checking database health", error=str(e))
            # Assume database is unhealthy on error
            if self.current_mode == FallbackMode.NORMAL:
                await self._activate_fallback()
    
    async def _activate_fallback(self) -> None:
        """Activate fallback mode."""
        try:
            self.current_mode = FallbackMode.FALLBACK
            self.stats.start_outage()
            
            logger.warning("Database outage detected, activating fallback mode",
                          mode=self.current_mode,
                          fallback_activations=self.stats.fallback_activations)
            
            # Notify external systems
            if self.on_fallback_activated:
                try:
                    await self.on_fallback_activated(self.stats.to_dict())
                except Exception as e:
                    logger.error("Error in fallback activation callback", error=str(e))
                    
        except Exception as e:
            logger.error("Error activating fallback mode", error=str(e))
    
    async def _initiate_recovery(self) -> None:
        """Initiate recovery from fallback mode."""
        if self._recovery_in_progress:
            logger.info("Recovery already in progress, skipping")
            return
        
        try:
            self._recovery_in_progress = True
            self.current_mode = FallbackMode.RECOVERY
            
            logger.info("Database connectivity restored, initiating recovery",
                       mode=self.current_mode)
            
            # Perform recovery
            recovered_count = await self._recover_tasks_from_fallback()
            
            # Update stats and mode
            self.current_mode = FallbackMode.NORMAL
            self.stats.end_outage()
            
            logger.info("Recovery completed successfully",
                       recovered_tasks=recovered_count,
                       mode=self.current_mode,
                       total_outage_duration=self.stats.database_outage_duration.total_seconds())
            
            # Notify external systems
            if self.on_recovery_completed:
                try:
                    await self.on_recovery_completed({
                        "recovered_tasks": recovered_count,
                        "stats": self.stats.to_dict()
                    })
                except Exception as e:
                    logger.error("Error in recovery completion callback", error=str(e))
                    
        except Exception as e:
            logger.error("Error during recovery process", error=str(e))
            # Reset recovery state but keep fallback mode
            self.current_mode = FallbackMode.FALLBACK
        finally:
            self._recovery_in_progress = False
    
    async def store_task_in_fallback(self, task_data: Dict[str, Any]) -> str:
        """
        Store a task in fallback storage.
        
        Args:
            task_data: Task data dictionary
            
        Returns:
            Task ID of stored task
        """
        try:
            # Create FallbackTask from data
            fallback_task = FallbackTask(
                id=task_data.get('id', str(uuid.uuid4())),
                task_type=task_data.get('task_type', 'unknown'),
                payload=task_data.get('payload', {}),
                priority=task_data.get('priority', 5),
                created_at=datetime.now(timezone.utc),
                retry_count=task_data.get('retry_count', 0),
                max_retries=task_data.get('max_retries', 3)
            )
            
            # Store in R2
            await self.spinal_cord.store_task(fallback_task)
            
            # Update stats
            self.stats.tasks_stored_in_fallback += 1
            self._pending_tasks.add(fallback_task.id)
            
            logger.info("Task stored in fallback storage",
                       task_id=fallback_task.id,
                       task_type=fallback_task.task_type,
                       total_stored=self.stats.tasks_stored_in_fallback)
            
            # Notify external systems
            if self.on_task_stored:
                try:
                    await self.on_task_stored(fallback_task.to_dict())
                except Exception as e:
                    logger.error("Error in task stored callback", error=str(e))
            
            return fallback_task.id
            
        except Exception as e:
            logger.error("Error storing task in fallback", task_data=task_data, error=str(e))
            raise
    
    async def _recover_tasks_from_fallback(self) -> int:
        """
        Recover all tasks from fallback storage and inject them into the database.
        
        Returns:
            Number of tasks recovered
        """
        try:
            # Get list of stored tasks
            task_ids = await self.spinal_cord.list_stored_tasks()
            
            if not task_ids:
                logger.info("No tasks found in fallback storage")
                return 0
            
            logger.info("Starting task recovery from fallback storage",
                       total_tasks=len(task_ids))
            
            recovered_count = 0
            failed_count = 0
            
            # Process tasks in batches
            for i in range(0, len(task_ids), self.recovery_batch_size):
                batch = task_ids[i:i + self.recovery_batch_size]
                batch_recovered = await self._recover_task_batch(batch)
                recovered_count += batch_recovered
                failed_count += len(batch) - batch_recovered
                
                # Small delay between batches to avoid overwhelming the database
                if i + self.recovery_batch_size < len(task_ids):
                    await asyncio.sleep(0.1)
            
            # Update stats
            self.stats.tasks_recovered_from_fallback += recovered_count
            
            logger.info("Task recovery completed",
                       recovered=recovered_count,
                       failed=failed_count,
                       total_processed=len(task_ids))
            
            return recovered_count
            
        except Exception as e:
            logger.error("Error recovering tasks from fallback", error=str(e))
            raise
    
    async def _recover_task_batch(self, task_ids: List[str]) -> int:
        """
        Recover a batch of tasks from fallback storage.
        
        Args:
            task_ids: List of task IDs to recover
            
        Returns:
            Number of tasks successfully recovered
        """
        recovered_count = 0
        
        try:
            async with self.db_manager.get_session() as session:
                repo_factory = RepositoryFactory(session)
                task_repo = repo_factory.task_queue
                
                for task_id in task_ids:
                    try:
                        # Retrieve task from fallback storage
                        fallback_task = await self.spinal_cord.retrieve_task(task_id)
                        
                        if not fallback_task:
                            logger.warning("Task not found in fallback storage", task_id=task_id)
                            continue
                        
                        # Convert to database task
                        task_create = TaskQueueCreate(
                            task_type=fallback_task.task_type,
                            payload=fallback_task.payload,
                            priority=fallback_task.priority,
                            max_retries=fallback_task.max_retries,
                            scheduled_at=fallback_task.created_at
                        )
                        
                        # Insert into database
                        db_task = await task_repo.create(task_create)
                        
                        # Delete from fallback storage
                        await self.spinal_cord.delete_task(task_id)
                        
                        # Remove from pending set
                        self._pending_tasks.discard(task_id)
                        
                        recovered_count += 1
                        
                        logger.debug("Task recovered from fallback",
                                   task_id=task_id,
                                   db_task_id=db_task.id,
                                   task_type=fallback_task.task_type)
                        
                        # Notify external systems
                        if self.on_task_recovered:
                            try:
                                await self.on_task_recovered({
                                    "fallback_task_id": task_id,
                                    "db_task_id": str(db_task.id),
                                    "task_type": fallback_task.task_type
                                })
                            except Exception as e:
                                logger.error("Error in task recovered callback", error=str(e))
                        
                    except Exception as e:
                        logger.error("Error recovering individual task",
                                   task_id=task_id,
                                   error=str(e))
                        # Continue with other tasks
                        continue
                        
        except Exception as e:
            logger.error("Error in batch recovery", error=str(e))
        
        return recovered_count
    
    async def add_task(self, task_data: Dict[str, Any]) -> str:
        """
        Add a task to the queue, using fallback storage if database is unavailable.
        
        Args:
            task_data: Task data dictionary
            
        Returns:
            Task ID
        """
        if self.current_mode == FallbackMode.NORMAL:
            # Try to add to database first
            try:
                async with self.db_manager.get_session() as session:
                    repo_factory = RepositoryFactory(session)
                    task_repo = repo_factory.task_queue
                    
                    task_create = TaskQueueCreate(
                        task_type=task_data.get('task_type', 'unknown'),
                        payload=task_data.get('payload', {}),
                        priority=task_data.get('priority', 5),
                        max_retries=task_data.get('max_retries', 3)
                    )
                    
                    db_task = await task_repo.create(task_create)
                    
                    logger.debug("Task added to database queue",
                               task_id=db_task.id,
                               task_type=task_create.task_type)
                    
                    return str(db_task.id)
                    
            except Exception as e:
                logger.warning("Failed to add task to database, using fallback",
                             error=str(e))
                # Fall through to fallback storage
        
        # Use fallback storage
        return await self.store_task_in_fallback(task_data)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status information.
        
        Returns:
            Status dictionary with current state and statistics
        """
        try:
            # Check current database health
            db_healthy = await self.db_manager.health_check()
            
            # Get R2 storage status
            try:
                stored_task_ids = await self.spinal_cord.list_stored_tasks(limit=10)
                r2_healthy = True
                r2_stored_tasks = len(stored_task_ids)
            except Exception as e:
                r2_healthy = False
                r2_stored_tasks = -1
                logger.error("Error checking R2 storage status", error=str(e))
            
            status = {
                "current_mode": self.current_mode,
                "database_healthy": db_healthy,
                "r2_storage_healthy": r2_healthy,
                "r2_stored_tasks": r2_stored_tasks,
                "pending_tasks": len(self._pending_tasks),
                "recovery_in_progress": self._recovery_in_progress,
                "health_check_interval": self.health_check_interval,
                "stats": self.stats.to_dict(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error("Error getting system status", error=str(e))
            return {
                "current_mode": self.current_mode,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def force_recovery(self) -> Dict[str, Any]:
        """
        Force a recovery operation (for manual intervention).
        
        Returns:
            Recovery result information
        """
        if self._recovery_in_progress:
            return {
                "status": "error",
                "message": "Recovery already in progress"
            }
        
        try:
            logger.info("Manual recovery initiated")
            
            # Check database health first
            db_healthy = await self.db_manager.health_check()
            if not db_healthy:
                return {
                    "status": "error",
                    "message": "Database is not healthy, cannot perform recovery"
                }
            
            # Perform recovery
            self.current_mode = FallbackMode.RECOVERY
            recovered_count = await self._recover_tasks_from_fallback()
            self.current_mode = FallbackMode.NORMAL
            
            return {
                "status": "success",
                "recovered_tasks": recovered_count,
                "message": f"Successfully recovered {recovered_count} tasks"
            }
            
        except Exception as e:
            logger.error("Error in forced recovery", error=str(e))
            self.current_mode = FallbackMode.FALLBACK  # Reset to fallback mode
            return {
                "status": "error",
                "message": f"Recovery failed: {str(e)}"
            }
    
    async def cleanup_old_fallback_tasks(self, max_age_hours: int = 72) -> int:
        """
        Clean up old tasks from fallback storage.
        
        Args:
            max_age_hours: Maximum age of tasks to keep
            
        Returns:
            Number of tasks cleaned up
        """
        try:
            cleaned_count = await self.spinal_cord.cleanup_old_tasks(max_age_hours)
            
            logger.info("Cleaned up old fallback tasks",
                       cleaned_count=cleaned_count,
                       max_age_hours=max_age_hours)
            
            return cleaned_count
            
        except Exception as e:
            logger.error("Error cleaning up old fallback tasks", error=str(e))
            raise


# Global instance for dependency injection
_fallback_manager: Optional[TaskQueueFallbackManager] = None


async def get_fallback_manager() -> TaskQueueFallbackManager:
    """
    Get TaskQueueFallbackManager instance for dependency injection.
    
    Usage in FastAPI:
        @app.post("/tasks")
        async def create_task(
            task_data: dict,
            fallback_manager: TaskQueueFallbackManager = Depends(get_fallback_manager)
        ):
            task_id = await fallback_manager.add_task(task_data)
            return {"task_id": task_id}
    """
    global _fallback_manager
    if _fallback_manager is None:
        _fallback_manager = TaskQueueFallbackManager()
        await _fallback_manager.start()
    return _fallback_manager


async def cleanup_fallback_manager():
    """Cleanup the global fallback manager instance."""
    global _fallback_manager
    if _fallback_manager:
        await _fallback_manager.stop()
        _fallback_manager = None