"""
Unit tests for Task Queue Fallback Manager.
Tests the automatic fallback and recovery mechanisms.
"""
import pytest
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.spinal_cord.fallback_manager import (
    TaskQueueFallbackManager, FallbackMode, FallbackStats,
    get_fallback_manager, cleanup_fallback_manager
)
from src.spinal_cord.r2_client import SpinalCordStorage, FallbackTask
from src.synaptic_vesicle.database import DatabaseManager
from src.synaptic_vesicle.repositories import RepositoryFactory, TaskQueueRepository
from src.synaptic_vesicle.models import TaskQueue
from src.shared.schemas import TaskQueueCreate


@pytest.fixture
def mock_spinal_cord():
    """Mock SpinalCordStorage."""
    mock = AsyncMock(spec=SpinalCordStorage)
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_db_manager():
    """Mock DatabaseManager."""
    mock = AsyncMock(spec=DatabaseManager)
    mock.health_check = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def sample_task_data():
    """Sample task data for testing."""
    return {
        "id": str(uuid.uuid4()),
        "task_type": "scrape_url",
        "payload": {"url": "https://example.com", "priority": True},
        "priority": 1,
        "max_retries": 3
    }


@pytest.fixture
def sample_fallback_task():
    """Sample FallbackTask for testing."""
    return FallbackTask(
        id=str(uuid.uuid4()),
        task_type="scrape_url",
        payload={"url": "https://example.com"},
        priority=1,
        created_at=datetime.now(timezone.utc)
    )


class TestFallbackStats:
    """Test FallbackStats functionality."""
    
    def test_initial_state(self):
        """Test initial stats state."""
        stats = FallbackStats()
        
        assert stats.tasks_stored_in_fallback == 0
        assert stats.tasks_recovered_from_fallback == 0
        assert stats.fallback_activations == 0
        assert stats.recovery_operations == 0
        assert stats.current_mode == FallbackMode.NORMAL
        assert stats.database_outage_duration == timedelta(0)
    
    def test_start_outage(self):
        """Test outage start tracking."""
        stats = FallbackStats()
        
        stats.start_outage()
        
        assert stats.fallback_activations == 1
        assert stats.current_mode == FallbackMode.FALLBACK
        assert stats.last_fallback_activation is not None
        assert stats._outage_start_time is not None
    
    def test_end_outage(self):
        """Test outage end tracking."""
        stats = FallbackStats()
        
        stats.start_outage()
        # Simulate some time passing
        stats._outage_start_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        stats.end_outage()
        
        assert stats.recovery_operations == 1
        assert stats.current_mode == FallbackMode.NORMAL
        assert stats.last_recovery_completion is not None
        assert stats.database_outage_duration > timedelta(0)
        assert stats._outage_start_time is None
    
    def test_to_dict(self):
        """Test stats serialization."""
        stats = FallbackStats()
        stats.tasks_stored_in_fallback = 5
        stats.tasks_recovered_from_fallback = 3
        
        stats_dict = stats.to_dict()
        
        assert stats_dict["tasks_stored_in_fallback"] == 5
        assert stats_dict["tasks_recovered_from_fallback"] == 3
        assert stats_dict["current_mode"] == FallbackMode.NORMAL
        assert "total_outage_duration_seconds" in stats_dict
        assert "current_outage_duration_seconds" in stats_dict


class TestTaskQueueFallbackManager:
    """Test TaskQueueFallbackManager functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_spinal_cord, mock_db_manager):
        """Test manager initialization."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager,
            health_check_interval=10
        )
        
        assert manager.current_mode == FallbackMode.NORMAL
        assert manager.health_check_interval == 10
        assert isinstance(manager.stats, FallbackStats)
        assert not manager._recovery_in_progress
    
    @pytest.mark.asyncio
    async def test_start_and_stop(self, mock_spinal_cord, mock_db_manager):
        """Test manager start and stop operations."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager,
            health_check_interval=1  # Short interval for testing
        )
        
        # Start manager
        await manager.start()
        
        assert manager._health_check_task is not None
        assert not manager._health_check_task.done()
        mock_spinal_cord.__aenter__.assert_called_once()
        
        # Stop manager
        await manager.stop()
        
        assert manager._health_check_task.cancelled()
        mock_spinal_cord.__aexit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_activate_fallback(self, mock_spinal_cord, mock_db_manager):
        """Test fallback activation."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Mock callback
        callback_called = False
        async def fallback_callback(stats):
            nonlocal callback_called
            callback_called = True
            assert stats["current_mode"] == FallbackMode.FALLBACK
        
        manager.on_fallback_activated = fallback_callback
        
        # Activate fallback
        await manager._activate_fallback()
        
        assert manager.current_mode == FallbackMode.FALLBACK
        assert manager.stats.fallback_activations == 1
        assert manager.stats.last_fallback_activation is not None
        assert callback_called
    
    @pytest.mark.asyncio
    async def test_store_task_in_fallback(self, mock_spinal_cord, mock_db_manager, sample_task_data):
        """Test storing task in fallback storage."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Mock callback
        callback_called = False
        async def task_stored_callback(task_data):
            nonlocal callback_called
            callback_called = True
            assert task_data["task_type"] == "scrape_url"
        
        manager.on_task_stored = task_stored_callback
        
        # Store task
        task_id = await manager.store_task_in_fallback(sample_task_data)
        
        assert task_id is not None
        assert manager.stats.tasks_stored_in_fallback == 1
        assert task_id in manager._pending_tasks
        assert callback_called
        mock_spinal_cord.store_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_task_normal_mode(self, mock_spinal_cord, mock_db_manager, sample_task_data):
        """Test adding task in normal mode (database available)."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Mock database session and repository
        mock_session = AsyncMock()
        mock_db_manager.get_session.return_value.__aenter__.return_value = mock_session
        
        mock_task_repo = AsyncMock(spec=TaskQueueRepository)
        mock_db_task = TaskQueue(id=uuid.uuid4(), task_type="scrape_url", payload={}, priority=1, status="pending")
        mock_task_repo.create.return_value = mock_db_task
        
        with patch('src.spinal_cord.fallback_manager.RepositoryFactory') as mock_repo_factory:
            mock_repo_factory.return_value.task_queue = mock_task_repo
            
            task_id = await manager.add_task(sample_task_data)
            
            assert task_id == str(mock_db_task.id)
            mock_task_repo.create.assert_called_once()
            # Should not use fallback storage
            mock_spinal_cord.store_task.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_add_task_fallback_mode(self, mock_spinal_cord, mock_db_manager, sample_task_data):
        """Test adding task in fallback mode (database unavailable)."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Set fallback mode
        manager.current_mode = FallbackMode.FALLBACK
        
        task_id = await manager.add_task(sample_task_data)
        
        assert task_id is not None
        assert manager.stats.tasks_stored_in_fallback == 1
        mock_spinal_cord.store_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_task_database_failure(self, mock_spinal_cord, mock_db_manager, sample_task_data):
        """Test adding task when database fails during operation."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Mock database failure
        mock_db_manager.get_session.side_effect = Exception("Database connection failed")
        
        task_id = await manager.add_task(sample_task_data)
        
        # Should fall back to R2 storage
        assert task_id is not None
        assert manager.stats.tasks_stored_in_fallback == 1
        mock_spinal_cord.store_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_recover_tasks_from_fallback(self, mock_spinal_cord, mock_db_manager, sample_fallback_task):
        """Test recovering tasks from fallback storage."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager,
            recovery_batch_size=2
        )
        
        # Mock stored tasks
        task_ids = ["task-1", "task-2", "task-3"]
        mock_spinal_cord.list_stored_tasks.return_value = task_ids
        mock_spinal_cord.retrieve_task.return_value = sample_fallback_task
        mock_spinal_cord.delete_task.return_value = True
        
        # Mock database session and repository
        mock_session = AsyncMock()
        mock_db_manager.get_session.return_value.__aenter__.return_value = mock_session
        
        mock_task_repo = AsyncMock(spec=TaskQueueRepository)
        mock_db_task = TaskQueue(id=uuid.uuid4(), task_type="scrape_url", payload={}, priority=1, status="pending")
        mock_task_repo.create.return_value = mock_db_task
        
        # Mock callback
        callback_count = 0
        async def task_recovered_callback(task_data):
            nonlocal callback_count
            callback_count += 1
        
        manager.on_task_recovered = task_recovered_callback
        
        with patch('src.spinal_cord.fallback_manager.RepositoryFactory') as mock_repo_factory:
            mock_repo_factory.return_value.task_queue = mock_task_repo
            
            recovered_count = await manager._recover_tasks_from_fallback()
            
            assert recovered_count == 3
            assert manager.stats.tasks_recovered_from_fallback == 3
            assert callback_count == 3
            
            # Verify all tasks were processed
            assert mock_spinal_cord.retrieve_task.call_count == 3
            assert mock_spinal_cord.delete_task.call_count == 3
            assert mock_task_repo.create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_recover_tasks_empty_storage(self, mock_spinal_cord, mock_db_manager):
        """Test recovery when no tasks are in fallback storage."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Mock empty storage
        mock_spinal_cord.list_stored_tasks.return_value = []
        
        recovered_count = await manager._recover_tasks_from_fallback()
        
        assert recovered_count == 0
        assert manager.stats.tasks_recovered_from_fallback == 0
    
    @pytest.mark.asyncio
    async def test_health_check_database_failure(self, mock_spinal_cord, mock_db_manager):
        """Test health check detecting database failure."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Mock database failure
        mock_db_manager.health_check.return_value = False
        
        await manager._check_database_health()
        
        assert manager.current_mode == FallbackMode.FALLBACK
        assert manager.stats.fallback_activations == 1
    
    @pytest.mark.asyncio
    async def test_health_check_database_recovery(self, mock_spinal_cord, mock_db_manager):
        """Test health check detecting database recovery."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Set initial fallback mode
        manager.current_mode = FallbackMode.FALLBACK
        manager.stats.start_outage()
        
        # Mock empty fallback storage for recovery
        mock_spinal_cord.list_stored_tasks.return_value = []
        
        # Mock database recovery
        mock_db_manager.health_check.return_value = True
        
        await manager._check_database_health()
        
        # Should trigger recovery
        assert manager.current_mode == FallbackMode.NORMAL
        assert manager.stats.recovery_operations == 1
    
    @pytest.mark.asyncio
    async def test_get_system_status(self, mock_spinal_cord, mock_db_manager):
        """Test getting system status."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Mock status data
        mock_db_manager.health_check.return_value = True
        mock_spinal_cord.list_stored_tasks.return_value = ["task-1", "task-2"]
        
        status = await manager.get_system_status()
        
        assert status["current_mode"] == FallbackMode.NORMAL
        assert status["database_healthy"] is True
        assert status["r2_storage_healthy"] is True
        assert status["r2_stored_tasks"] == 2
        assert status["recovery_in_progress"] is False
        assert "stats" in status
        assert "timestamp" in status
    
    @pytest.mark.asyncio
    async def test_force_recovery_success(self, mock_spinal_cord, mock_db_manager):
        """Test successful forced recovery."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Mock healthy database and empty storage
        mock_db_manager.health_check.return_value = True
        mock_spinal_cord.list_stored_tasks.return_value = []
        
        result = await manager.force_recovery()
        
        assert result["status"] == "success"
        assert result["recovered_tasks"] == 0
        assert "Successfully recovered" in result["message"]
    
    @pytest.mark.asyncio
    async def test_force_recovery_unhealthy_database(self, mock_spinal_cord, mock_db_manager):
        """Test forced recovery with unhealthy database."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Mock unhealthy database
        mock_db_manager.health_check.return_value = False
        
        result = await manager.force_recovery()
        
        assert result["status"] == "error"
        assert "not healthy" in result["message"]
    
    @pytest.mark.asyncio
    async def test_force_recovery_already_in_progress(self, mock_spinal_cord, mock_db_manager):
        """Test forced recovery when already in progress."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Set recovery in progress
        manager._recovery_in_progress = True
        
        result = await manager.force_recovery()
        
        assert result["status"] == "error"
        assert "already in progress" in result["message"]
    
    @pytest.mark.asyncio
    async def test_cleanup_old_fallback_tasks(self, mock_spinal_cord, mock_db_manager):
        """Test cleaning up old fallback tasks."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Mock cleanup result
        mock_spinal_cord.cleanup_old_tasks.return_value = 5
        
        cleaned_count = await manager.cleanup_old_fallback_tasks(max_age_hours=48)
        
        assert cleaned_count == 5
        mock_spinal_cord.cleanup_old_tasks.assert_called_once_with(48)
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, mock_spinal_cord, mock_db_manager):
        """Test that callback errors don't break the system."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Mock failing callback
        async def failing_callback(data):
            raise Exception("Callback failed")
        
        manager.on_fallback_activated = failing_callback
        
        # Should not raise exception
        await manager._activate_fallback()
        
        # System should still work
        assert manager.current_mode == FallbackMode.FALLBACK
        assert manager.stats.fallback_activations == 1


class TestGlobalFallbackManager:
    """Test global fallback manager functions."""
    
    @pytest.mark.asyncio
    async def test_get_fallback_manager_singleton(self):
        """Test that get_fallback_manager returns singleton instance."""
        # Clean up any existing instance
        await cleanup_fallback_manager()
        
        with patch('src.spinal_cord.fallback_manager.TaskQueueFallbackManager') as mock_manager_class:
            mock_instance = AsyncMock()
            mock_instance.start = AsyncMock()
            mock_manager_class.return_value = mock_instance
            
            # First call should create instance
            manager1 = await get_fallback_manager()
            assert manager1 == mock_instance
            mock_instance.start.assert_called_once()
            
            # Second call should return same instance
            manager2 = await get_fallback_manager()
            assert manager2 == manager1
            # Start should not be called again
            assert mock_instance.start.call_count == 1
        
        # Cleanup
        await cleanup_fallback_manager()
    
    @pytest.mark.asyncio
    async def test_cleanup_fallback_manager(self):
        """Test cleanup of global fallback manager."""
        # Clean up any existing instance
        await cleanup_fallback_manager()
        
        with patch('src.spinal_cord.fallback_manager.TaskQueueFallbackManager') as mock_manager_class:
            mock_instance = AsyncMock()
            mock_instance.start = AsyncMock()
            mock_instance.stop = AsyncMock()
            mock_manager_class.return_value = mock_instance
            
            # Create instance
            manager = await get_fallback_manager()
            assert manager == mock_instance
            
            # Cleanup
            await cleanup_fallback_manager()
            mock_instance.stop.assert_called_once()
            
            # Next call should create new instance
            manager2 = await get_fallback_manager()
            assert manager2 == mock_instance  # Same mock, but would be new instance in reality


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_recovery_with_database_errors(self, mock_spinal_cord, mock_db_manager, sample_fallback_task):
        """Test recovery handling database errors gracefully."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Mock tasks in storage
        mock_spinal_cord.list_stored_tasks.return_value = ["task-1"]
        mock_spinal_cord.retrieve_task.return_value = sample_fallback_task
        
        # Mock database session failure
        mock_db_manager.get_session.side_effect = Exception("Database error")
        
        # Recovery should handle the error gracefully
        recovered_count = await manager._recover_tasks_from_fallback()
        
        # Should return 0 due to database error, but not crash
        assert recovered_count == 0
    
    @pytest.mark.asyncio
    async def test_health_check_with_exceptions(self, mock_spinal_cord, mock_db_manager):
        """Test health check handling exceptions."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Mock health check exception
        mock_db_manager.health_check.side_effect = Exception("Health check failed")
        
        # Should not raise exception and should activate fallback
        await manager._check_database_health()
        
        assert manager.current_mode == FallbackMode.FALLBACK
    
    @pytest.mark.asyncio
    async def test_store_task_with_r2_error(self, mock_spinal_cord, mock_db_manager, sample_task_data):
        """Test storing task when R2 storage fails."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Mock R2 storage failure
        mock_spinal_cord.store_task.side_effect = Exception("R2 storage failed")
        
        # Should raise exception
        with pytest.raises(Exception, match="R2 storage failed"):
            await manager.store_task_in_fallback(sample_task_data)