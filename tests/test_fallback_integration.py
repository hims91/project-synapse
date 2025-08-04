"""
Integration tests for fallback scenarios and recovery.
Tests the complete fallback system workflow including monitoring.
"""
import pytest
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.spinal_cord.fallback_manager import (
    TaskQueueFallbackManager, FallbackMode, get_fallback_manager, cleanup_fallback_manager
)
from src.spinal_cord.monitoring import (
    FallbackMonitor, AlertType, AlertSeverity, get_fallback_monitor, cleanup_fallback_monitor
)
from src.spinal_cord.r2_client import SpinalCordStorage, FallbackTask
from src.synaptic_vesicle.database import DatabaseManager
from src.synaptic_vesicle.repositories import RepositoryFactory, TaskQueueRepository
from src.synaptic_vesicle.models import TaskQueue
from src.shared.schemas import TaskQueueCreate


@pytest.fixture
def mock_spinal_cord():
    """Mock SpinalCordStorage for integration tests."""
    mock = AsyncMock(spec=SpinalCordStorage)
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_db_manager():
    """Mock DatabaseManager for integration tests."""
    mock = AsyncMock(spec=DatabaseManager)
    mock.health_check = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def sample_tasks():
    """Sample tasks for integration testing."""
    return [
        {
            "id": str(uuid.uuid4()),
            "task_type": "scrape_url",
            "payload": {"url": "https://example.com/1"},
            "priority": 1,
            "max_retries": 3
        },
        {
            "id": str(uuid.uuid4()),
            "task_type": "process_feed",
            "payload": {"feed_url": "https://example.com/feed.xml"},
            "priority": 2,
            "max_retries": 3
        },
        {
            "id": str(uuid.uuid4()),
            "task_type": "analyze_content",
            "payload": {"article_id": "123", "analysis_type": "sentiment"},
            "priority": 3,
            "max_retries": 2
        }
    ]


class TestFallbackIntegration:
    """Integration tests for complete fallback workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_fallback_and_recovery_workflow(self, mock_spinal_cord, mock_db_manager, sample_tasks):
        """Test complete workflow from normal operation through fallback to recovery."""
        # Setup manager and monitor
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager,
            health_check_interval=1  # Fast for testing
        )
        
        monitor = FallbackMonitor()
        
        # Track alerts
        alerts_received = []
        async def alert_handler(alert):
            alerts_received.append(alert)
        
        monitor.add_alert_handler(alert_handler)
        
        # Connect manager to monitor
        manager.on_fallback_activated = monitor.on_fallback_activated
        manager.on_recovery_completed = monitor.on_recovery_completed
        manager.on_task_stored = monitor.on_task_stored
        manager.on_task_recovered = monitor.on_task_recovered
        
        await manager.start()
        await monitor.start_monitoring()
        
        try:
            # Phase 1: Normal operation
            assert manager.current_mode == FallbackMode.NORMAL
            
            # Mock successful database operations
            mock_session = AsyncMock()
            mock_db_manager.get_session.return_value.__aenter__.return_value = mock_session
            
            mock_task_repo = AsyncMock(spec=TaskQueueRepository)
            mock_db_task = TaskQueue(id=uuid.uuid4(), task_type="scrape_url", payload={}, priority=1, status="pending")
            mock_task_repo.create.return_value = mock_db_task
            
            with patch('src.spinal_cord.fallback_manager.RepositoryFactory') as mock_repo_factory:
                mock_repo_factory.return_value.task_queue = mock_task_repo
                
                # Add task in normal mode
                task_id = await manager.add_task(sample_tasks[0])
                assert task_id == str(mock_db_task.id)
                mock_task_repo.create.assert_called_once()
            
            # Phase 2: Database failure - activate fallback
            mock_db_manager.health_check.return_value = False
            
            # Trigger health check
            await manager._check_database_health()
            
            assert manager.current_mode == FallbackMode.FALLBACK
            assert manager.stats.fallback_activations == 1
            
            # Verify fallback activation alert
            await asyncio.sleep(0.1)  # Allow alert processing
            assert len(alerts_received) >= 1
            fallback_alert = next((a for a in alerts_received if a.alert_type == AlertType.FALLBACK_ACTIVATED), None)
            assert fallback_alert is not None
            assert fallback_alert.severity == AlertSeverity.ERROR
            
            # Phase 3: Add tasks during fallback
            for task in sample_tasks[1:]:
                task_id = await manager.add_task(task)
                assert task_id is not None
                mock_spinal_cord.store_task.assert_called()
            
            assert manager.stats.tasks_stored_in_fallback == 2
            
            # Phase 4: Database recovery
            mock_db_manager.health_check.return_value = True
            
            # Mock recovery operations
            stored_task_ids = ["task-1", "task-2"]
            mock_spinal_cord.list_stored_tasks.return_value = stored_task_ids
            
            sample_fallback_task = FallbackTask(
                id="task-1",
                task_type="process_feed",
                payload={"feed_url": "https://example.com/feed.xml"},
                priority=2,
                created_at=datetime.now(timezone.utc)
            )
            mock_spinal_cord.retrieve_task.return_value = sample_fallback_task
            mock_spinal_cord.delete_task.return_value = True
            
            # Trigger recovery
            await manager._check_database_health()
            
            # Wait for recovery to complete
            await asyncio.sleep(0.2)
            
            assert manager.current_mode == FallbackMode.NORMAL
            assert manager.stats.recovery_operations == 1
            assert manager.stats.tasks_recovered_from_fallback == 2
            
            # Verify recovery completion alert
            recovery_alert = next((a for a in alerts_received if a.alert_type == AlertType.RECOVERY_COMPLETED), None)
            assert recovery_alert is not None
            assert recovery_alert.severity in [AlertSeverity.INFO, AlertSeverity.WARNING]
            
        finally:
            await manager.stop()
            await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_fallback_with_storage_errors(self, mock_spinal_cord, mock_db_manager, sample_tasks):
        """Test fallback behavior when storage operations fail."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        monitor = FallbackMonitor()
        
        # Track storage error alerts
        storage_errors = []
        async def storage_error_handler(alert):
            if alert.alert_type == AlertType.STORAGE_ERROR:
                storage_errors.append(alert)
        
        monitor.add_alert_handler(storage_error_handler)
        
        await manager.start()
        await monitor.start_monitoring()
        
        try:
            # Set fallback mode
            manager.current_mode = FallbackMode.FALLBACK
            
            # Mock storage failure
            mock_spinal_cord.store_task.side_effect = Exception("R2 storage unavailable")
            
            # Attempt to store task should fail
            with pytest.raises(Exception, match="R2 storage unavailable"):
                await manager.store_task_in_fallback(sample_tasks[0])
            
            # Simulate storage error alert
            await monitor.on_storage_error({
                "error": "R2 storage unavailable",
                "error_type": "connection_error",
                "operation": "store_task",
                "task_id": sample_tasks[0]["id"]
            })
            
            # Verify storage error alert was generated
            assert len(storage_errors) == 1
            assert storage_errors[0].severity == AlertSeverity.CRITICAL
            
        finally:
            await manager.stop()
            await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_recovery_with_partial_failures(self, mock_spinal_cord, mock_db_manager):
        """Test recovery process with some task failures."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager,
            recovery_batch_size=2
        )
        
        monitor = FallbackMonitor()
        
        await manager.start()
        await monitor.start_monitoring()
        
        try:
            # Set recovery mode
            manager.current_mode = FallbackMode.RECOVERY
            
            # Mock stored tasks
            task_ids = ["task-1", "task-2", "task-3"]
            mock_spinal_cord.list_stored_tasks.return_value = task_ids
            
            # Mock successful retrieval for first two tasks
            def mock_retrieve_task(task_id):
                if task_id in ["task-1", "task-2"]:
                    return FallbackTask(
                        id=task_id,
                        task_type="test_task",
                        payload={"test": True},
                        priority=1,
                        created_at=datetime.now(timezone.utc)
                    )
                else:
                    return None  # Simulate missing task
            
            mock_spinal_cord.retrieve_task.side_effect = mock_retrieve_task
            mock_spinal_cord.delete_task.return_value = True
            
            # Mock database operations
            mock_session = AsyncMock()
            mock_db_manager.get_session.return_value.__aenter__.return_value = mock_session
            
            mock_task_repo = AsyncMock(spec=TaskQueueRepository)
            mock_db_task = TaskQueue(id=uuid.uuid4(), task_type="test_task", payload={}, priority=1, status="pending")
            mock_task_repo.create.return_value = mock_db_task
            
            with patch('src.spinal_cord.fallback_manager.RepositoryFactory') as mock_repo_factory:
                mock_repo_factory.return_value.task_queue = mock_task_repo
                
                # Perform recovery
                recovered_count = await manager._recover_tasks_from_fallback()
                
                # Should recover 2 out of 3 tasks
                assert recovered_count == 2
                assert mock_task_repo.create.call_count == 2
                assert mock_spinal_cord.delete_task.call_count == 2
        
        finally:
            await manager.stop()
            await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_health_check_monitoring(self, mock_spinal_cord, mock_db_manager):
        """Test health check monitoring and failure detection."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager,
            health_check_interval=0.1  # Very fast for testing
        )
        
        monitor = FallbackMonitor()
        
        # Track health check alerts
        health_alerts = []
        async def health_alert_handler(alert):
            if alert.alert_type == AlertType.HEALTH_CHECK_FAILED:
                health_alerts.append(alert)
        
        monitor.add_alert_handler(health_alert_handler)
        
        await manager.start()
        await monitor.start_monitoring()
        
        try:
            # Simulate consecutive health check failures
            mock_db_manager.health_check.side_effect = Exception("Database connection failed")
            
            # Simulate multiple failures
            for i in range(4):
                await monitor.on_health_check_failed({
                    "check_type": "database",
                    "error": "Database connection failed"
                })
            
            # Should generate alert after 3 consecutive failures
            assert len(health_alerts) >= 1
            # Find the error-level alert (there may be an initial info alert)
            error_alerts = [alert for alert in health_alerts if alert.severity == AlertSeverity.ERROR]
            assert len(error_alerts) >= 1
            assert "3 times consecutively" in error_alerts[0].message
            
            # Simulate recovery
            mock_db_manager.health_check.return_value = True
            await monitor.on_health_check_success()
            
            # Consecutive failures should be reset
            assert monitor.consecutive_failures == 0
            
        finally:
            await manager.stop()
            await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_monitoring_status_and_history(self, mock_spinal_cord, mock_db_manager):
        """Test monitoring status reporting and alert history."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        monitor = FallbackMonitor(max_alert_history=10)
        
        await manager.start()
        await monitor.start_monitoring()
        
        try:
            # Generate some alerts
            await monitor.on_fallback_activated({"current_mode": FallbackMode.FALLBACK})
            await monitor.on_task_stored({"id": "task-1", "task_type": "test"})
            await monitor.on_recovery_completed({"recovered_tasks": 5, "stats": {}})
            
            # Check monitoring status
            status = monitor.get_monitoring_status()
            assert status["monitoring_active"] is True
            assert status["current_alerts"] >= 0
            assert status["alert_history_size"] >= 3
            
            # Check alert history
            history = monitor.get_alert_history(limit=5)
            assert len(history) <= 5
            assert all("timestamp" in alert for alert in history)
            assert all("alert_type" in alert for alert in history)
            
            # Check current alerts
            current = monitor.get_current_alerts()
            assert isinstance(current, list)
            
        finally:
            await manager.stop()
            await monitor.stop_monitoring()


class TestGlobalInstances:
    """Test global instance management for fallback system."""
    
    @pytest.mark.asyncio
    async def test_global_fallback_manager_lifecycle(self):
        """Test global fallback manager creation and cleanup."""
        # Ensure clean state
        await cleanup_fallback_manager()
        
        with patch('src.spinal_cord.fallback_manager.TaskQueueFallbackManager') as mock_manager_class:
            mock_instance = AsyncMock()
            mock_instance.start = AsyncMock()
            mock_instance.stop = AsyncMock()
            mock_manager_class.return_value = mock_instance
            
            # Get manager instance
            manager1 = await get_fallback_manager()
            assert manager1 == mock_instance
            mock_instance.start.assert_called_once()
            
            # Get same instance again
            manager2 = await get_fallback_manager()
            assert manager2 == manager1
            assert mock_instance.start.call_count == 1  # Should not start again
            
            # Cleanup
            await cleanup_fallback_manager()
            mock_instance.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_global_monitor_lifecycle(self):
        """Test global monitor creation and cleanup."""
        # Ensure clean state
        await cleanup_fallback_monitor()
        
        with patch('src.spinal_cord.monitoring.FallbackMonitor') as mock_monitor_class:
            mock_instance = AsyncMock()
            mock_instance.start_monitoring = AsyncMock()
            mock_instance.stop_monitoring = AsyncMock()
            mock_monitor_class.return_value = mock_instance
            
            # Get monitor instance
            monitor1 = await get_fallback_monitor()
            assert monitor1 == mock_instance
            mock_instance.start_monitoring.assert_called_once()
            
            # Get same instance again
            monitor2 = await get_fallback_monitor()
            assert monitor2 == monitor1
            assert mock_instance.start_monitoring.call_count == 1  # Should not start again
            
            # Cleanup
            await cleanup_fallback_monitor()
            mock_instance.stop_monitoring.assert_called_once()


class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_manager_startup_failure(self, mock_spinal_cord, mock_db_manager):
        """Test handling of manager startup failures."""
        # Mock startup failure
        mock_spinal_cord.__aenter__.side_effect = Exception("Connection failed")
        
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        # Should raise exception on startup failure
        with pytest.raises(Exception, match="Connection failed"):
            await manager.start()
    
    @pytest.mark.asyncio
    async def test_monitor_alert_handler_failures(self):
        """Test handling of alert handler failures."""
        monitor = FallbackMonitor()
        
        # Add failing handler
        def failing_handler(alert):
            raise Exception("Handler failed")
        
        monitor.add_alert_handler(failing_handler)
        
        await monitor.start_monitoring()
        
        try:
            # Should not raise exception even if handler fails
            await monitor.on_fallback_activated({"current_mode": FallbackMode.FALLBACK})
            
            # Alert should still be recorded despite handler failure
            history = monitor.get_alert_history()
            assert len(history) >= 1
            
        finally:
            await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_concurrent_recovery_operations(self, mock_spinal_cord, mock_db_manager):
        """Test handling of concurrent recovery attempts."""
        manager = TaskQueueFallbackManager(
            spinal_cord=mock_spinal_cord,
            db_manager=mock_db_manager
        )
        
        await manager.start()
        
        try:
            # Set fallback mode
            manager.current_mode = FallbackMode.FALLBACK
            
            # Mock empty storage for quick recovery
            mock_spinal_cord.list_stored_tasks.return_value = []
            mock_db_manager.health_check.return_value = True
            
            # Start multiple recovery operations concurrently
            recovery_tasks = [
                asyncio.create_task(manager._initiate_recovery()),
                asyncio.create_task(manager._initiate_recovery()),
                asyncio.create_task(manager._initiate_recovery())
            ]
            
            # Wait for all to complete
            await asyncio.gather(*recovery_tasks, return_exceptions=True)
            
            # Should have completed successfully without conflicts
            assert manager.current_mode == FallbackMode.NORMAL
            assert not manager._recovery_in_progress
            
        finally:
            await manager.stop()