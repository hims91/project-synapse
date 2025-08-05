"""
Central Cortex - Monitoring Router
Layer 3: Cerebral Cortex

This module implements system monitoring and alerting endpoints.
Provides real-time system health, performance metrics, and alerting.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import (
    get_db_session_dependency, get_repository_factory, get_current_user,
    require_user_tier, require_active_user, get_system_health,
    get_database_manager, get_task_dispatcher, get_fallback_manager
)
from ...shared.schemas import MonitoringSubscriptionCreate, MonitoringSubscriptionResponse
from ...synaptic_vesicle.repositories import RepositoryFactory

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket connection manager for real-time monitoring
class ConnectionManager:
    """Manages WebSocket connections for real-time monitoring."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

# Global connection manager
connection_manager = ConnectionManager()


@router.get("/dashboard")
async def get_monitoring_dashboard(
    current_user: Dict[str, Any] = Depends(require_active_user)
) -> Dict[str, Any]:
    """
    Get comprehensive monitoring dashboard data.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Dashboard data with system metrics
    """
    try:
        # Get system health
        health_info = await get_system_health()
        
        # Get component statistics
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": health_info,
            "metrics": {}
        }
        
        # Database metrics
        try:
            db_manager = await get_database_manager()
            db_health = await db_manager.health_check()
            dashboard_data["metrics"]["database"] = {
                "status": db_health["status"],
                "connection_pool": db_health.get("connection_pool", {}),
                "query_performance": db_health.get("query_performance", {})
            }
        except Exception as e:
            logger.error(f"Failed to get database metrics: {e}")
            dashboard_data["metrics"]["database"] = {"status": "error", "error": str(e)}
        
        # Task dispatcher metrics
        try:
            task_dispatcher = await get_task_dispatcher()
            dispatcher_stats = task_dispatcher.get_stats()
            dashboard_data["metrics"]["task_dispatcher"] = {
                "status": "healthy",
                "queue_size": dispatcher_stats.get("queue_size", 0),
                "active_workers": dispatcher_stats.get("active_workers", 0),
                "completed_tasks": dispatcher_stats.get("completed_tasks", 0),
                "failed_tasks": dispatcher_stats.get("failed_tasks", 0),
                "average_execution_time": dispatcher_stats.get("average_execution_time", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get task dispatcher metrics: {e}")
            dashboard_data["metrics"]["task_dispatcher"] = {"status": "error", "error": str(e)}
        
        # Fallback manager metrics
        try:
            fallback_manager = await get_fallback_manager()
            fallback_stats = fallback_manager.get_stats()
            dashboard_data["metrics"]["fallback_manager"] = {
                "status": "healthy",
                "mode": fallback_stats.get("mode", "normal"),
                "fallback_tasks": fallback_stats.get("fallback_tasks", 0),
                "recovered_tasks": fallback_stats.get("recovered_tasks", 0),
                "success_rate": fallback_stats.get("success_rate", 1.0)
            }
        except Exception as e:
            logger.error(f"Failed to get fallback manager metrics: {e}")
            dashboard_data["metrics"]["fallback_manager"] = {"status": "error", "error": str(e)}
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get monitoring dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve monitoring dashboard"
        )


@router.get("/metrics")
async def get_system_metrics(
    time_range: str = Query("1h", description="Time range (1h, 6h, 24h, 7d)"),
    current_user: Dict[str, Any] = Depends(require_active_user)
) -> Dict[str, Any]:
    """
    Get detailed system metrics over time.
    
    Args:
        time_range: Time range for metrics
        current_user: Current authenticated user
        
    Returns:
        Time-series metrics data
    """
    try:
        # Validate time range
        valid_ranges = ["1h", "6h", "24h", "7d"]
        if time_range not in valid_ranges:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid time range. Must be one of: {valid_ranges}"
            )
        
        # Calculate time window
        time_windows = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(days=1),
            "7d": timedelta(days=7)
        }
        
        end_time = datetime.utcnow()
        start_time = end_time - time_windows[time_range]
        
        # Get metrics (simplified implementation)
        metrics_data = {
            "time_range": time_range,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "metrics": {
                "system_health": [],
                "task_throughput": [],
                "error_rates": [],
                "response_times": []
            }
        }
        
        # In a real implementation, this would query historical metrics from a time-series database
        # For now, return current snapshot
        current_health = await get_system_health()
        metrics_data["current_snapshot"] = current_health
        
        return metrics_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system metrics"
        )


@router.get("/alerts")
async def get_active_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity (low, medium, high, critical)"),
    current_user: Dict[str, Any] = Depends(require_user_tier("premium"))
) -> List[Dict[str, Any]]:
    """
    Get active system alerts.
    
    Args:
        severity: Severity filter
        current_user: Current authenticated user
        
    Returns:
        List of active alerts
    """
    try:
        # Get alerts from fallback manager
        fallback_manager = await get_fallback_manager()
        alerts = fallback_manager.get_current_alerts()
        
        # Filter by severity if specified
        if severity:
            alerts = [alert for alert in alerts if alert.get("severity") == severity]
        
        return alerts
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve alerts"
        )


@router.post("/subscriptions", response_model=MonitoringSubscriptionResponse)
async def create_monitoring_subscription(
    subscription_data: MonitoringSubscriptionCreate,
    current_user: Dict[str, Any] = Depends(require_user_tier("premium")),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> MonitoringSubscriptionResponse:
    """
    Create monitoring subscription for keyword alerts.
    
    Args:
        subscription_data: Subscription data
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        Created subscription
    """
    try:
        monitoring_repo = repo_factory.get_monitoring_subscription_repository()
        
        # Create subscription with user ID
        subscription = await monitoring_repo.create(
            subscription_data,
            user_id=current_user["user_id"]
        )
        
        logger.info(f"Monitoring subscription created: {subscription.id}")
        
        return subscription
        
    except Exception as e:
        logger.error(f"Failed to create monitoring subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create monitoring subscription"
        )


@router.get("/subscriptions")
async def get_monitoring_subscriptions(
    current_user: Dict[str, Any] = Depends(require_user_tier("premium")),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> List[MonitoringSubscriptionResponse]:
    """
    Get user's monitoring subscriptions.
    
    Args:
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        List of user's subscriptions
    """
    try:
        monitoring_repo = repo_factory.get_monitoring_subscription_repository()
        subscriptions = await monitoring_repo.get_by_user_id(current_user["user_id"])
        
        return subscriptions
        
    except Exception as e:
        logger.error(f"Failed to get monitoring subscriptions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve monitoring subscriptions"
        )


@router.websocket("/live")
async def monitoring_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time monitoring updates.
    
    Args:
        websocket: WebSocket connection
    """
    await connection_manager.connect(websocket)
    
    try:
        while True:
            # Wait for client messages (ping/pong or commands)
            try:
                data = await websocket.receive_json()
                
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("type") == "subscribe":
                    # Handle subscription to specific metrics
                    await websocket.send_json({
                        "type": "subscribed",
                        "metrics": data.get("metrics", [])
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        connection_manager.disconnect(websocket)


@router.get("/performance")
async def get_performance_metrics(
    component: Optional[str] = Query(None, description="Component filter (database, tasks, fallback)"),
    current_user: Dict[str, Any] = Depends(require_active_user)
) -> Dict[str, Any]:
    """
    Get performance metrics for system components.
    
    Args:
        component: Component filter
        current_user: Current authenticated user
        
    Returns:
        Performance metrics data
    """
    try:
        performance_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Database performance
        if not component or component == "database":
            try:
                db_manager = await get_database_manager()
                db_health = await db_manager.health_check()
                performance_data["components"]["database"] = {
                    "connection_pool": db_health.get("connection_pool", {}),
                    "query_performance": db_health.get("query_performance", {}),
                    "status": db_health["status"]
                }
            except Exception as e:
                performance_data["components"]["database"] = {"error": str(e)}
        
        # Task dispatcher performance
        if not component or component == "tasks":
            try:
                task_dispatcher = await get_task_dispatcher()
                stats = task_dispatcher.get_stats()
                performance_data["components"]["tasks"] = {
                    "throughput": stats.get("tasks_per_second", 0),
                    "queue_size": stats.get("queue_size", 0),
                    "average_execution_time": stats.get("average_execution_time", 0),
                    "success_rate": stats.get("success_rate", 1.0)
                }
            except Exception as e:
                performance_data["components"]["tasks"] = {"error": str(e)}
        
        # Fallback manager performance
        if not component or component == "fallback":
            try:
                fallback_manager = await get_fallback_manager()
                stats = fallback_manager.get_stats()
                performance_data["components"]["fallback"] = {
                    "mode": stats.get("mode", "normal"),
                    "recovery_rate": stats.get("recovery_rate", 1.0),
                    "fallback_usage": stats.get("fallback_usage_percentage", 0)
                }
            except Exception as e:
                performance_data["components"]["fallback"] = {"error": str(e)}
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )


# Background task to broadcast real-time updates
async def broadcast_system_updates():
    """Background task to broadcast system updates to WebSocket clients."""
    import asyncio
    
    while True:
        try:
            if connection_manager.active_connections:
                # Get current system health
                health_info = await get_system_health()
                
                # Broadcast to all connected clients
                await connection_manager.broadcast({
                    "type": "system_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": health_info
                })
            
            # Wait 30 seconds before next update
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Error broadcasting system updates: {e}")
            await asyncio.sleep(60)  # Wait longer on error