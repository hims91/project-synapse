"""
Central Cortex - Health Check Router
Layer 3: Cerebral Cortex

This module implements health check endpoints for system monitoring.
Provides comprehensive health status for all system components.
"""
import logging
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import JSONResponse

from ..dependencies import get_system_health, get_database_manager, get_task_dispatcher, get_fallback_manager
from ...shared.schemas import HealthCheckResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=HealthCheckResponse)
@router.get("", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Comprehensive health check endpoint.
    
    Returns:
        System health status with component details
    """
    try:
        health_info = await get_system_health()
        
        # Determine HTTP status code based on health
        status_code = status.HTTP_200_OK
        if health_info["status"] == "degraded":
            status_code = status.HTTP_207_MULTI_STATUS
        elif health_info["status"] == "unhealthy":
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": health_info["status"],
                "timestamp": datetime.utcnow().isoformat(),
                "version": "2.2.0",
                "components": health_info["components"]
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "2.2.0",
                "error": str(e)
            }
        )


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint for Kubernetes/container orchestration.
    
    Returns:
        Simple ready status
    """
    try:
        # Check if core components are ready
        db_manager = await get_database_manager()
        db_health = await db_manager.health_check()
        
        if db_health["status"] != "healthy":
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_ready",
                    "reason": "database_not_ready"
                }
            )
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "reason": str(e)
            }
        )


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check endpoint for Kubernetes/container orchestration.
    
    Returns:
        Simple alive status
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/components")
async def component_health() -> Dict[str, Any]:
    """
    Detailed component health check.
    
    Returns:
        Health status for each system component
    """
    try:
        components = {}
        
        # Database health
        try:
            db_manager = await get_database_manager()
            db_health = await db_manager.health_check()
            components["database"] = db_health
        except Exception as e:
            components["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Task dispatcher health
        try:
            task_dispatcher = await get_task_dispatcher()
            dispatcher_stats = task_dispatcher.get_stats()
            components["task_dispatcher"] = {
                "status": "healthy",
                "stats": dispatcher_stats
            }
        except Exception as e:
            components["task_dispatcher"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Fallback manager health
        try:
            fallback_manager = await get_fallback_manager()
            fallback_stats = fallback_manager.get_stats()
            components["fallback_manager"] = {
                "status": "healthy",
                "stats": fallback_stats
            }
        except Exception as e:
            components["fallback_manager"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Determine overall status
        overall_status = "healthy"
        unhealthy_count = sum(1 for comp in components.values() if comp.get("status") != "healthy")
        
        if unhealthy_count > 0:
            if unhealthy_count == len(components):
                overall_status = "unhealthy"
            else:
                overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": components
        }
        
    except Exception as e:
        logger.error(f"Component health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )


@router.get("/metrics")
async def health_metrics() -> Dict[str, Any]:
    """
    Health metrics endpoint for monitoring systems.
    
    Returns:
        System metrics and performance data
    """
    try:
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": 0,  # TODO: Implement uptime tracking
            "version": "2.2.0"
        }
        
        # Database metrics
        try:
            db_manager = await get_database_manager()
            db_health = await db_manager.health_check()
            metrics["database"] = {
                "status": db_health["status"],
                "connection_pool_size": db_health.get("connection_pool", {}).get("size", 0),
                "active_connections": db_health.get("connection_pool", {}).get("checked_out", 0)
            }
        except Exception as e:
            metrics["database"] = {"status": "error", "error": str(e)}
        
        # Task dispatcher metrics
        try:
            task_dispatcher = await get_task_dispatcher()
            dispatcher_stats = task_dispatcher.get_stats()
            metrics["task_dispatcher"] = {
                "status": "healthy",
                "queue_size": dispatcher_stats.get("queue_size", 0),
                "active_workers": dispatcher_stats.get("active_workers", 0),
                "completed_tasks": dispatcher_stats.get("completed_tasks", 0),
                "failed_tasks": dispatcher_stats.get("failed_tasks", 0)
            }
        except Exception as e:
            metrics["task_dispatcher"] = {"status": "error", "error": str(e)}
        
        # Fallback manager metrics
        try:
            fallback_manager = await get_fallback_manager()
            fallback_stats = fallback_manager.get_stats()
            metrics["fallback_manager"] = {
                "status": "healthy",
                "mode": fallback_stats.get("mode", "unknown"),
                "fallback_tasks": fallback_stats.get("fallback_tasks", 0),
                "recovered_tasks": fallback_stats.get("recovered_tasks", 0)
            }
        except Exception as e:
            metrics["fallback_manager"] = {"status": "error", "error": str(e)}
        
        return metrics
        
    except Exception as e:
        logger.error(f"Health metrics failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )