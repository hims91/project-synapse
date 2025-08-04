"""
Spinal Cord - Fallback System Monitoring and Alerting
Layer 2: Signal Network

This module implements monitoring and alerting for the fallback system activation.
Provides real-time notifications and health status tracking.
"""
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import structlog

from ..shared.config import get_settings
from .fallback_manager import FallbackMode, FallbackStats

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of fallback system alerts."""
    FALLBACK_ACTIVATED = "fallback_activated"
    RECOVERY_COMPLETED = "recovery_completed"
    RECOVERY_FAILED = "recovery_failed"
    STORAGE_ERROR = "storage_error"
    DATABASE_OUTAGE = "database_outage"
    TASK_STORED = "task_stored"
    TASK_RECOVERED = "task_recovered"
    HEALTH_CHECK_FAILED = "health_check_failed"


class FallbackAlert:
    """Represents a fallback system alert."""
    
    def __init__(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.id = f"{alert_type}_{int(self.timestamp.timestamp())}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "id": self.id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "component": "spinal_cord_fallback"
        }


class FallbackMonitor:
    """
    Monitors fallback system operations and generates alerts.
    
    Features:
    - Real-time monitoring of fallback system state
    - Alert generation for critical events
    - Health status tracking and reporting
    - Integration with external monitoring systems
    - Webhook notifications for alerts
    """
    
    def __init__(
        self,
        alert_handlers: Optional[List[Callable]] = None,
        max_alert_history: int = 1000
    ):
        self.alert_handlers = alert_handlers or []
        self.max_alert_history = max_alert_history
        self.alert_history: List[FallbackAlert] = []
        self.current_alerts: Dict[str, FallbackAlert] = {}
        self.settings = get_settings()
        
        # Monitoring state
        self.monitoring_active = False
        self.last_health_check = None
        self.consecutive_failures = 0
        
    async def start_monitoring(self) -> None:
        """Start the monitoring system."""
        try:
            self.monitoring_active = True
            
            logger.info("Fallback system monitoring started",
                       max_alert_history=self.max_alert_history,
                       alert_handlers=len(self.alert_handlers))
            
            # Send startup notification
            await self._send_alert(
                AlertType.HEALTH_CHECK_FAILED,  # Using as generic startup
                AlertSeverity.INFO,
                "Fallback system monitoring started",
                {"monitoring_active": True}
            )
            
        except Exception as e:
            logger.error("Failed to start fallback monitoring", error=str(e))
            raise
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        try:
            self.monitoring_active = False
            
            logger.info("Fallback system monitoring stopped")
            
        except Exception as e:
            logger.error("Error stopping fallback monitoring", error=str(e))
    
    async def on_fallback_activated(self, stats: Dict[str, Any]) -> None:
        """Handle fallback activation event."""
        try:
            await self._send_alert(
                AlertType.FALLBACK_ACTIVATED,
                AlertSeverity.ERROR,
                "Database outage detected - Fallback system activated",
                {
                    "current_mode": stats.get("current_mode"),
                    "fallback_activations": stats.get("fallback_activations"),
                    "last_activation": stats.get("last_fallback_activation"),
                    "database_healthy": False
                }
            )
            
            logger.warning("Fallback activation alert sent",
                          activations=stats.get("fallback_activations"))
            
        except Exception as e:
            logger.error("Error handling fallback activation alert", error=str(e))
    
    async def on_recovery_completed(self, recovery_data: Dict[str, Any]) -> None:
        """Handle recovery completion event."""
        try:
            recovered_tasks = recovery_data.get("recovered_tasks", 0)
            stats = recovery_data.get("stats", {})
            
            severity = AlertSeverity.INFO if recovered_tasks == 0 else AlertSeverity.WARNING
            
            await self._send_alert(
                AlertType.RECOVERY_COMPLETED,
                severity,
                f"Database connectivity restored - {recovered_tasks} tasks recovered",
                {
                    "recovered_tasks": recovered_tasks,
                    "current_mode": stats.get("current_mode"),
                    "recovery_operations": stats.get("recovery_operations"),
                    "total_outage_duration": stats.get("total_outage_duration_seconds"),
                    "database_healthy": True
                }
            )
            
            # Clear fallback activation alert
            self._clear_alert(AlertType.FALLBACK_ACTIVATED)
            
            logger.info("Recovery completion alert sent",
                       recovered_tasks=recovered_tasks)
            
        except Exception as e:
            logger.error("Error handling recovery completion alert", error=str(e))
    
    async def on_task_stored(self, task_data: Dict[str, Any]) -> None:
        """Handle task stored in fallback event."""
        try:
            # Only send alert for first few tasks to avoid spam
            if len(self.alert_history) < 5 or self.consecutive_failures > 10:
                await self._send_alert(
                    AlertType.TASK_STORED,
                    AlertSeverity.WARNING,
                    f"Task stored in fallback storage: {task_data.get('task_type')}",
                    {
                        "task_id": task_data.get("id"),
                        "task_type": task_data.get("task_type"),
                        "priority": task_data.get("priority"),
                        "fallback_storage": True
                    }
                )
            
            logger.debug("Task stored in fallback alert processed",
                        task_id=task_data.get("id"))
            
        except Exception as e:
            logger.error("Error handling task stored alert", error=str(e))
    
    async def on_task_recovered(self, task_data: Dict[str, Any]) -> None:
        """Handle task recovered from fallback event."""
        try:
            logger.debug("Task recovered from fallback",
                        fallback_task_id=task_data.get("fallback_task_id"),
                        db_task_id=task_data.get("db_task_id"))
            
            # Don't send individual recovery alerts to avoid spam
            # Recovery completion alert covers the overall operation
            
        except Exception as e:
            logger.error("Error handling task recovered alert", error=str(e))
    
    async def on_storage_error(self, error_details: Dict[str, Any]) -> None:
        """Handle storage error event."""
        try:
            await self._send_alert(
                AlertType.STORAGE_ERROR,
                AlertSeverity.CRITICAL,
                f"Fallback storage error: {error_details.get('error')}",
                {
                    "error_type": error_details.get("error_type"),
                    "operation": error_details.get("operation"),
                    "task_id": error_details.get("task_id"),
                    "retry_count": error_details.get("retry_count", 0)
                }
            )
            
            self.consecutive_failures += 1
            
            logger.error("Storage error alert sent",
                        error=error_details.get("error"),
                        consecutive_failures=self.consecutive_failures)
            
        except Exception as e:
            logger.error("Error handling storage error alert", error=str(e))
    
    async def on_health_check_failed(self, check_details: Dict[str, Any]) -> None:
        """Handle health check failure event."""
        try:
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= 3:  # Alert after 3 consecutive failures
                await self._send_alert(
                    AlertType.HEALTH_CHECK_FAILED,
                    AlertSeverity.ERROR,
                    f"Health check failed {self.consecutive_failures} times consecutively",
                    {
                        "consecutive_failures": self.consecutive_failures,
                        "last_success": self.last_health_check.isoformat() if self.last_health_check else None,
                        "check_type": check_details.get("check_type"),
                        "error": check_details.get("error")
                    }
                )
            
            logger.warning("Health check failure recorded",
                          consecutive_failures=self.consecutive_failures)
            
        except Exception as e:
            logger.error("Error handling health check failure alert", error=str(e))
    
    async def on_health_check_success(self) -> None:
        """Handle successful health check."""
        try:
            if self.consecutive_failures > 0:
                # Clear health check failure alert
                self._clear_alert(AlertType.HEALTH_CHECK_FAILED)
                
                logger.info("Health check recovered",
                           previous_failures=self.consecutive_failures)
            
            self.consecutive_failures = 0
            self.last_health_check = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error("Error handling health check success", error=str(e))
    
    async def _send_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        details: Dict[str, Any]
    ) -> None:
        """Send an alert through all configured handlers."""
        try:
            alert = FallbackAlert(alert_type, severity, message, details)
            
            # Add to history
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_alert_history:
                self.alert_history.pop(0)
            
            # Update current alerts
            self.current_alerts[alert_type] = alert
            
            # Send through all handlers
            for handler in self.alert_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    logger.error("Error in alert handler", handler=str(handler), error=str(e))
            
            logger.info("Alert sent",
                       alert_type=alert_type,
                       severity=severity,
                       handlers=len(self.alert_handlers))
            
        except Exception as e:
            logger.error("Error sending alert", error=str(e))
    
    def _clear_alert(self, alert_type: AlertType) -> None:
        """Clear a current alert."""
        if alert_type in self.current_alerts:
            del self.current_alerts[alert_type]
            logger.debug("Alert cleared", alert_type=alert_type)
    
    def get_current_alerts(self) -> List[Dict[str, Any]]:
        """Get all current active alerts."""
        return [alert.to_dict() for alert in self.current_alerts.values()]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        recent_alerts = self.alert_history[-limit:] if limit > 0 else self.alert_history
        return [alert.to_dict() for alert in recent_alerts]
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring system status."""
        return {
            "monitoring_active": self.monitoring_active,
            "alert_handlers": len(self.alert_handlers),
            "current_alerts": len(self.current_alerts),
            "alert_history_size": len(self.alert_history),
            "consecutive_failures": self.consecutive_failures,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def add_alert_handler(self, handler: Callable) -> None:
        """Add an alert handler."""
        self.alert_handlers.append(handler)
        logger.info("Alert handler added", total_handlers=len(self.alert_handlers))
    
    def remove_alert_handler(self, handler: Callable) -> None:
        """Remove an alert handler."""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)
            logger.info("Alert handler removed", total_handlers=len(self.alert_handlers))


# Global monitor instance
_fallback_monitor: Optional[FallbackMonitor] = None


async def get_fallback_monitor() -> FallbackMonitor:
    """Get FallbackMonitor instance for dependency injection."""
    global _fallback_monitor
    if _fallback_monitor is None:
        _fallback_monitor = FallbackMonitor()
        await _fallback_monitor.start_monitoring()
    return _fallback_monitor


async def cleanup_fallback_monitor():
    """Cleanup the global fallback monitor instance."""
    global _fallback_monitor
    if _fallback_monitor:
        await _fallback_monitor.stop_monitoring()
        _fallback_monitor = None


# Default alert handlers
async def webhook_alert_handler(alert: FallbackAlert) -> None:
    """Send alert via webhook (placeholder implementation)."""
    try:
        settings = get_settings()
        
        # This would integrate with actual webhook service
        logger.info("Webhook alert sent",
                   alert_id=alert.id,
                   alert_type=alert.alert_type,
                   severity=alert.severity)
        
        # TODO: Implement actual webhook delivery
        # await send_webhook(settings.monitoring.webhook_url, alert.to_dict())
        
    except Exception as e:
        logger.error("Error sending webhook alert", error=str(e))


def console_alert_handler(alert: FallbackAlert) -> None:
    """Log alert to console."""
    try:
        log_method = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical
        }.get(alert.severity, logger.info)
        
        log_method("FALLBACK ALERT",
                  alert_type=alert.alert_type,
                  message=alert.message,
                  details=alert.details)
        
    except Exception as e:
        logger.error("Error in console alert handler", error=str(e))