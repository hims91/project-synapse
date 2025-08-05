"""
WebSocket event types and data structures.

Defines the event types and data structures used for WebSocket communication
between the server and clients.
"""

from enum import Enum
from typing import Any, Dict, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID


class EventType(str, Enum):
    """WebSocket event types."""
    # Job-related events
    JOB_CREATED = "job_created"
    JOB_STARTED = "job_started"
    JOB_PROGRESS = "job_progress"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"
    
    # System health events
    SYSTEM_HEALTH = "system_health"
    SERVICE_STATUS = "service_status"
    RESOURCE_USAGE = "resource_usage"
    
    # Monitoring alerts
    ALERT_CREATED = "alert_created"
    ALERT_RESOLVED = "alert_resolved"
    ALERT_ESCALATED = "alert_escalated"
    
    # Feed updates
    FEED_UPDATED = "feed_updated"
    FEED_ERROR = "feed_error"
    
    # Analytics updates
    ANALYTICS_UPDATE = "analytics_update"
    TREND_DETECTED = "trend_detected"
    
    # User-specific events
    USER_NOTIFICATION = "user_notification"
    USER_SESSION = "user_session"
    
    # Connection events
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_AUTHENTICATED = "connection_authenticated"
    CONNECTION_ERROR = "connection_error"
    HEARTBEAT = "heartbeat"


class WebSocketEvent(BaseModel):
    """Base WebSocket event structure."""
    event_type: EventType = Field(..., description="Type of the event")
    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    user_id: Optional[str] = Field(None, description="Target user ID")
    channel: Optional[str] = Field(None, description="Target channel")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event payload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class JobEvent(WebSocketEvent):
    """Job-related WebSocket event."""
    job_id: str = Field(..., description="Job identifier")
    job_type: str = Field(..., description="Type of job")
    status: str = Field(..., description="Current job status")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Job progress percentage")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    result_data: Optional[Dict[str, Any]] = Field(None, description="Job result data")


class SystemHealthEvent(WebSocketEvent):
    """System health WebSocket event."""
    component: str = Field(..., description="System component name")
    status: str = Field(..., description="Component status")
    metrics: Dict[str, Union[int, float, str]] = Field(default_factory=dict, description="Health metrics")
    issues: list[str] = Field(default_factory=list, description="Detected issues")


class AlertEvent(WebSocketEvent):
    """Alert WebSocket event."""
    alert_id: str = Field(..., description="Alert identifier")
    severity: str = Field(..., description="Alert severity level")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    source: str = Field(..., description="Alert source component")
    tags: list[str] = Field(default_factory=list, description="Alert tags")


class FeedEvent(WebSocketEvent):
    """Feed update WebSocket event."""
    feed_id: str = Field(..., description="Feed identifier")
    feed_url: str = Field(..., description="Feed URL")
    items_count: int = Field(..., description="Number of new items")
    last_updated: datetime = Field(..., description="Last update timestamp")
    error_message: Optional[str] = Field(None, description="Error message if update failed")


class AnalyticsEvent(WebSocketEvent):
    """Analytics update WebSocket event."""
    metric_name: str = Field(..., description="Metric name")
    metric_value: Union[int, float] = Field(..., description="Metric value")
    metric_type: str = Field(..., description="Type of metric")
    time_period: str = Field(..., description="Time period for the metric")
    comparison_data: Optional[Dict[str, Any]] = Field(None, description="Comparison data")


class UserNotificationEvent(WebSocketEvent):
    """User notification WebSocket event."""
    notification_id: str = Field(..., description="Notification identifier")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    type: str = Field(..., description="Notification type")
    priority: str = Field(default="normal", description="Notification priority")
    action_url: Optional[str] = Field(None, description="Action URL")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")


class ConnectionEvent(WebSocketEvent):
    """Connection-related WebSocket event."""
    connection_id: str = Field(..., description="Connection identifier")
    client_info: Dict[str, Any] = Field(default_factory=dict, description="Client information")
    authentication_status: str = Field(..., description="Authentication status")
    channels: list[str] = Field(default_factory=list, description="Subscribed channels")


# Event factory functions
def create_job_event(
    event_type: EventType,
    job_id: str,
    job_type: str,
    status: str,
    user_id: Optional[str] = None,
    progress: Optional[float] = None,
    error_message: Optional[str] = None,
    result_data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> JobEvent:
    """Create a job-related WebSocket event."""
    return JobEvent(
        event_type=event_type,
        event_id=f"job_{job_id}_{event_type.value}_{int(datetime.utcnow().timestamp())}",
        user_id=user_id,
        job_id=job_id,
        job_type=job_type,
        status=status,
        progress=progress,
        error_message=error_message,
        result_data=result_data,
        data=kwargs
    )


def create_system_health_event(
    component: str,
    status: str,
    metrics: Optional[Dict[str, Union[int, float, str]]] = None,
    issues: Optional[list[str]] = None,
    **kwargs
) -> SystemHealthEvent:
    """Create a system health WebSocket event."""
    return SystemHealthEvent(
        event_type=EventType.SYSTEM_HEALTH,
        event_id=f"health_{component}_{int(datetime.utcnow().timestamp())}",
        component=component,
        status=status,
        metrics=metrics or {},
        issues=issues or [],
        data=kwargs
    )


def create_alert_event(
    event_type: EventType,
    alert_id: str,
    severity: str,
    title: str,
    message: str,
    source: str,
    tags: Optional[list[str]] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> AlertEvent:
    """Create an alert WebSocket event."""
    return AlertEvent(
        event_type=event_type,
        event_id=f"alert_{alert_id}_{event_type.value}_{int(datetime.utcnow().timestamp())}",
        user_id=user_id,
        alert_id=alert_id,
        severity=severity,
        title=title,
        message=message,
        source=source,
        tags=tags or [],
        data=kwargs
    )


def create_user_notification_event(
    notification_id: str,
    title: str,
    message: str,
    notification_type: str,
    user_id: str,
    priority: str = "normal",
    action_url: Optional[str] = None,
    expires_at: Optional[datetime] = None,
    **kwargs
) -> UserNotificationEvent:
    """Create a user notification WebSocket event."""
    return UserNotificationEvent(
        event_type=EventType.USER_NOTIFICATION,
        event_id=f"notification_{notification_id}_{int(datetime.utcnow().timestamp())}",
        user_id=user_id,
        notification_id=notification_id,
        title=title,
        message=message,
        type=notification_type,
        priority=priority,
        action_url=action_url,
        expires_at=expires_at,
        data=kwargs
    )


def create_analytics_event(
    metric_name: str,
    metric_value: Union[int, float],
    metric_type: str,
    time_period: str,
    comparison_data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> AnalyticsEvent:
    """Create an analytics WebSocket event."""
    return AnalyticsEvent(
        event_type=EventType.ANALYTICS_UPDATE,
        event_id=f"analytics_{metric_name}_{int(datetime.utcnow().timestamp())}",
        metric_name=metric_name,
        metric_value=metric_value,
        metric_type=metric_type,
        time_period=time_period,
        comparison_data=comparison_data,
        data=kwargs
    )