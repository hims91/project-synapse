"""
Webhook data models.

Defines the data structures used for webhook endpoints, deliveries, and events.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from uuid import UUID, uuid4


class WebhookEventType(str, Enum):
    """Webhook event types."""
    JOB_CREATED = "job.created"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    
    ARTICLE_SCRAPED = "article.scraped"
    ARTICLE_PROCESSED = "article.processed"
    
    FEED_UPDATED = "feed.updated"
    FEED_ERROR = "feed.error"
    
    ALERT_CREATED = "alert.created"
    ALERT_RESOLVED = "alert.resolved"
    
    SYSTEM_HEALTH = "system.health"
    SYSTEM_ERROR = "system.error"
    
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"


class WebhookStatus(str, Enum):
    """Webhook endpoint status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISABLED = "disabled"
    ERROR = "error"


class DeliveryStatus(str, Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    SENDING = "sending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    ABANDONED = "abandoned"


class WebhookEndpoint(BaseModel):
    """Webhook endpoint configuration."""
    id: UUID = Field(default_factory=uuid4, description="Unique endpoint identifier")
    url: HttpUrl = Field(..., description="Webhook URL")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Endpoint description")
    
    # Event filtering
    event_types: List[WebhookEventType] = Field(
        default_factory=list, 
        description="Event types to deliver to this endpoint"
    )
    event_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event filters"
    )
    
    # Security
    secret: Optional[str] = Field(None, description="Secret for signature verification")
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional headers to send"
    )
    
    # Configuration
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay_seconds: int = Field(default=60, ge=1, description="Initial retry delay")
    
    # Status
    status: WebhookStatus = Field(default=WebhookStatus.ACTIVE, description="Endpoint status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    last_success_at: Optional[datetime] = Field(None, description="Last successful delivery")
    last_failure_at: Optional[datetime] = Field(None, description="Last failed delivery")
    
    # Statistics
    total_deliveries: int = Field(default=0, description="Total delivery attempts")
    successful_deliveries: int = Field(default=0, description="Successful deliveries")
    failed_deliveries: int = Field(default=0, description="Failed deliveries")
    
    # User association
    user_id: Optional[str] = Field(None, description="Associated user ID")
    
    def update_stats(self, success: bool):
        """Update delivery statistics."""
        self.total_deliveries += 1
        self.updated_at = datetime.utcnow()
        
        if success:
            self.successful_deliveries += 1
            self.last_success_at = datetime.utcnow()
        else:
            self.failed_deliveries += 1
            self.last_failure_at = datetime.utcnow()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_deliveries == 0:
            return 0.0
        return self.successful_deliveries / self.total_deliveries
    
    def matches_event(self, event_type: WebhookEventType, event_data: Dict[str, Any]) -> bool:
        """Check if this endpoint should receive the event."""
        # Check event type filter
        if self.event_types and event_type not in self.event_types:
            return False
        
        # Check additional filters
        for filter_key, filter_value in self.event_filters.items():
            if filter_key in event_data:
                if isinstance(filter_value, list):
                    if event_data[filter_key] not in filter_value:
                        return False
                else:
                    if event_data[filter_key] != filter_value:
                        return False
        
        return True


class WebhookEvent(BaseModel):
    """Webhook event data."""
    id: UUID = Field(default_factory=uuid4, description="Unique event identifier")
    event_type: WebhookEventType = Field(..., description="Type of event")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    
    # Event data
    data: Dict[str, Any] = Field(default_factory=dict, description="Event payload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Event metadata")
    
    # Context
    user_id: Optional[str] = Field(None, description="Associated user ID")
    source: str = Field(default="synapse", description="Event source")
    version: str = Field(default="1.0", description="Event schema version")
    
    def to_webhook_payload(self) -> Dict[str, Any]:
        """Convert to webhook payload format."""
        return {
            "id": str(self.id),
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata,
            "user_id": self.user_id,
            "source": self.source,
            "version": self.version
        }


class WebhookDelivery(BaseModel):
    """Webhook delivery attempt."""
    id: UUID = Field(default_factory=uuid4, description="Unique delivery identifier")
    endpoint_id: UUID = Field(..., description="Target endpoint ID")
    event_id: UUID = Field(..., description="Event ID")
    
    # Delivery details
    status: DeliveryStatus = Field(default=DeliveryStatus.PENDING, description="Delivery status")
    attempt_count: int = Field(default=0, description="Number of attempts")
    max_attempts: int = Field(default=3, description="Maximum attempts")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    scheduled_at: datetime = Field(default_factory=datetime.utcnow, description="Scheduled delivery time")
    started_at: Optional[datetime] = Field(None, description="Delivery start time")
    completed_at: Optional[datetime] = Field(None, description="Delivery completion time")
    next_retry_at: Optional[datetime] = Field(None, description="Next retry time")
    
    # Response details
    response_status_code: Optional[int] = Field(None, description="HTTP response status code")
    response_headers: Dict[str, str] = Field(default_factory=dict, description="Response headers")
    response_body: Optional[str] = Field(None, description="Response body")
    error_message: Optional[str] = Field(None, description="Error message")
    
    # Request details
    request_headers: Dict[str, str] = Field(default_factory=dict, description="Request headers")
    request_body: Optional[str] = Field(None, description="Request body")
    request_url: Optional[str] = Field(None, description="Request URL")
    
    def is_success(self) -> bool:
        """Check if delivery was successful."""
        return (
            self.status == DeliveryStatus.SUCCESS and
            self.response_status_code is not None and
            200 <= self.response_status_code < 300
        )
    
    def should_retry(self) -> bool:
        """Check if delivery should be retried."""
        return (
            self.status in [DeliveryStatus.FAILED, DeliveryStatus.RETRYING] and
            self.attempt_count < self.max_attempts and
            self.next_retry_at is not None and
            self.next_retry_at <= datetime.utcnow()
        )
    
    def calculate_next_retry(self, base_delay_seconds: int = 60) -> datetime:
        """Calculate next retry time using exponential backoff."""
        delay = base_delay_seconds * (2 ** self.attempt_count)
        # Cap at 1 hour
        delay = min(delay, 3600)
        return datetime.utcnow() + timedelta(seconds=delay)
    
    @property
    def duration_ms(self) -> Optional[int]:
        """Get delivery duration in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None


class WebhookDeliveryAttempt(BaseModel):
    """Individual delivery attempt details."""
    attempt_number: int = Field(..., description="Attempt number")
    started_at: datetime = Field(..., description="Attempt start time")
    completed_at: Optional[datetime] = Field(None, description="Attempt completion time")
    
    # Request details
    request_url: str = Field(..., description="Request URL")
    request_method: str = Field(default="POST", description="HTTP method")
    request_headers: Dict[str, str] = Field(default_factory=dict, description="Request headers")
    request_body: Optional[str] = Field(None, description="Request body")
    
    # Response details
    response_status_code: Optional[int] = Field(None, description="Response status code")
    response_headers: Dict[str, str] = Field(default_factory=dict, description="Response headers")
    response_body: Optional[str] = Field(None, description="Response body")
    
    # Error details
    error_type: Optional[str] = Field(None, description="Error type")
    error_message: Optional[str] = Field(None, description="Error message")
    
    # Timing
    timeout_seconds: int = Field(default=30, description="Request timeout")
    duration_ms: Optional[int] = Field(None, description="Request duration in milliseconds")
    
    def is_success(self) -> bool:
        """Check if attempt was successful."""
        return (
            self.response_status_code is not None and
            200 <= self.response_status_code < 300
        )


# Event factory functions
def create_job_webhook_event(
    event_type: WebhookEventType,
    job_id: str,
    job_type: str,
    status: str,
    user_id: Optional[str] = None,
    **kwargs
) -> WebhookEvent:
    """Create a job-related webhook event."""
    return WebhookEvent(
        event_type=event_type,
        user_id=user_id,
        data={
            "job_id": job_id,
            "job_type": job_type,
            "status": status,
            **kwargs
        },
        metadata={
            "category": "job",
            "source_component": "task_dispatcher"
        }
    )


def create_article_webhook_event(
    event_type: WebhookEventType,
    article_id: str,
    url: str,
    title: str,
    user_id: Optional[str] = None,
    **kwargs
) -> WebhookEvent:
    """Create an article-related webhook event."""
    return WebhookEvent(
        event_type=event_type,
        user_id=user_id,
        data={
            "article_id": article_id,
            "url": url,
            "title": title,
            **kwargs
        },
        metadata={
            "category": "article",
            "source_component": "content_processor"
        }
    )


def create_system_webhook_event(
    event_type: WebhookEventType,
    component: str,
    status: str,
    **kwargs
) -> WebhookEvent:
    """Create a system-related webhook event."""
    return WebhookEvent(
        event_type=event_type,
        data={
            "component": component,
            "status": status,
            **kwargs
        },
        metadata={
            "category": "system",
            "source_component": component
        }
    )