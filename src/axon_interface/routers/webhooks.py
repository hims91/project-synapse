"""
Webhook API endpoints.

Provides REST API for managing webhook endpoints, testing deliveries,
and viewing delivery history.
"""

import logging
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl

from ..webhooks.models import (
    WebhookEndpoint, WebhookEvent, WebhookEventType, WebhookStatus,
    create_job_webhook_event, create_article_webhook_event, create_system_webhook_event
)
from ..webhooks.delivery import get_webhook_event_bus, get_webhook_delivery_service
from ..webhooks.validation import WebhookValidator

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response models
class CreateWebhookRequest(BaseModel):
    """Request model for creating webhook endpoint."""
    url: HttpUrl
    name: str
    description: Optional[str] = None
    event_types: List[WebhookEventType] = []
    event_filters: dict = {}
    headers: dict = {}
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 60
    secret: Optional[str] = None


class UpdateWebhookRequest(BaseModel):
    """Request model for updating webhook endpoint."""
    name: Optional[str] = None
    description: Optional[str] = None
    event_types: Optional[List[WebhookEventType]] = None
    event_filters: Optional[dict] = None
    headers: Optional[dict] = None
    timeout_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    retry_delay_seconds: Optional[int] = None
    status: Optional[WebhookStatus] = None


class WebhookResponse(BaseModel):
    """Response model for webhook endpoint."""
    id: UUID
    url: str
    name: str
    description: Optional[str]
    event_types: List[WebhookEventType]
    event_filters: dict
    status: WebhookStatus
    created_at: datetime
    updated_at: datetime
    last_success_at: Optional[datetime]
    last_failure_at: Optional[datetime]
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    success_rate: float


class TestWebhookRequest(BaseModel):
    """Request model for testing webhook endpoint."""
    event_type: WebhookEventType = WebhookEventType.SYSTEM_HEALTH
    test_data: dict = {}


class TestWebhookResponse(BaseModel):
    """Response model for webhook test."""
    success: bool
    message: str
    duration_ms: Optional[int] = None


class WebhookStatsResponse(BaseModel):
    """Response model for webhook statistics."""
    total_endpoints: int
    active_endpoints: int
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    average_delivery_time_ms: float
    active_deliveries: int
    queue_size: int


@router.post("/", response_model=WebhookResponse, status_code=status.HTTP_201_CREATED)
async def create_webhook(
    request: CreateWebhookRequest,
    event_bus = Depends(get_webhook_event_bus)
):
    """Create a new webhook endpoint."""
    try:
        # Create webhook endpoint
        endpoint = WebhookEndpoint(
            url=request.url,
            name=request.name,
            description=request.description,
            event_types=request.event_types,
            event_filters=request.event_filters,
            headers=request.headers,
            timeout_seconds=request.timeout_seconds,
            max_retries=request.max_retries,
            retry_delay_seconds=request.retry_delay_seconds,
            secret=request.secret,
            user_id="anonymous"  # TODO: Get from auth context
        )
        
        # Register endpoint
        success = await event_bus.register_endpoint(endpoint)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to register webhook endpoint"
            )
        
        return WebhookResponse(
            id=endpoint.id,
            url=str(endpoint.url),
            name=endpoint.name,
            description=endpoint.description,
            event_types=endpoint.event_types,
            event_filters=endpoint.event_filters,
            status=endpoint.status,
            created_at=endpoint.created_at,
            updated_at=endpoint.updated_at,
            last_success_at=endpoint.last_success_at,
            last_failure_at=endpoint.last_failure_at,
            total_deliveries=endpoint.total_deliveries,
            successful_deliveries=endpoint.successful_deliveries,
            failed_deliveries=endpoint.failed_deliveries,
            success_rate=endpoint.success_rate
        )
        
    except Exception as e:
        logger.error(f"Error creating webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create webhook endpoint"
        )


@router.get("/", response_model=List[WebhookResponse])
async def list_webhooks(
    status_filter: Optional[WebhookStatus] = Query(None, alias="status"),
    event_bus = Depends(get_webhook_event_bus)
):
    """List webhook endpoints."""
    try:
        # TODO: Get user_id from auth context
        user_id = "anonymous"
        
        endpoints = event_bus.get_endpoints(user_id)
        
        # Apply status filter
        if status_filter:
            endpoints = [ep for ep in endpoints if ep.status == status_filter]
        
        return [
            WebhookResponse(
                id=endpoint.id,
                url=str(endpoint.url),
                name=endpoint.name,
                description=endpoint.description,
                event_types=endpoint.event_types,
                event_filters=endpoint.event_filters,
                status=endpoint.status,
                created_at=endpoint.created_at,
                updated_at=endpoint.updated_at,
                last_success_at=endpoint.last_success_at,
                last_failure_at=endpoint.last_failure_at,
                total_deliveries=endpoint.total_deliveries,
                successful_deliveries=endpoint.successful_deliveries,
                failed_deliveries=endpoint.failed_deliveries,
                success_rate=endpoint.success_rate
            )
            for endpoint in endpoints
        ]
        
    except Exception as e:
        logger.error(f"Error listing webhooks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list webhook endpoints"
        )


@router.get("/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(
    webhook_id: UUID,
    event_bus = Depends(get_webhook_event_bus)
):
    """Get a specific webhook endpoint."""
    endpoint = event_bus.get_endpoint(webhook_id)
    
    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook endpoint not found"
        )
    
    return WebhookResponse(
        id=endpoint.id,
        url=str(endpoint.url),
        name=endpoint.name,
        description=endpoint.description,
        event_types=endpoint.event_types,
        event_filters=endpoint.event_filters,
        status=endpoint.status,
        created_at=endpoint.created_at,
        updated_at=endpoint.updated_at,
        last_success_at=endpoint.last_success_at,
        last_failure_at=endpoint.last_failure_at,
        total_deliveries=endpoint.total_deliveries,
        successful_deliveries=endpoint.successful_deliveries,
        failed_deliveries=endpoint.failed_deliveries,
        success_rate=endpoint.success_rate
    )


@router.put("/{webhook_id}", response_model=WebhookResponse)
async def update_webhook(
    webhook_id: UUID,
    request: UpdateWebhookRequest,
    event_bus = Depends(get_webhook_event_bus)
):
    """Update a webhook endpoint."""
    endpoint = event_bus.get_endpoint(webhook_id)
    
    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook endpoint not found"
        )
    
    try:
        # Update fields
        if request.name is not None:
            endpoint.name = request.name
        if request.description is not None:
            endpoint.description = request.description
        if request.event_types is not None:
            endpoint.event_types = request.event_types
        if request.event_filters is not None:
            endpoint.event_filters = request.event_filters
        if request.headers is not None:
            endpoint.headers = request.headers
        if request.timeout_seconds is not None:
            endpoint.timeout_seconds = request.timeout_seconds
        if request.max_retries is not None:
            endpoint.max_retries = request.max_retries
        if request.retry_delay_seconds is not None:
            endpoint.retry_delay_seconds = request.retry_delay_seconds
        if request.status is not None:
            endpoint.status = request.status
        
        endpoint.updated_at = datetime.utcnow()
        
        # Update endpoint
        success = await event_bus.update_endpoint(endpoint)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update webhook endpoint"
            )
        
        return WebhookResponse(
            id=endpoint.id,
            url=str(endpoint.url),
            name=endpoint.name,
            description=endpoint.description,
            event_types=endpoint.event_types,
            event_filters=endpoint.event_filters,
            status=endpoint.status,
            created_at=endpoint.created_at,
            updated_at=endpoint.updated_at,
            last_success_at=endpoint.last_success_at,
            last_failure_at=endpoint.last_failure_at,
            total_deliveries=endpoint.total_deliveries,
            successful_deliveries=endpoint.successful_deliveries,
            failed_deliveries=endpoint.failed_deliveries,
            success_rate=endpoint.success_rate
        )
        
    except Exception as e:
        logger.error(f"Error updating webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update webhook endpoint"
        )


@router.delete("/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_webhook(
    webhook_id: UUID,
    event_bus = Depends(get_webhook_event_bus)
):
    """Delete a webhook endpoint."""
    success = await event_bus.unregister_endpoint(webhook_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook endpoint not found"
        )


@router.post("/{webhook_id}/test", response_model=TestWebhookResponse)
async def test_webhook(
    webhook_id: UUID,
    request: TestWebhookRequest,
    event_bus = Depends(get_webhook_event_bus),
    delivery_service = Depends(get_webhook_delivery_service)
):
    """Test a webhook endpoint."""
    endpoint = event_bus.get_endpoint(webhook_id)
    
    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook endpoint not found"
        )
    
    try:
        start_time = datetime.utcnow()
        success, message = await delivery_service.test_endpoint(endpoint)
        end_time = datetime.utcnow()
        
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        return TestWebhookResponse(
            success=success,
            message=message,
            duration_ms=duration_ms
        )
        
    except Exception as e:
        logger.error(f"Error testing webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test webhook endpoint"
        )


@router.get("/{webhook_id}/deliveries")
async def get_webhook_deliveries(
    webhook_id: UUID,
    limit: int = Query(50, ge=1, le=100),
    event_bus = Depends(get_webhook_event_bus)
):
    """Get delivery history for a webhook endpoint."""
    endpoint = event_bus.get_endpoint(webhook_id)
    
    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook endpoint not found"
        )
    
    # TODO: Implement delivery history retrieval from storage
    # For now, return empty list
    return {
        "webhook_id": webhook_id,
        "deliveries": [],
        "total": 0,
        "limit": limit
    }


@router.get("/events/types")
async def get_event_types():
    """Get available webhook event types."""
    return {
        "event_types": [
            {
                "name": event_type.value,
                "description": _get_event_type_description(event_type)
            }
            for event_type in WebhookEventType
        ]
    }


@router.post("/events/simulate")
async def simulate_webhook_event(
    event_type: WebhookEventType,
    test_data: dict = {},
    event_bus = Depends(get_webhook_event_bus)
):
    """Simulate a webhook event for testing purposes."""
    try:
        # Create test event based on type
        if event_type in [WebhookEventType.JOB_CREATED, WebhookEventType.JOB_STARTED, 
                         WebhookEventType.JOB_COMPLETED, WebhookEventType.JOB_FAILED]:
            event = create_job_webhook_event(
                event_type=event_type,
                job_id=test_data.get("job_id", "test-job-123"),
                job_type=test_data.get("job_type", "scrape"),
                status=test_data.get("status", "completed"),
                user_id="anonymous",
                **test_data
            )
        elif event_type in [WebhookEventType.ARTICLE_SCRAPED, WebhookEventType.ARTICLE_PROCESSED]:
            event = create_article_webhook_event(
                event_type=event_type,
                article_id=test_data.get("article_id", "test-article-123"),
                url=test_data.get("url", "https://example.com/test"),
                title=test_data.get("title", "Test Article"),
                user_id="anonymous",
                **test_data
            )
        else:
            event = create_system_webhook_event(
                event_type=event_type,
                component=test_data.get("component", "test_system"),
                status=test_data.get("status", "healthy"),
                **test_data
            )
        
        # Publish event
        deliveries = await event_bus.publish_event(event)
        
        return {
            "event_id": str(event.id),
            "event_type": event.event_type,
            "deliveries_created": len(deliveries),
            "delivery_ids": [str(d.id) for d in deliveries]
        }
        
    except Exception as e:
        logger.error(f"Error simulating webhook event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to simulate webhook event"
        )


@router.get("/stats", response_model=WebhookStatsResponse)
async def get_webhook_stats(
    event_bus = Depends(get_webhook_event_bus),
    delivery_service = Depends(get_webhook_delivery_service)
):
    """Get webhook system statistics."""
    try:
        endpoints = event_bus.get_endpoints()
        active_endpoints = [ep for ep in endpoints if ep.status == WebhookStatus.ACTIVE]
        
        delivery_stats = delivery_service.get_stats()
        
        return WebhookStatsResponse(
            total_endpoints=len(endpoints),
            active_endpoints=len(active_endpoints),
            total_deliveries=delivery_stats['total_deliveries'],
            successful_deliveries=delivery_stats['successful_deliveries'],
            failed_deliveries=delivery_stats['failed_deliveries'],
            average_delivery_time_ms=delivery_stats['average_delivery_time_ms'],
            active_deliveries=delivery_stats['active_deliveries'],
            queue_size=delivery_stats['queue_size']
        )
        
    except Exception as e:
        logger.error(f"Error getting webhook stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get webhook statistics"
        )


def _get_event_type_description(event_type: WebhookEventType) -> str:
    """Get description for webhook event type."""
    descriptions = {
        WebhookEventType.JOB_CREATED: "Triggered when a new scraping job is created",
        WebhookEventType.JOB_STARTED: "Triggered when a scraping job starts processing",
        WebhookEventType.JOB_COMPLETED: "Triggered when a scraping job completes successfully",
        WebhookEventType.JOB_FAILED: "Triggered when a scraping job fails",
        WebhookEventType.ARTICLE_SCRAPED: "Triggered when an article is successfully scraped",
        WebhookEventType.ARTICLE_PROCESSED: "Triggered when an article is processed and analyzed",
        WebhookEventType.FEED_UPDATED: "Triggered when a content feed is updated",
        WebhookEventType.FEED_ERROR: "Triggered when a feed update fails",
        WebhookEventType.ALERT_CREATED: "Triggered when a monitoring alert is created",
        WebhookEventType.ALERT_RESOLVED: "Triggered when a monitoring alert is resolved",
        WebhookEventType.SYSTEM_HEALTH: "Triggered for system health updates",
        WebhookEventType.SYSTEM_ERROR: "Triggered when system errors occur",
        WebhookEventType.USER_CREATED: "Triggered when a new user is created",
        WebhookEventType.USER_UPDATED: "Triggered when user information is updated"
    }
    
    return descriptions.get(event_type, "No description available")


@router.get("/test-page", response_class=HTMLResponse)
async def webhook_test_page():
    """Serve webhook test page."""
    try:
        with open("src/axon_interface/templates/webhook_test.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Webhook test page not found"
        )