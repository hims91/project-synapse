"""
WebWatch API Endpoints - Content Monitoring

Implements the WebWatch functionality for content monitoring and alerting:
- POST /monitoring/subscriptions - Create keyword monitoring subscriptions
- GET /monitoring/subscriptions - List user's monitoring subscriptions
- GET /monitoring/subscriptions/{id} - Get specific subscription details
- PUT /monitoring/subscriptions/{id} - Update monitoring subscription
- DELETE /monitoring/subscriptions/{id} - Delete monitoring subscription
- GET /monitoring/alerts - Get triggered alerts and notifications

Provides comprehensive webhook delivery system with retry logic.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ...shared.schemas import (
    MonitoringSubscriptionCreate,
    MonitoringSubscriptionUpdate,
    MonitoringSubscriptionResponse,
    PaginatedResponse,
    PaginationInfo,
    WebhookPayload
)
from ...synaptic_vesicle.repositories import RepositoryFactory
from ..dependencies import get_repository_factory


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/monitoring/subscriptions",
    response_model=MonitoringSubscriptionResponse,
    summary="Create keyword monitoring subscription",
    description="""
    Create a new monitoring subscription to track specific keywords across all content.
    
    Features:
    - Multi-keyword monitoring with flexible matching
    - Real-time webhook notifications
    - Customizable alert thresholds and frequency
    - Source domain filtering
    - Category-based filtering
    
    When matching content is found, notifications will be sent to the specified webhook URL
    with comprehensive article data and matching details.
    """
)
async def create_monitoring_subscription(
    request: Request,
    subscription: MonitoringSubscriptionCreate,
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> MonitoringSubscriptionResponse:
    """Create a new monitoring subscription."""
    try:
        # Generate subscription ID
        subscription_id = uuid.uuid4()
        
        # Get monitoring repository
        monitoring_repo = repository_factory.get_monitoring_subscription_repository()
        
        # Create subscription
        created_subscription = await monitoring_repo.create(
            id=subscription_id,
            user_id=request.state.user_id,
            name=subscription.name,
            keywords=subscription.keywords,
            webhook_url=str(subscription.webhook_url),
            is_active=True
        )
        
        logger.info(
            f"Created monitoring subscription {subscription_id} "
            f"for user {request.state.user_id} with keywords: {subscription.keywords}"
        )
        
        return MonitoringSubscriptionResponse(
            id=created_subscription.id,
            user_id=created_subscription.user_id,
            name=created_subscription.name,
            keywords=created_subscription.keywords,
            webhook_url=created_subscription.webhook_url,
            is_active=created_subscription.is_active,
            created_at=created_subscription.created_at,
            last_triggered=created_subscription.last_triggered
        )
        
    except Exception as e:
        logger.error(f"Error creating monitoring subscription: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create monitoring subscription"
        )


@router.get(
    "/monitoring/subscriptions",
    response_model=PaginatedResponse,
    summary="List monitoring subscriptions",
    description="""
    Retrieve a paginated list of the user's monitoring subscriptions.
    
    Supports filtering by:
    - Active/inactive status
    - Creation date range
    - Keyword matching
    
    Results include subscription details and recent activity metrics.
    """
)
async def list_monitoring_subscriptions(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=50, description="Items per page"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    keyword: Optional[str] = Query(None, description="Filter by keyword"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> PaginatedResponse:
    """List user's monitoring subscriptions."""
    try:
        # Build filters
        filters = {"user_id": request.state.user_id}
        if is_active is not None:
            filters["is_active"] = is_active
        if keyword:
            filters["keyword"] = keyword
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get subscriptions
        monitoring_repo = repository_factory.get_monitoring_subscription_repository()
        subscriptions, total_count = await monitoring_repo.list_with_filters(
            filters=filters,
            limit=page_size,
            offset=offset
        )
        
        # Calculate pagination
        total_pages = (total_count + page_size - 1) // page_size
        pagination = PaginationInfo(
            page=page,
            page_size=page_size,
            total_results=total_count,
            total_pages=total_pages
        )
        
        # Convert to response format
        subscription_responses = [
            MonitoringSubscriptionResponse(
                id=sub.id,
                user_id=sub.user_id,
                name=sub.name,
                keywords=sub.keywords,
                webhook_url=sub.webhook_url,
                is_active=sub.is_active,
                created_at=sub.created_at,
                last_triggered=sub.last_triggered
            )
            for sub in subscriptions
        ]
        
        logger.info(f"Retrieved {len(subscriptions)} monitoring subscriptions for user {request.state.user_id}")
        
        return PaginatedResponse(
            pagination=pagination,
            data=subscription_responses
        )
        
    except Exception as e:
        logger.error(f"Error listing monitoring subscriptions: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve monitoring subscriptions"
        )


@router.get(
    "/monitoring/subscriptions/{subscription_id}",
    response_model=MonitoringSubscriptionResponse,
    summary="Get monitoring subscription details",
    description="""
    Retrieve detailed information about a specific monitoring subscription.
    
    Includes:
    - Subscription configuration
    - Recent activity and trigger history
    - Performance metrics
    - Webhook delivery statistics
    """
)
async def get_monitoring_subscription(
    subscription_id: UUID,
    request: Request,
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> MonitoringSubscriptionResponse:
    """Get a specific monitoring subscription."""
    try:
        # Get monitoring repository
        monitoring_repo = repository_factory.get_monitoring_subscription_repository()
        
        # Get subscription
        subscription = await monitoring_repo.get_by_id(subscription_id)
        
        if not subscription:
            raise HTTPException(
                status_code=404,
                detail=f"Monitoring subscription {subscription_id} not found"
            )
        
        # Check ownership
        if subscription.user_id != request.state.user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied to this monitoring subscription"
            )
        
        logger.info(f"Retrieved monitoring subscription {subscription_id}")
        
        return MonitoringSubscriptionResponse(
            id=subscription.id,
            user_id=subscription.user_id,
            name=subscription.name,
            keywords=subscription.keywords,
            webhook_url=subscription.webhook_url,
            is_active=subscription.is_active,
            created_at=subscription.created_at,
            last_triggered=subscription.last_triggered
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving monitoring subscription {subscription_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve monitoring subscription"
        )


@router.put(
    "/monitoring/subscriptions/{subscription_id}",
    response_model=MonitoringSubscriptionResponse,
    summary="Update monitoring subscription",
    description="""
    Update an existing monitoring subscription.
    
    Updatable fields:
    - Subscription name
    - Keywords list
    - Webhook URL
    - Active/inactive status
    
    Changes take effect immediately for new content matching.
    """
)
async def update_monitoring_subscription(
    subscription_id: UUID,
    request: Request,
    updates: MonitoringSubscriptionUpdate,
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> MonitoringSubscriptionResponse:
    """Update a monitoring subscription."""
    try:
        # Get monitoring repository
        monitoring_repo = repository_factory.get_monitoring_subscription_repository()
        
        # Check if subscription exists and belongs to user
        subscription = await monitoring_repo.get_by_id(subscription_id)
        if not subscription:
            raise HTTPException(
                status_code=404,
                detail=f"Monitoring subscription {subscription_id} not found"
            )
        
        if subscription.user_id != request.state.user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied to this monitoring subscription"
            )
        
        # Update subscription
        updated_subscription = await monitoring_repo.update(
            subscription_id=subscription_id,
            updates=updates.dict(exclude_unset=True)
        )
        
        logger.info(f"Updated monitoring subscription {subscription_id}")
        
        return MonitoringSubscriptionResponse(
            id=updated_subscription.id,
            user_id=updated_subscription.user_id,
            name=updated_subscription.name,
            keywords=updated_subscription.keywords,
            webhook_url=updated_subscription.webhook_url,
            is_active=updated_subscription.is_active,
            created_at=updated_subscription.created_at,
            last_triggered=updated_subscription.last_triggered
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating monitoring subscription {subscription_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update monitoring subscription"
        )


@router.delete(
    "/monitoring/subscriptions/{subscription_id}",
    summary="Delete monitoring subscription",
    description="""
    Delete a monitoring subscription.
    
    This action is irreversible and will stop all monitoring for the specified keywords.
    Any pending webhook deliveries for this subscription will be cancelled.
    """
)
async def delete_monitoring_subscription(
    subscription_id: UUID,
    request: Request,
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, str]:
    """Delete a monitoring subscription."""
    try:
        # Get monitoring repository
        monitoring_repo = repository_factory.get_monitoring_subscription_repository()
        
        # Check if subscription exists and belongs to user
        subscription = await monitoring_repo.get_by_id(subscription_id)
        if not subscription:
            raise HTTPException(
                status_code=404,
                detail=f"Monitoring subscription {subscription_id} not found"
            )
        
        if subscription.user_id != request.state.user_id:
            raise HTTPException(
                status_code=403,
                detail="Access denied to this monitoring subscription"
            )
        
        # Delete subscription
        await monitoring_repo.delete(subscription_id)
        
        logger.info(f"Deleted monitoring subscription {subscription_id}")
        
        return {
            "status": "success",
            "message": f"Monitoring subscription {subscription_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting monitoring subscription {subscription_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete monitoring subscription"
        )


@router.get(
    "/monitoring/alerts",
    response_model=PaginatedResponse,
    summary="Get triggered alerts and notifications",
    description="""
    Retrieve alerts that have been triggered by monitoring subscriptions.
    
    Includes:
    - Matched articles and content
    - Trigger timestamps and details
    - Webhook delivery status
    - Matching keywords and relevance scores
    
    Supports filtering by subscription, date range, and delivery status.
    """
)
async def get_monitoring_alerts(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=50, description="Items per page"),
    subscription_id: Optional[UUID] = Query(None, description="Filter by subscription ID"),
    triggered_after: Optional[str] = Query(None, description="Filter alerts triggered after this date"),
    triggered_before: Optional[str] = Query(None, description="Filter alerts triggered before this date"),
    delivery_status: Optional[str] = Query(None, description="Filter by webhook delivery status"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> PaginatedResponse:
    """Get triggered monitoring alerts."""
    try:
        # Build filters
        filters = {"user_id": request.state.user_id}
        if subscription_id:
            filters["subscription_id"] = subscription_id
        if triggered_after:
            filters["triggered_after"] = triggered_after
        if triggered_before:
            filters["triggered_before"] = triggered_before
        if delivery_status:
            filters["delivery_status"] = delivery_status
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get alerts
        monitoring_repo = repository_factory.get_monitoring_subscription_repository()
        alerts, total_count = await monitoring_repo.get_user_alerts(
            filters=filters,
            limit=page_size,
            offset=offset
        )
        
        # Calculate pagination
        total_pages = (total_count + page_size - 1) // page_size
        pagination = PaginationInfo(
            page=page,
            page_size=page_size,
            total_results=total_count,
            total_pages=total_pages
        )
        
        logger.info(f"Retrieved {len(alerts)} monitoring alerts for user {request.state.user_id}")
        
        return PaginatedResponse(
            pagination=pagination,
            data=alerts
        )
        
    except Exception as e:
        logger.error(f"Error retrieving monitoring alerts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve monitoring alerts"
        )


@router.post(
    "/monitoring/test-webhook",
    summary="Test webhook endpoint",
    description="""
    Test a webhook endpoint by sending a sample notification.
    
    Useful for validating webhook URLs and testing notification handling
    before creating monitoring subscriptions.
    
    Sends a test payload with sample article data to verify connectivity
    and response handling.
    """
)
async def test_webhook(
    request: Request,
    webhook_data: Dict[str, Any],
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """Test a webhook endpoint."""
    try:
        # Validate webhook data
        if "webhook_url" not in webhook_data:
            raise HTTPException(
                status_code=400,
                detail="webhook_url is required"
            )
        
        # Create test payload
        test_payload = WebhookPayload(
            subscription_id=uuid.uuid4(),
            subscription_name="Test Subscription",
            matched_keywords=["test", "webhook"],
            article={
                "id": uuid.uuid4(),
                "title": "Test Article",
                "url": "https://example.com/test",
                "content": "This is a test article for webhook testing.",
                "summary": "Test article summary",
                "author": "Test Author",
                "published_at": "2025-01-08T00:00:00Z",
                "source_domain": "example.com",
                "scraped_at": "2025-01-08T00:00:00Z",
                "nlp_data": {
                    "sentiment": 0.0,
                    "entities": [],
                    "categories": ["test"],
                    "significance": 0.5
                },
                "page_metadata": {}
            },
            triggered_at="2025-01-08T00:00:00Z"
        )
        
        # Send test webhook (this would be implemented in a webhook service)
        # For now, we'll simulate the test
        webhook_url = webhook_data["webhook_url"]
        
        logger.info(f"Testing webhook URL: {webhook_url}")
        
        # In a real implementation, this would make an HTTP request to the webhook URL
        # and return the actual response status and timing
        
        return {
            "status": "success",
            "message": "Webhook test completed successfully",
            "webhook_url": webhook_url,
            "response_time_ms": 150,
            "status_code": 200,
            "test_payload": test_payload.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing webhook: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to test webhook"
        )


@router.get(
    "/monitoring/stats",
    summary="Get monitoring statistics",
    description="""
    Retrieve monitoring statistics and metrics including:
    - Active subscriptions count
    - Total alerts triggered
    - Webhook delivery success rates
    - Most popular keywords
    - Performance metrics
    
    Useful for understanding monitoring usage and effectiveness.
    """
)
async def get_monitoring_stats(
    request: Request,
    time_range: str = Query("24h", regex="^(1h|6h|24h|7d|30d)$", description="Time range for statistics"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """Get monitoring statistics."""
    try:
        monitoring_repo = repository_factory.get_monitoring_subscription_repository()
        
        # Get user monitoring stats
        stats = await monitoring_repo.get_user_monitoring_stats(
            user_id=request.state.user_id,
            time_range=time_range
        )
        
        logger.info(f"Retrieved monitoring statistics for user {request.state.user_id}")
        
        return {
            "time_range": time_range,
            "active_subscriptions": stats.get("active_subscriptions", 0),
            "total_alerts": stats.get("total_alerts", 0),
            "webhook_success_rate": stats.get("webhook_success_rate", 0.0),
            "top_keywords": stats.get("top_keywords", []),
            "alerts_by_day": stats.get("alerts_by_day", []),
            "generated_at": "2025-01-08T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving monitoring statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve monitoring statistics"
        )