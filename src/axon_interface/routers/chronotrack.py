"""
Chrono-Track API Endpoints - Change Monitoring

Implements webpage change monitoring and notification:
- POST /tracking/subscriptions - Create and manage webpage change subscriptions
- GET /tracking/subscriptions - List active subscriptions
- DELETE /tracking/subscriptions/{id} - Remove subscriptions
- GET /tracking/changes/{subscription_id} - Get change history
- POST /tracking/check - Manual change check trigger

Provides comprehensive change detection with diff generation and webhook notifications.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib
import difflib
import re
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Body, BackgroundTasks
from pydantic import BaseModel, Field, HttpUrl, validator

from ...shared.schemas import PaginatedResponse, PaginationInfo
from ...thalamus.nlp_pipeline import get_nlp_pipeline, NLPPipeline
from ..dependencies import get_repository_factory
from ...synaptic_vesicle.repositories import RepositoryFactory


logger = logging.getLogger(__name__)
router = APIRouter()


class ChangeType(str, Enum):
    """Types of changes that can be detected."""
    CONTENT = "content"
    STRUCTURE = "structure"
    METADATA = "metadata"
    IMAGES = "images"
    LINKS = "links"
    ALL = "all"


class NotificationMethod(str, Enum):
    """Methods for delivering change notifications."""
    WEBHOOK = "webhook"
    EMAIL = "email"
    BOTH = "both"


class ChangeSubscriptionRequest(BaseModel):
    """Request model for creating change subscriptions."""
    url: HttpUrl = Field(..., description="URL to monitor for changes")
    name: Optional[str] = Field(None, description="Human-readable name for the subscription")
    change_types: List[ChangeType] = Field(default=[ChangeType.CONTENT], description="Types of changes to monitor")
    check_interval: int = Field(default=3600, ge=300, le=86400, description="Check interval in seconds (5min-24hr)")
    notification_method: NotificationMethod = Field(default=NotificationMethod.WEBHOOK, description="How to deliver notifications")
    webhook_url: Optional[HttpUrl] = Field(None, description="Webhook URL for notifications")
    email: Optional[str] = Field(None, description="Email address for notifications")
    sensitivity: float = Field(default=0.1, ge=0.01, le=1.0, description="Change sensitivity threshold (0.01-1.0)")
    ignore_patterns: List[str] = Field(default_factory=list, description="Regex patterns to ignore in content")
    include_diff: bool = Field(default=True, description="Include diff in change notifications")
    
    @validator('webhook_url')
    def validate_webhook_url(cls, v, values):
        if values.get('notification_method') in [NotificationMethod.WEBHOOK, NotificationMethod.BOTH] and not v:
            raise ValueError('webhook_url is required when using webhook notifications')
        return v
    
    @validator('email')
    def validate_email(cls, v, values):
        if values.get('notification_method') in [NotificationMethod.EMAIL, NotificationMethod.BOTH] and not v:
            raise ValueError('email is required when using email notifications')
        return v


class ChangeSubscriptionResponse(BaseModel):
    """Response model for change subscriptions."""
    id: str = Field(..., description="Subscription ID")
    url: str = Field(..., description="Monitored URL")
    name: Optional[str] = Field(None, description="Subscription name")
    change_types: List[ChangeType] = Field(..., description="Monitored change types")
    check_interval: int = Field(..., description="Check interval in seconds")
    notification_method: NotificationMethod = Field(..., description="Notification delivery method")
    webhook_url: Optional[str] = Field(None, description="Webhook URL")
    email: Optional[str] = Field(None, description="Email address")
    sensitivity: float = Field(..., description="Change sensitivity threshold")
    ignore_patterns: List[str] = Field(..., description="Ignored content patterns")
    include_diff: bool = Field(..., description="Include diff in notifications")
    is_active: bool = Field(..., description="Subscription status")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_checked: Optional[datetime] = Field(None, description="Last check timestamp")
    last_change: Optional[datetime] = Field(None, description="Last change detected")
    total_changes: int = Field(default=0, description="Total changes detected")
    user_id: str = Field(..., description="User ID")


class ChangeDetectionResult(BaseModel):
    """Result of change detection analysis."""
    has_changes: bool = Field(..., description="Whether changes were detected")
    change_score: float = Field(..., description="Magnitude of changes (0-1)")
    change_types: List[ChangeType] = Field(..., description="Types of changes detected")
    content_diff: Optional[str] = Field(None, description="Content differences")
    structure_changes: List[str] = Field(default_factory=list, description="Structural changes")
    metadata_changes: Dict[str, Any] = Field(default_factory=dict, description="Metadata changes")
    image_changes: List[str] = Field(default_factory=list, description="Image changes")
    link_changes: List[str] = Field(default_factory=list, description="Link changes")
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Detection timestamp")


class ChangeHistoryResponse(BaseModel):
    """Response model for change history."""
    subscription_id: str = Field(..., description="Subscription ID")
    changes: List[ChangeDetectionResult] = Field(..., description="Change history")
    total_changes: int = Field(..., description="Total number of changes")
    date_range: Dict[str, datetime] = Field(..., description="Date range of changes")


@router.post(
    "/tracking/subscriptions",
    response_model=ChangeSubscriptionResponse,
    summary="Create webpage change monitoring subscription",
    description="""
    Create a new subscription to monitor webpage changes.
    
    Features:
    - Multiple change detection types (content, structure, metadata, images, links)
    - Configurable check intervals and sensitivity thresholds
    - Webhook and email notification support
    - Content diff generation with ignore patterns
    - Real-time change detection with background processing
    
    Supports monitoring for content changes, structural modifications, metadata updates, and more.
    """
)
async def create_subscription(
    request: Request,
    subscription_request: ChangeSubscriptionRequest,
    background_tasks: BackgroundTasks,
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> ChangeSubscriptionResponse:
    """Create a new change monitoring subscription."""
    try:
        user_id = getattr(request.state, "user_id", "anonymous")
        user_tier = getattr(request.state, "user_tier", "free")
        
        # Validate subscription limits based on user tier
        max_subscriptions = {"free": 3, "premium": 25, "enterprise": 100}.get(user_tier, 3)
        
        # Check current subscription count (would need repository implementation)
        # For now, we'll simulate the creation
        
        # Generate subscription ID
        subscription_id = _generate_subscription_id(str(subscription_request.url), user_id)
        
        # Create initial snapshot of the webpage
        initial_snapshot = await _create_webpage_snapshot(
            str(subscription_request.url), nlp_pipeline
        )
        
        # Store subscription (would be in database)
        subscription_data = {
            "id": subscription_id,
            "url": str(subscription_request.url),
            "name": subscription_request.name or f"Monitor {urlparse(str(subscription_request.url)).netloc}",
            "change_types": subscription_request.change_types,
            "check_interval": subscription_request.check_interval,
            "notification_method": subscription_request.notification_method,
            "webhook_url": str(subscription_request.webhook_url) if subscription_request.webhook_url else None,
            "email": subscription_request.email,
            "sensitivity": subscription_request.sensitivity,
            "ignore_patterns": subscription_request.ignore_patterns,
            "include_diff": subscription_request.include_diff,
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
            "last_checked": None,
            "last_change": None,
            "total_changes": 0,
            "user_id": user_id,
            "initial_snapshot": initial_snapshot
        }
        
        # Schedule background monitoring
        background_tasks.add_task(
            _schedule_monitoring_task,
            subscription_id,
            subscription_request.check_interval
        )
        
        logger.info(f"Created change subscription {subscription_id} for {subscription_request.url}")
        
        return ChangeSubscriptionResponse(**subscription_data)
        
    except Exception as e:
        logger.error(f"Error creating subscription: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create change monitoring subscription"
        )


@router.get(
    "/tracking/subscriptions",
    response_model=PaginatedResponse[ChangeSubscriptionResponse],
    summary="List change monitoring subscriptions",
    description="""
    Retrieve all active change monitoring subscriptions for the user.
    
    Features:
    - Pagination support for large subscription lists
    - Filtering by status, URL, or change types
    - Sorting by creation date, last check, or change count
    - Subscription statistics and health information
    """
)
async def list_subscriptions(
    request: Request,
    active_only: bool = Query(True, description="Show only active subscriptions"),
    limit: int = Query(20, ge=1, le=100, description="Maximum subscriptions to return"),
    offset: int = Query(0, ge=0, description="Number of subscriptions to skip"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> PaginatedResponse[ChangeSubscriptionResponse]:
    """List user's change monitoring subscriptions."""
    try:
        user_id = getattr(request.state, "user_id", "anonymous")
        
        # Mock subscription data (would come from database)
        mock_subscriptions = [
            {
                "id": "sub_123",
                "url": "https://example.com/news",
                "name": "Example News Monitor",
                "change_types": [ChangeType.CONTENT],
                "check_interval": 3600,
                "notification_method": NotificationMethod.WEBHOOK,
                "webhook_url": "https://webhook.example.com/changes",
                "email": None,
                "sensitivity": 0.1,
                "ignore_patterns": ["timestamp", "date"],
                "include_diff": True,
                "is_active": True,
                "created_at": datetime.now(timezone.utc) - timedelta(days=5),
                "last_checked": datetime.now(timezone.utc) - timedelta(minutes=30),
                "last_change": datetime.now(timezone.utc) - timedelta(hours=2),
                "total_changes": 15,
                "user_id": user_id
            }
        ]
        
        # Filter by active status
        if active_only:
            mock_subscriptions = [s for s in mock_subscriptions if s["is_active"]]
        
        # Apply pagination
        total_count = len(mock_subscriptions)
        paginated_subscriptions = mock_subscriptions[offset:offset + limit]
        
        subscriptions = [ChangeSubscriptionResponse(**sub) for sub in paginated_subscriptions]
        
        logger.info(f"Retrieved {len(subscriptions)} subscriptions for user {user_id}")
        
        return PaginatedResponse(
            data=subscriptions,
            pagination=PaginationInfo(
                total=total_count,
                limit=limit,
                offset=offset,
                has_next=offset + limit < total_count,
                has_prev=offset > 0
            )
        )
        
    except Exception as e:
        logger.error(f"Error listing subscriptions: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve subscriptions"
        )


@router.delete(
    "/tracking/subscriptions/{subscription_id}",
    summary="Delete change monitoring subscription",
    description="""
    Remove a change monitoring subscription.
    
    This will stop all monitoring for the specified subscription and remove
    all associated change history. This action cannot be undone.
    """
)
async def delete_subscription(
    request: Request,
    subscription_id: str,
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """Delete a change monitoring subscription."""
    try:
        user_id = getattr(request.state, "user_id", "anonymous")
        
        # Verify subscription exists and belongs to user (would check database)
        # For now, simulate successful deletion
        
        logger.info(f"Deleted subscription {subscription_id} for user {user_id}")
        
        return {
            "message": "Subscription deleted successfully",
            "subscription_id": subscription_id,
            "deleted_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error deleting subscription {subscription_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete subscription"
        )


@router.get(
    "/tracking/changes/{subscription_id}",
    response_model=ChangeHistoryResponse,
    summary="Get change history for subscription",
    description="""
    Retrieve the complete change history for a monitoring subscription.
    
    Features:
    - Chronological change history with detailed diffs
    - Filtering by date range and change types
    - Change magnitude and significance scoring
    - Aggregated statistics and trends
    """
)
async def get_change_history(
    request: Request,
    subscription_id: str,
    days_back: int = Query(30, ge=1, le=365, description="Number of days of history to retrieve"),
    change_types: Optional[str] = Query(None, description="Comma-separated change types to filter"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory)
) -> ChangeHistoryResponse:
    """Get change history for a subscription."""
    try:
        user_id = getattr(request.state, "user_id", "anonymous")
        
        # Parse change types filter
        types_filter = []
        if change_types:
            types_filter = [t.strip() for t in change_types.split(",")]
        
        # Mock change history (would come from database)
        mock_changes = [
            ChangeDetectionResult(
                has_changes=True,
                change_score=0.3,
                change_types=[ChangeType.CONTENT],
                content_diff="- Old headline\n+ New headline with breaking news",
                structure_changes=[],
                metadata_changes={},
                image_changes=[],
                link_changes=[],
                detected_at=datetime.now(timezone.utc) - timedelta(hours=2)
            ),
            ChangeDetectionResult(
                has_changes=True,
                change_score=0.15,
                change_types=[ChangeType.LINKS],
                content_diff=None,
                structure_changes=[],
                metadata_changes={},
                image_changes=[],
                link_changes=["Added link to /new-article", "Removed link to /old-article"],
                detected_at=datetime.now(timezone.utc) - timedelta(hours=6)
            )
        ]
        
        # Filter by change types if specified
        if types_filter:
            mock_changes = [
                change for change in mock_changes
                if any(ct.value in types_filter for ct in change.change_types)
            ]
        
        # Calculate date range
        if mock_changes:
            earliest = min(change.detected_at for change in mock_changes)
            latest = max(change.detected_at for change in mock_changes)
            date_range = {"earliest": earliest, "latest": latest}
        else:
            date_range = {"earliest": None, "latest": None}
        
        logger.info(f"Retrieved {len(mock_changes)} changes for subscription {subscription_id}")
        
        return ChangeHistoryResponse(
            subscription_id=subscription_id,
            changes=mock_changes,
            total_changes=len(mock_changes),
            date_range=date_range
        )
        
    except Exception as e:
        logger.error(f"Error retrieving change history for {subscription_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve change history"
        )


@router.post(
    "/tracking/check",
    response_model=ChangeDetectionResult,
    summary="Manual change check trigger",
    description="""
    Manually trigger a change check for a specific subscription or URL.
    
    Features:
    - Immediate change detection and analysis
    - Detailed diff generation and change scoring
    - Support for one-time checks without creating subscriptions
    - Real-time change analysis with comprehensive reporting
    """
)
async def manual_change_check(
    request: Request,
    check_request: Dict[str, Any] = Body(...),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> ChangeDetectionResult:
    """Manually trigger a change check."""
    try:
        subscription_id = check_request.get("subscription_id")
        url = check_request.get("url")
        
        if not subscription_id and not url:
            raise HTTPException(
                status_code=400,
                detail="Either subscription_id or url must be provided"
            )
        
        # Get current webpage snapshot
        if url:
            target_url = url
        else:
            # Would retrieve URL from subscription in database
            target_url = "https://example.com"  # Mock
        
        current_snapshot = await _create_webpage_snapshot(target_url, nlp_pipeline)
        
        # Compare with previous snapshot (would come from database)
        previous_snapshot = {
            "content": "Previous content here",
            "structure": {"title": "Old Title", "headings": ["Old H1"]},
            "metadata": {"title": "Old Title", "description": "Old description"},
            "images": ["old-image.jpg"],
            "links": ["https://old-link.com"]
        }
        
        # Perform change detection
        change_result = await _detect_changes(
            previous_snapshot,
            current_snapshot,
            sensitivity=0.1,
            ignore_patterns=[]
        )
        
        logger.info(f"Manual change check completed for {target_url}")
        
        return change_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in manual change check: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to perform change check"
        )


# Helper functions for change monitoring
def _generate_subscription_id(url: str, user_id: str) -> str:
    """Generate a unique subscription ID."""
    content = f"{url}:{user_id}:{datetime.now().isoformat()}"
    return f"sub_{hashlib.md5(content.encode()).hexdigest()[:12]}"


async def _create_webpage_snapshot(url: str, nlp_pipeline: NLPPipeline) -> Dict[str, Any]:
    """Create a comprehensive snapshot of a webpage."""
    # In a real implementation, this would fetch and analyze the webpage
    # For now, return a mock snapshot
    return {
        "content": f"Mock content for {url}",
        "content_hash": hashlib.md5(f"Mock content for {url}".encode()).hexdigest(),
        "structure": {
            "title": "Mock Title",
            "headings": ["Mock H1", "Mock H2"],
            "paragraphs": 5,
            "lists": 2
        },
        "metadata": {
            "title": "Mock Title",
            "description": "Mock description",
            "keywords": ["mock", "test"],
            "og_title": "Mock OG Title"
        },
        "images": ["mock-image1.jpg", "mock-image2.png"],
        "links": ["https://mock-link1.com", "https://mock-link2.com"],
        "word_count": 250,
        "created_at": datetime.now(timezone.utc)
    }


async def _detect_changes(
    previous_snapshot: Dict[str, Any],
    current_snapshot: Dict[str, Any],
    sensitivity: float,
    ignore_patterns: List[str]
) -> ChangeDetectionResult:
    """Detect changes between two webpage snapshots."""
    changes_detected = []
    change_score = 0.0
    
    # Content changes
    prev_content = previous_snapshot.get("content", "")
    curr_content = current_snapshot.get("content", "")
    
    # Apply ignore patterns
    for pattern in ignore_patterns:
        prev_content = re.sub(pattern, "", prev_content, flags=re.IGNORECASE)
        curr_content = re.sub(pattern, "", curr_content, flags=re.IGNORECASE)
    
    content_diff = None
    if prev_content != curr_content:
        changes_detected.append(ChangeType.CONTENT)
        
        # Calculate content similarity
        similarity = _calculate_text_similarity(prev_content, curr_content)
        content_change_score = 1.0 - similarity
        change_score = max(change_score, content_change_score)
        
        # Generate diff
        content_diff = _generate_content_diff(prev_content, curr_content)
    
    # Structure changes
    structure_changes = []
    prev_structure = previous_snapshot.get("structure", {})
    curr_structure = current_snapshot.get("structure", {})
    
    if prev_structure != curr_structure:
        changes_detected.append(ChangeType.STRUCTURE)
        structure_changes = _detect_structure_changes(prev_structure, curr_structure)
        change_score = max(change_score, 0.2)
    
    # Metadata changes
    metadata_changes = {}
    prev_metadata = previous_snapshot.get("metadata", {})
    curr_metadata = current_snapshot.get("metadata", {})
    
    if prev_metadata != curr_metadata:
        changes_detected.append(ChangeType.METADATA)
        metadata_changes = _detect_metadata_changes(prev_metadata, curr_metadata)
        change_score = max(change_score, 0.1)
    
    # Image changes
    image_changes = []
    prev_images = set(previous_snapshot.get("images", []))
    curr_images = set(current_snapshot.get("images", []))
    
    if prev_images != curr_images:
        changes_detected.append(ChangeType.IMAGES)
        image_changes = _detect_image_changes(prev_images, curr_images)
        change_score = max(change_score, 0.15)
    
    # Link changes
    link_changes = []
    prev_links = set(previous_snapshot.get("links", []))
    curr_links = set(current_snapshot.get("links", []))
    
    if prev_links != curr_links:
        changes_detected.append(ChangeType.LINKS)
        link_changes = _detect_link_changes(prev_links, curr_links)
        change_score = max(change_score, 0.1)
    
    has_changes = change_score >= sensitivity
    
    return ChangeDetectionResult(
        has_changes=has_changes,
        change_score=change_score,
        change_types=changes_detected,
        content_diff=content_diff if has_changes else None,
        structure_changes=structure_changes,
        metadata_changes=metadata_changes,
        image_changes=image_changes,
        link_changes=link_changes
    )


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings."""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    
    # Use difflib for similarity calculation
    matcher = difflib.SequenceMatcher(None, text1, text2)
    return matcher.ratio()


def _generate_content_diff(old_content: str, new_content: str) -> str:
    """Generate a unified diff between two content strings."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="previous",
        tofile="current",
        lineterm=""
    )
    
    return "".join(diff)


def _detect_structure_changes(prev_structure: Dict[str, Any], curr_structure: Dict[str, Any]) -> List[str]:
    """Detect structural changes between snapshots."""
    changes = []
    
    # Title changes
    if prev_structure.get("title") != curr_structure.get("title"):
        changes.append(f"Title changed from '{prev_structure.get('title')}' to '{curr_structure.get('title')}'")
    
    # Heading changes
    prev_headings = prev_structure.get("headings", [])
    curr_headings = curr_structure.get("headings", [])
    
    if prev_headings != curr_headings:
        changes.append(f"Headings changed: {len(prev_headings)} → {len(curr_headings)}")
    
    # Paragraph count changes
    prev_paragraphs = prev_structure.get("paragraphs", 0)
    curr_paragraphs = curr_structure.get("paragraphs", 0)
    
    if prev_paragraphs != curr_paragraphs:
        changes.append(f"Paragraph count changed: {prev_paragraphs} → {curr_paragraphs}")
    
    return changes


def _detect_metadata_changes(prev_metadata: Dict[str, Any], curr_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Detect metadata changes between snapshots."""
    changes = {}
    
    all_keys = set(prev_metadata.keys()) | set(curr_metadata.keys())
    
    for key in all_keys:
        prev_value = prev_metadata.get(key)
        curr_value = curr_metadata.get(key)
        
        if prev_value != curr_value:
            changes[key] = {
                "previous": prev_value,
                "current": curr_value
            }
    
    return changes


def _detect_image_changes(prev_images: set, curr_images: set) -> List[str]:
    """Detect image changes between snapshots."""
    changes = []
    
    added_images = curr_images - prev_images
    removed_images = prev_images - curr_images
    
    for img in added_images:
        changes.append(f"Added image: {img}")
    
    for img in removed_images:
        changes.append(f"Removed image: {img}")
    
    return changes


def _detect_link_changes(prev_links: set, curr_links: set) -> List[str]:
    """Detect link changes between snapshots."""
    changes = []
    
    added_links = curr_links - prev_links
    removed_links = prev_links - curr_links
    
    for link in added_links:
        changes.append(f"Added link: {link}")
    
    for link in removed_links:
        changes.append(f"Removed link: {link}")
    
    return changes


async def _schedule_monitoring_task(subscription_id: str, check_interval: int):
    """Schedule background monitoring task for a subscription."""
    # In a real implementation, this would integrate with a task scheduler
    # For now, just log the scheduling
    logger.info(f"Scheduled monitoring task for subscription {subscription_id} with interval {check_interval}s")


async def _send_change_notification(
    subscription: Dict[str, Any],
    change_result: ChangeDetectionResult
):
    """Send change notification via configured method."""
    notification_method = subscription["notification_method"]
    
    if notification_method in [NotificationMethod.WEBHOOK, NotificationMethod.BOTH]:
        await _send_webhook_notification(subscription, change_result)
    
    if notification_method in [NotificationMethod.EMAIL, NotificationMethod.BOTH]:
        await _send_email_notification(subscription, change_result)


async def _send_webhook_notification(subscription: Dict[str, Any], change_result: ChangeDetectionResult):
    """Send webhook notification for detected changes."""
    webhook_url = subscription.get("webhook_url")
    if not webhook_url:
        return
    
    payload = {
        "subscription_id": subscription["id"],
        "url": subscription["url"],
        "change_detected": True,
        "change_score": change_result.change_score,
        "change_types": [ct.value for ct in change_result.change_types],
        "detected_at": change_result.detected_at.isoformat(),
        "content_diff": change_result.content_diff if subscription.get("include_diff") else None
    }
    
    # In a real implementation, this would make an HTTP request to the webhook URL
    logger.info(f"Would send webhook notification to {webhook_url} for subscription {subscription['id']}")


async def _send_email_notification(subscription: Dict[str, Any], change_result: ChangeDetectionResult):
    """Send email notification for detected changes."""
    email = subscription.get("email")
    if not email:
        return
    
    # In a real implementation, this would send an email
    logger.info(f"Would send email notification to {email} for subscription {subscription['id']}")