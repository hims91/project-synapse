"""
Shared Schemas - Pydantic Models for Validation and Serialization
Common data models used across all components of Project Synapse.

These schemas provide:
- Request/response validation
- Data serialization/deserialization
- API documentation via OpenAPI
- Type safety across the application
"""
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel, Field, validator, HttpUrl, EmailStr, constr, conint, confloat


# Enums for controlled values
class TaskStatus(str, Enum):
    """Task queue status options."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UserTier(str, Enum):
    """User subscription tiers."""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class RecipeCreatedBy(str, Enum):
    """How scraping recipes were created."""
    LEARNING = "learning"
    MANUAL = "manual"


# Base schemas
class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    class Config:
        orm_mode = True
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True


class TimestampMixin(BaseModel):
    """Mixin for models with timestamps."""
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# Article schemas
class ArticleNLPData(BaseModel):
    """NLP enrichment data for articles."""
    sentiment: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Sentiment score (-1 to 1)")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Named entities")
    categories: List[str] = Field(default_factory=list, description="Content categories")
    significance: Optional[float] = Field(None, ge=0.0, le=10.0, description="Significance score")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    language: Optional[str] = Field(None, description="Detected language")


class ArticleMetadata(BaseModel):
    """Technical metadata about webpages."""
    paywall: Optional[bool] = Field(None, description="Has paywall")
    canonical_url: Optional[str] = Field(None, description="Canonical URL")
    amp_url: Optional[str] = Field(None, description="AMP URL")
    tech_stack: List[str] = Field(default_factory=list, description="Detected technologies")
    word_count: Optional[int] = Field(None, ge=0, description="Word count")
    reading_time: Optional[int] = Field(None, ge=0, description="Estimated reading time in minutes")
    images: List[str] = Field(default_factory=list, description="Image URLs")
    videos: List[str] = Field(default_factory=list, description="Video URLs")


class ArticleBase(BaseSchema):
    """Base article schema."""
    url: HttpUrl = Field(..., description="Article URL")
    title: constr(min_length=1, max_length=1000) = Field(..., description="Article title")
    content: Optional[str] = Field(None, description="Article content")
    summary: Optional[str] = Field(None, description="Article summary")
    author: Optional[constr(max_length=255)] = Field(None, description="Article author")
    published_at: Optional[datetime] = Field(None, description="Publication timestamp")
    source_domain: constr(min_length=1, max_length=255) = Field(..., description="Source domain")


class ArticleCreate(ArticleBase):
    """Schema for creating articles."""
    nlp_data: Optional[ArticleNLPData] = Field(default_factory=ArticleNLPData, description="NLP data")
    page_metadata: Optional[ArticleMetadata] = Field(default_factory=ArticleMetadata, description="Page metadata")


class ArticleUpdate(BaseSchema):
    """Schema for updating articles."""
    title: Optional[constr(min_length=1, max_length=1000)] = None
    content: Optional[str] = None
    summary: Optional[str] = None
    author: Optional[constr(max_length=255)] = None
    published_at: Optional[datetime] = None
    nlp_data: Optional[ArticleNLPData] = None
    page_metadata: Optional[ArticleMetadata] = None


class ArticleResponse(ArticleBase):
    """Schema for article responses."""
    id: uuid.UUID = Field(..., description="Article ID")
    scraped_at: datetime = Field(..., description="Scraping timestamp")
    nlp_data: ArticleNLPData = Field(..., description="NLP enrichment data")
    page_metadata: ArticleMetadata = Field(..., description="Technical metadata")


# Scraping Recipe schemas
class ScrapingSelectors(BaseModel):
    """CSS selectors for content extraction."""
    title: str = Field(..., description="Title selector")
    content: str = Field(..., description="Content selector")
    author: Optional[str] = Field(None, description="Author selector")
    publish_date: Optional[str] = Field(None, description="Publish date selector")
    summary: Optional[str] = Field(None, description="Summary selector")


class ScrapingAction(BaseModel):
    """Action to perform during scraping."""
    type: str = Field(..., description="Action type (click, wait, scroll, etc.)")
    selector: Optional[str] = Field(None, description="CSS selector for action")
    value: Optional[str] = Field(None, description="Action value")
    timeout: Optional[int] = Field(5, ge=1, le=60, description="Timeout in seconds")


class ScrapingRecipeBase(BaseSchema):
    """Base scraping recipe schema."""
    domain: constr(min_length=1, max_length=255) = Field(..., description="Target domain")
    selectors: ScrapingSelectors = Field(..., description="CSS selectors")
    actions: List[ScrapingAction] = Field(default_factory=list, description="Scraping actions")


class ScrapingRecipeCreate(ScrapingRecipeBase):
    """Schema for creating scraping recipes."""
    created_by: RecipeCreatedBy = Field(RecipeCreatedBy.LEARNING, description="Creation method")


class ScrapingRecipeUpdate(BaseSchema):
    """Schema for updating scraping recipes."""
    selectors: Optional[ScrapingSelectors] = None
    actions: Optional[List[ScrapingAction]] = None
    success_rate: Optional[confloat(ge=0.0, le=1.0)] = None
    usage_count: Optional[conint(ge=0)] = None


class ScrapingRecipeResponse(ScrapingRecipeBase):
    """Schema for scraping recipe responses."""
    id: uuid.UUID = Field(..., description="Recipe ID")
    success_rate: Decimal = Field(..., description="Success rate (0.0-1.0)")
    usage_count: int = Field(..., description="Usage count")
    last_updated: datetime = Field(..., description="Last update timestamp")
    created_by: RecipeCreatedBy = Field(..., description="Creation method")


# Task Queue schemas
class TaskPayload(BaseModel):
    """Generic task payload."""
    url: Optional[HttpUrl] = Field(None, description="Target URL")
    priority: Optional[bool] = Field(False, description="High priority flag")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TaskQueueBase(BaseSchema):
    """Base task queue schema."""
    task_type: constr(min_length=1, max_length=100) = Field(..., description="Task type")
    payload: TaskPayload = Field(..., description="Task payload")
    priority: conint(ge=1, le=10) = Field(5, description="Priority (1=highest, 10=lowest)")
    max_retries: conint(ge=0) = Field(3, description="Maximum retry attempts")


class TaskQueueCreate(TaskQueueBase):
    """Schema for creating tasks."""
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled execution time")


class TaskQueueUpdate(BaseSchema):
    """Schema for updating tasks."""
    status: Optional[TaskStatus] = None
    priority: Optional[conint(ge=1, le=10)] = None
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class TaskQueueResponse(TaskQueueBase):
    """Schema for task queue responses."""
    id: uuid.UUID = Field(..., description="Task ID")
    status: TaskStatus = Field(..., description="Task status")
    retry_count: int = Field(..., description="Current retry count")
    scheduled_at: datetime = Field(..., description="Scheduled time")
    processed_at: Optional[datetime] = Field(None, description="Processing time")
    error_message: Optional[str] = Field(None, description="Error message")


# Monitoring Subscription schemas
class MonitoringSubscriptionBase(BaseSchema):
    """Base monitoring subscription schema."""
    name: constr(min_length=1, max_length=255) = Field(..., description="Subscription name")
    keywords: List[constr(min_length=1)] = Field(..., min_items=1, description="Keywords to monitor")
    webhook_url: HttpUrl = Field(..., description="Webhook URL for notifications")


class MonitoringSubscriptionCreate(MonitoringSubscriptionBase):
    """Schema for creating monitoring subscriptions."""
    pass


class MonitoringSubscriptionUpdate(BaseSchema):
    """Schema for updating monitoring subscriptions."""
    name: Optional[constr(min_length=1, max_length=255)] = None
    keywords: Optional[List[constr(min_length=1)]] = None
    webhook_url: Optional[HttpUrl] = None
    is_active: Optional[bool] = None


class MonitoringSubscriptionResponse(MonitoringSubscriptionBase):
    """Schema for monitoring subscription responses."""
    id: uuid.UUID = Field(..., description="Subscription ID")
    user_id: uuid.UUID = Field(..., description="User ID")
    is_active: bool = Field(..., description="Active status")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_triggered: Optional[datetime] = Field(None, description="Last trigger timestamp")


# API Usage schemas
class APIUsageBase(BaseSchema):
    """Base API usage schema."""
    endpoint: constr(min_length=1, max_length=255) = Field(..., description="API endpoint")
    method: constr(min_length=1, max_length=10) = Field(..., description="HTTP method")
    status_code: Optional[conint(ge=100, le=599)] = Field(None, description="HTTP status code")
    response_time_ms: Optional[conint(ge=0)] = Field(None, description="Response time in milliseconds")


class APIUsageCreate(APIUsageBase):
    """Schema for creating API usage records."""
    api_key_id: uuid.UUID = Field(..., description="API key ID")


class APIUsageResponse(APIUsageBase):
    """Schema for API usage responses."""
    id: uuid.UUID = Field(..., description="Usage record ID")
    api_key_id: uuid.UUID = Field(..., description="API key ID")
    timestamp: datetime = Field(..., description="Request timestamp")


# Feed schemas
class FeedBase(BaseSchema):
    """Base feed schema."""
    url: HttpUrl = Field(..., description="Feed URL")
    name: constr(min_length=1, max_length=255) = Field(..., description="Feed name")
    category: Optional[constr(max_length=100)] = Field(None, description="Feed category")
    priority: conint(ge=1, le=10) = Field(5, description="Polling priority")
    polling_interval: conint(ge=60) = Field(300, description="Polling interval in seconds")


class FeedCreate(FeedBase):
    """Schema for creating feeds."""
    pass


class FeedUpdate(BaseSchema):
    """Schema for updating feeds."""
    name: Optional[constr(min_length=1, max_length=255)] = None
    category: Optional[constr(max_length=100)] = None
    priority: Optional[conint(ge=1, le=10)] = None
    polling_interval: Optional[conint(ge=60)] = None
    is_active: Optional[bool] = None


class FeedResponse(FeedBase):
    """Schema for feed responses."""
    id: uuid.UUID = Field(..., description="Feed ID")
    is_active: bool = Field(..., description="Active status")
    last_polled: Optional[datetime] = Field(None, description="Last poll timestamp")
    last_successful_poll: Optional[datetime] = Field(None, description="Last successful poll")
    consecutive_failures: int = Field(..., description="Consecutive failure count")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Update timestamp")


# User schemas
class UserBase(BaseSchema):
    """Base user schema."""
    email: EmailStr = Field(..., description="User email")
    username: constr(min_length=3, max_length=100) = Field(..., description="Username")


class UserCreate(UserBase):
    """Schema for creating users."""
    password: constr(min_length=8) = Field(..., description="User password")
    tier: UserTier = Field(UserTier.FREE, description="User tier")


class UserUpdate(BaseSchema):
    """Schema for updating users."""
    email: Optional[EmailStr] = None
    username: Optional[constr(min_length=3, max_length=100)] = None
    tier: Optional[UserTier] = None
    api_call_limit: Optional[conint(ge=1)] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None


class UserResponse(UserBase):
    """Schema for user responses."""
    id: uuid.UUID = Field(..., description="User ID")
    api_key: str = Field(..., description="API key")
    tier: UserTier = Field(..., description="User tier")
    monthly_api_calls: int = Field(..., description="Monthly API call count")
    api_call_limit: int = Field(..., description="API call limit")
    is_active: bool = Field(..., description="Active status")
    is_verified: bool = Field(..., description="Verification status")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")


# Trends schemas
class TrendingTopic(BaseModel):
    """Trending topic data."""
    topic: str = Field(..., description="Topic name")
    velocity: confloat(ge=0.0, le=10.0) = Field(..., description="Trend velocity")
    volume: conint(ge=0) = Field(..., description="Mention volume")


class TrendingEntity(BaseModel):
    """Trending entity data."""
    entity: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type (PERSON, ORG, etc.)")
    velocity: confloat(ge=0.0, le=10.0) = Field(..., description="Trend velocity")
    volume: conint(ge=0) = Field(..., description="Mention volume")


class TrendsSummaryResponse(BaseSchema):
    """Schema for trends summary responses."""
    id: uuid.UUID = Field(..., description="Trends summary ID")
    time_window: str = Field(..., description="Time window (1h, 6h, 24h)")
    category: Optional[str] = Field(None, description="Category filter")
    trending_topics: List[TrendingTopic] = Field(..., description="Trending topics")
    trending_entities: List[TrendingEntity] = Field(..., description="Trending entities")
    generated_at: datetime = Field(..., description="Generation timestamp")


# API Response schemas
class PaginationInfo(BaseModel):
    """Pagination information."""
    page: conint(ge=1) = Field(..., description="Current page number")
    page_size: conint(ge=1, le=100) = Field(..., description="Items per page")
    total_results: conint(ge=0) = Field(..., description="Total number of results")
    total_pages: conint(ge=0) = Field(..., description="Total number of pages")


class PaginatedResponse(BaseModel):
    """Generic paginated response."""
    pagination: PaginationInfo = Field(..., description="Pagination information")
    data: List[Any] = Field(..., description="Response data")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: Dict[str, str] = Field(..., description="Error information")
    
    @validator('error')
    def validate_error(cls, v):
        required_fields = {'type', 'message'}
        if not all(field in v for field in required_fields):
            raise ValueError('Error must contain type and message fields')
        return v


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component status")


# Search schemas
class SearchQuery(BaseModel):
    """Search query parameters."""
    q: constr(min_length=1) = Field(..., description="Search query")
    limit: conint(ge=1, le=100) = Field(10, description="Maximum results")
    offset: conint(ge=0) = Field(0, description="Result offset")


class SearchResult(BaseModel):
    """Search result item."""
    score: confloat(ge=0.0, le=1.0) = Field(..., description="Relevance score")
    article: ArticleResponse = Field(..., description="Article data")


class SearchResponse(BaseModel):
    """Search response."""
    query: str = Field(..., description="Original query")
    total_hits: conint(ge=0) = Field(..., description="Total matching results")
    results: List[SearchResult] = Field(..., description="Search results")
    took_ms: conint(ge=0) = Field(..., description="Query execution time")


# Webhook schemas
class WebhookPayload(BaseModel):
    """Webhook notification payload."""
    subscription_id: uuid.UUID = Field(..., description="Subscription ID")
    subscription_name: str = Field(..., description="Subscription name")
    matched_keywords: List[str] = Field(..., description="Matched keywords")
    article: ArticleResponse = Field(..., description="Matching article")
    triggered_at: datetime = Field(..., description="Trigger timestamp")


# Job status schemas
class JobStatus(str, Enum):
    """Job status options."""
    PENDING = "pending"
    PROCESSING = "processing"
    LEARNING = "learning"
    COMPLETED = "completed"
    FAILED = "failed"


class ScrapeJobCreate(BaseModel):
    """Schema for creating scrape jobs."""
    url: HttpUrl = Field(..., description="URL to scrape")
    priority: bool = Field(False, description="High priority flag")


class ScrapeJobResponse(BaseModel):
    """Schema for scrape job responses."""
    job_id: uuid.UUID = Field(..., description="Job ID")
    status: JobStatus = Field(..., description="Job status")
    message: str = Field(..., description="Status message")
    data: Optional[ArticleResponse] = Field(None, description="Scraped data (when completed)")
    created_at: datetime = Field(..., description="Job creation time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")