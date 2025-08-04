"""
Synaptic Vesicle - Repository Pattern for Data Access
Layer 2: Signal Network

This module implements the repository pattern for clean data access abstraction.
Provides CRUD operations, query optimization, and caching mechanisms.
"""
import uuid
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Generic, TypeVar, Union
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, text, and_, or_
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import IntegrityError, NoResultFound
import structlog

from .models import (
    Article, ScrapingRecipe, TaskQueue, MonitoringSubscription,
    APIUsage, Feed, User, TrendsSummary
)
from ..shared.schemas import (
    ArticleCreate, ArticleUpdate, ScrapingRecipeCreate, ScrapingRecipeUpdate,
    TaskQueueCreate, TaskQueueUpdate, MonitoringSubscriptionCreate, MonitoringSubscriptionUpdate,
    APIUsageCreate, FeedCreate, FeedUpdate, UserCreate, UserUpdate,
    TaskStatus, UserTier
)

logger = structlog.get_logger(__name__)

# Generic type for model classes
ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType], ABC):
    """
    Base repository class with common CRUD operations.
    Provides a consistent interface for all data access operations.
    """
    
    def __init__(self, session: AsyncSession, model: type[ModelType]):
        self.session = session
        self.model = model
    
    async def create(self, obj_in: CreateSchemaType) -> ModelType:
        """Create a new record."""
        try:
            # Convert Pydantic model to dict, excluding None values
            obj_data = obj_in.model_dump(exclude_unset=True)
            db_obj = self.model(**obj_data)
            
            self.session.add(db_obj)
            await self.session.commit()
            await self.session.refresh(db_obj)
            
            logger.info(f"Created {self.model.__name__}", id=getattr(db_obj, 'id', None))
            return db_obj
            
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Integrity error creating {self.model.__name__}", error=str(e))
            raise ValueError(f"Data integrity violation: {str(e)}")
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating {self.model.__name__}", error=str(e))
            raise
    
    async def get(self, id: uuid.UUID) -> Optional[ModelType]:
        """Get a record by ID."""
        try:
            result = await self.session.execute(
                select(self.model).where(self.model.id == id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting {self.model.__name__}", id=id, error=str(e))
            raise
    
    async def get_multi(
        self, 
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None
    ) -> List[ModelType]:
        """Get multiple records with pagination and filtering."""
        try:
            query = select(self.model)
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model, field):
                        if isinstance(value, list):
                            query = query.where(getattr(self.model, field).in_(value))
                        else:
                            query = query.where(getattr(self.model, field) == value)
            
            # Apply ordering
            if order_by:
                if order_by.startswith('-'):
                    field = order_by[1:]
                    if hasattr(self.model, field):
                        query = query.order_by(getattr(self.model, field).desc())
                else:
                    if hasattr(self.model, order_by):
                        query = query.order_by(getattr(self.model, order_by))
            
            # Apply pagination
            query = query.offset(skip).limit(limit)
            
            result = await self.session.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            logger.error(f"Error getting multiple {self.model.__name__}", error=str(e))
            raise
    
    async def update(self, id: uuid.UUID, obj_in: UpdateSchemaType) -> Optional[ModelType]:
        """Update a record by ID."""
        try:
            # Get existing record
            db_obj = await self.get(id)
            if not db_obj:
                return None
            
            # Update fields
            obj_data = obj_in.model_dump(exclude_unset=True)
            for field, value in obj_data.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)
            
            await self.session.commit()
            await self.session.refresh(db_obj)
            
            logger.info(f"Updated {self.model.__name__}", id=id)
            return db_obj
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating {self.model.__name__}", id=id, error=str(e))
            raise
    
    async def delete(self, id: uuid.UUID) -> bool:
        """Delete a record by ID."""
        try:
            result = await self.session.execute(
                delete(self.model).where(self.model.id == id)
            )
            await self.session.commit()
            
            deleted = result.rowcount > 0
            if deleted:
                logger.info(f"Deleted {self.model.__name__}", id=id)
            
            return deleted
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error deleting {self.model.__name__}", id=id, error=str(e))
            raise
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count records with optional filtering."""
        try:
            query = select(func.count(self.model.id))
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model, field):
                        if isinstance(value, list):
                            query = query.where(getattr(self.model, field).in_(value))
                        else:
                            query = query.where(getattr(self.model, field) == value)
            
            result = await self.session.execute(query)
            return result.scalar()
            
        except Exception as e:
            logger.error(f"Error counting {self.model.__name__}", error=str(e))
            raise
    
    async def exists(self, id: uuid.UUID) -> bool:
        """Check if a record exists by ID."""
        try:
            result = await self.session.execute(
                select(func.count(self.model.id)).where(self.model.id == id)
            )
            return result.scalar() > 0
        except Exception as e:
            logger.error(f"Error checking existence of {self.model.__name__}", id=id, error=str(e))
            raise


class ArticleRepository(BaseRepository[Article, ArticleCreate, ArticleUpdate]):
    """Repository for Article operations with specialized queries."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Article)
    
    async def get_by_url(self, url: str) -> Optional[Article]:
        """Get article by URL."""
        try:
            result = await self.session.execute(
                select(Article).where(Article.url == url)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error getting article by URL", url=url, error=str(e))
            raise
    
    async def search_full_text(self, query: str, limit: int = 10) -> List[Article]:
        """Perform full-text search on articles."""
        try:
            # Use PostgreSQL full-text search
            search_query = select(Article).where(
                Article.search_vector.match(query)
            ).order_by(
                func.ts_rank(Article.search_vector, func.plainto_tsquery(query)).desc()
            ).limit(limit)
            
            result = await self.session.execute(search_query)
            return result.scalars().all()
        except Exception as e:
            logger.error("Error performing full-text search", query=query, error=str(e))
            raise
    
    async def get_by_domain(self, domain: str, limit: int = 100) -> List[Article]:
        """Get articles by source domain."""
        try:
            result = await self.session.execute(
                select(Article)
                .where(Article.source_domain == domain)
                .order_by(Article.scraped_at.desc())
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error("Error getting articles by domain", domain=domain, error=str(e))
            raise
    
    async def get_recent(self, hours: int = 24, limit: int = 100) -> List[Article]:
        """Get recent articles within specified hours."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            result = await self.session.execute(
                select(Article)
                .where(Article.scraped_at >= cutoff_time)
                .order_by(Article.scraped_at.desc())
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error("Error getting recent articles", hours=hours, error=str(e))
            raise
    
    async def get_by_sentiment_range(
        self, 
        min_sentiment: float = -1.0, 
        max_sentiment: float = 1.0,
        limit: int = 100
    ) -> List[Article]:
        """Get articles by sentiment range."""
        try:
            result = await self.session.execute(
                select(Article)
                .where(
                    and_(
                        Article.nlp_data['sentiment'].astext.cast(func.numeric) >= min_sentiment,
                        Article.nlp_data['sentiment'].astext.cast(func.numeric) <= max_sentiment
                    )
                )
                .order_by(Article.scraped_at.desc())
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error("Error getting articles by sentiment", error=str(e))
            raise


class ScrapingRecipeRepository(BaseRepository[ScrapingRecipe, ScrapingRecipeCreate, ScrapingRecipeUpdate]):
    """Repository for ScrapingRecipe operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, ScrapingRecipe)
    
    async def get_by_domain(self, domain: str) -> Optional[ScrapingRecipe]:
        """Get scraping recipe by domain."""
        try:
            result = await self.session.execute(
                select(ScrapingRecipe).where(ScrapingRecipe.domain == domain)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error getting recipe by domain", domain=domain, error=str(e))
            raise
    
    async def get_best_recipes(self, limit: int = 10) -> List[ScrapingRecipe]:
        """Get recipes with highest success rates."""
        try:
            result = await self.session.execute(
                select(ScrapingRecipe)
                .order_by(ScrapingRecipe.success_rate.desc())
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error("Error getting best recipes", error=str(e))
            raise
    
    async def increment_usage(self, domain: str) -> bool:
        """Increment usage count for a recipe."""
        try:
            result = await self.session.execute(
                update(ScrapingRecipe)
                .where(ScrapingRecipe.domain == domain)
                .values(usage_count=ScrapingRecipe.usage_count + 1)
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error("Error incrementing recipe usage", domain=domain, error=str(e))
            raise
    
    async def update_success_rate(self, domain: str, success_rate: float) -> bool:
        """Update success rate for a recipe."""
        try:
            result = await self.session.execute(
                update(ScrapingRecipe)
                .where(ScrapingRecipe.domain == domain)
                .values(success_rate=success_rate, last_updated=func.now())
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error("Error updating recipe success rate", domain=domain, error=str(e))
            raise


class TaskQueueRepository(BaseRepository[TaskQueue, TaskQueueCreate, TaskQueueUpdate]):
    """Repository for TaskQueue operations with priority handling."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, TaskQueue)
    
    async def get_next_task(self) -> Optional[TaskQueue]:
        """Get the next task to process (highest priority, oldest first)."""
        try:
            result = await self.session.execute(
                select(TaskQueue)
                .where(TaskQueue.status == TaskStatus.PENDING)
                .order_by(TaskQueue.priority.asc(), TaskQueue.scheduled_at.asc())
                .limit(1)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error getting next task", error=str(e))
            raise
    
    async def get_by_status(self, status: TaskStatus, limit: int = 100) -> List[TaskQueue]:
        """Get tasks by status."""
        try:
            result = await self.session.execute(
                select(TaskQueue)
                .where(TaskQueue.status == status)
                .order_by(TaskQueue.scheduled_at.desc())
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error("Error getting tasks by status", status=status, error=str(e))
            raise
    
    async def mark_as_processing(self, task_id: uuid.UUID) -> bool:
        """Mark a task as processing."""
        try:
            result = await self.session.execute(
                update(TaskQueue)
                .where(TaskQueue.id == task_id)
                .values(status=TaskStatus.PROCESSING, processed_at=func.now())
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error("Error marking task as processing", task_id=task_id, error=str(e))
            raise
    
    async def mark_as_completed(self, task_id: uuid.UUID) -> bool:
        """Mark a task as completed."""
        try:
            result = await self.session.execute(
                update(TaskQueue)
                .where(TaskQueue.id == task_id)
                .values(status=TaskStatus.COMPLETED)
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error("Error marking task as completed", task_id=task_id, error=str(e))
            raise
    
    async def mark_as_failed(self, task_id: uuid.UUID, error_message: str) -> bool:
        """Mark a task as failed with error message."""
        try:
            result = await self.session.execute(
                update(TaskQueue)
                .where(TaskQueue.id == task_id)
                .values(
                    status=TaskStatus.FAILED,
                    error_message=error_message,
                    retry_count=TaskQueue.retry_count + 1
                )
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error("Error marking task as failed", task_id=task_id, error=str(e))
            raise
    
    async def get_failed_tasks_for_retry(self) -> List[TaskQueue]:
        """Get failed tasks that can be retried."""
        try:
            result = await self.session.execute(
                select(TaskQueue)
                .where(
                    and_(
                        TaskQueue.status == TaskStatus.FAILED,
                        TaskQueue.retry_count < TaskQueue.max_retries
                    )
                )
                .order_by(TaskQueue.scheduled_at.asc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error("Error getting failed tasks for retry", error=str(e))
            raise


class MonitoringSubscriptionRepository(BaseRepository[MonitoringSubscription, MonitoringSubscriptionCreate, MonitoringSubscriptionUpdate]):
    """Repository for MonitoringSubscription operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, MonitoringSubscription)
    
    async def get_by_user(self, user_id: uuid.UUID) -> List[MonitoringSubscription]:
        """Get all subscriptions for a user."""
        try:
            result = await self.session.execute(
                select(MonitoringSubscription)
                .where(MonitoringSubscription.user_id == user_id)
                .order_by(MonitoringSubscription.created_at.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error("Error getting subscriptions by user", user_id=user_id, error=str(e))
            raise
    
    async def get_active_subscriptions(self) -> List[MonitoringSubscription]:
        """Get all active subscriptions."""
        try:
            result = await self.session.execute(
                select(MonitoringSubscription)
                .where(MonitoringSubscription.is_active == True)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error("Error getting active subscriptions", error=str(e))
            raise
    
    async def find_matching_subscriptions(self, text: str) -> List[MonitoringSubscription]:
        """Find subscriptions with keywords that match the given text."""
        try:
            # Use PostgreSQL array operations to find matching keywords
            result = await self.session.execute(
                select(MonitoringSubscription)
                .where(
                    and_(
                        MonitoringSubscription.is_active == True,
                        func.array_to_string(MonitoringSubscription.keywords, ' ').ilike(f'%{text}%')
                    )
                )
            )
            return result.scalars().all()
        except Exception as e:
            logger.error("Error finding matching subscriptions", text=text, error=str(e))
            raise
    
    async def update_last_triggered(self, subscription_id: uuid.UUID) -> bool:
        """Update the last triggered timestamp."""
        try:
            result = await self.session.execute(
                update(MonitoringSubscription)
                .where(MonitoringSubscription.id == subscription_id)
                .values(last_triggered=func.now())
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error("Error updating last triggered", subscription_id=subscription_id, error=str(e))
            raise


class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    """Repository for User operations with authentication support."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, User)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            result = await self.session.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error getting user by email", email=email, error=str(e))
            raise
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            result = await self.session.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error getting user by username", username=username, error=str(e))
            raise
    
    async def get_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        try:
            result = await self.session.execute(
                select(User).where(User.api_key == api_key)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error getting user by API key", error=str(e))
            raise
    
    async def increment_api_calls(self, user_id: uuid.UUID) -> bool:
        """Increment monthly API call count."""
        try:
            result = await self.session.execute(
                update(User)
                .where(User.id == user_id)
                .values(monthly_api_calls=User.monthly_api_calls + 1)
            )
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error("Error incrementing API calls", user_id=user_id, error=str(e))
            raise
    
    async def reset_monthly_api_calls(self) -> int:
        """Reset monthly API calls for all users (run monthly)."""
        try:
            result = await self.session.execute(
                update(User).values(monthly_api_calls=0)
            )
            await self.session.commit()
            return result.rowcount
        except Exception as e:
            await self.session.rollback()
            logger.error("Error resetting monthly API calls", error=str(e))
            raise
    
    async def get_by_tier(self, tier: UserTier) -> List[User]:
        """Get users by tier."""
        try:
            result = await self.session.execute(
                select(User).where(User.tier == tier)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error("Error getting users by tier", tier=tier, error=str(e))
            raise


class APIUsageRepository(BaseRepository[APIUsage, APIUsageCreate, None]):
    """Repository for API usage tracking."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, APIUsage)
    
    async def get_usage_stats(
        self, 
        api_key_id: uuid.UUID, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get usage statistics for an API key within date range."""
        try:
            # Total requests
            total_result = await self.session.execute(
                select(func.count(APIUsage.id))
                .where(
                    and_(
                        APIUsage.api_key_id == api_key_id,
                        APIUsage.timestamp >= start_date,
                        APIUsage.timestamp <= end_date
                    )
                )
            )
            total_requests = total_result.scalar()
            
            # Average response time
            avg_result = await self.session.execute(
                select(func.avg(APIUsage.response_time_ms))
                .where(
                    and_(
                        APIUsage.api_key_id == api_key_id,
                        APIUsage.timestamp >= start_date,
                        APIUsage.timestamp <= end_date
                    )
                )
            )
            avg_response_time = avg_result.scalar() or 0
            
            # Error rate
            error_result = await self.session.execute(
                select(func.count(APIUsage.id))
                .where(
                    and_(
                        APIUsage.api_key_id == api_key_id,
                        APIUsage.timestamp >= start_date,
                        APIUsage.timestamp <= end_date,
                        APIUsage.status_code >= 400
                    )
                )
            )
            error_count = error_result.scalar()
            error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "total_requests": total_requests,
                "avg_response_time_ms": round(avg_response_time, 2),
                "error_count": error_count,
                "error_rate_percent": round(error_rate, 2)
            }
            
        except Exception as e:
            logger.error("Error getting usage stats", api_key_id=api_key_id, error=str(e))
            raise
    
    async def get_endpoint_stats(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get usage statistics by endpoint."""
        try:
            result = await self.session.execute(
                select(
                    APIUsage.endpoint,
                    func.count(APIUsage.id).label('request_count'),
                    func.avg(APIUsage.response_time_ms).label('avg_response_time')
                )
                .where(
                    and_(
                        APIUsage.timestamp >= start_date,
                        APIUsage.timestamp <= end_date
                    )
                )
                .group_by(APIUsage.endpoint)
                .order_by(func.count(APIUsage.id).desc())
            )
            
            return [
                {
                    "endpoint": row.endpoint,
                    "request_count": row.request_count,
                    "avg_response_time_ms": round(row.avg_response_time or 0, 2)
                }
                for row in result
            ]
            
        except Exception as e:
            logger.error("Error getting endpoint stats", error=str(e))
            raise


class FeedRepository(BaseRepository[Feed, FeedCreate, FeedUpdate]):
    """Repository for Feed operations with polling management."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Feed)
    
    async def get_active_feeds(self) -> List[Feed]:
        """Get all active feeds."""
        try:
            result = await self.session.execute(
                select(Feed)
                .where(Feed.is_active == True)
                .order_by(Feed.priority.asc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error("Error getting active feeds", error=str(e))
            raise
    
    async def get_feeds_due_for_polling(self) -> List[Feed]:
        """Get feeds that are due for polling."""
        try:
            current_time = datetime.utcnow()
            result = await self.session.execute(
                select(Feed)
                .where(
                    and_(
                        Feed.is_active == True,
                        or_(
                            Feed.last_polled.is_(None),
                            Feed.last_polled + func.make_interval(0, 0, 0, 0, 0, 0, Feed.polling_interval) <= current_time
                        )
                    )
                )
                .order_by(Feed.priority.asc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error("Error getting feeds due for polling", error=str(e))
            raise
    
    async def update_poll_status(self, feed_id: uuid.UUID, success: bool) -> bool:
        """Update feed polling status."""
        try:
            if success:
                result = await self.session.execute(
                    update(Feed)
                    .where(Feed.id == feed_id)
                    .values(
                        last_polled=func.now(),
                        last_successful_poll=func.now(),
                        consecutive_failures=0
                    )
                )
            else:
                result = await self.session.execute(
                    update(Feed)
                    .where(Feed.id == feed_id)
                    .values(
                        last_polled=func.now(),
                        consecutive_failures=Feed.consecutive_failures + 1
                    )
                )
            
            await self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self.session.rollback()
            logger.error("Error updating poll status", feed_id=feed_id, error=str(e))
            raise


class TrendsSummaryRepository(BaseRepository[TrendsSummary, None, None]):
    """Repository for TrendsSummary operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, TrendsSummary)
    
    async def get_latest_trends(
        self, 
        time_window: str, 
        category: Optional[str] = None
    ) -> Optional[TrendsSummary]:
        """Get the latest trends for a time window and category."""
        try:
            query = select(TrendsSummary).where(
                TrendsSummary.time_window == time_window
            )
            
            if category:
                query = query.where(TrendsSummary.category == category)
            
            query = query.order_by(TrendsSummary.generated_at.desc()).limit(1)
            
            result = await self.session.execute(query)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error getting latest trends", time_window=time_window, category=category, error=str(e))
            raise
    
    async def cleanup_old_trends(self, days_to_keep: int = 7) -> int:
        """Clean up old trend summaries."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            result = await self.session.execute(
                delete(TrendsSummary)
                .where(TrendsSummary.generated_at < cutoff_date)
            )
            await self.session.commit()
            return result.rowcount
        except Exception as e:
            await self.session.rollback()
            logger.error("Error cleaning up old trends", error=str(e))
            raise


# Repository factory for dependency injection
class RepositoryFactory:
    """Factory for creating repository instances."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    @property
    def articles(self) -> ArticleRepository:
        return ArticleRepository(self.session)
    
    @property
    def scraping_recipes(self) -> ScrapingRecipeRepository:
        return ScrapingRecipeRepository(self.session)
    
    @property
    def task_queue(self) -> TaskQueueRepository:
        return TaskQueueRepository(self.session)
    
    @property
    def monitoring_subscriptions(self) -> MonitoringSubscriptionRepository:
        return MonitoringSubscriptionRepository(self.session)
    
    @property
    def users(self) -> UserRepository:
        return UserRepository(self.session)
    
    @property
    def api_usage(self) -> APIUsageRepository:
        return APIUsageRepository(self.session)
    
    @property
    def feeds(self) -> FeedRepository:
        return FeedRepository(self.session)
    
    @property
    def trends_summary(self) -> TrendsSummaryRepository:
        return TrendsSummaryRepository(self.session)


# Dependency function for FastAPI
async def get_repository_factory(session: AsyncSession) -> RepositoryFactory:
    """
    Get repository factory for dependency injection.
    
    Usage in FastAPI:
        @app.get("/articles")
        async def get_articles(repos: RepositoryFactory = Depends(get_repository_factory)):
            articles = await repos.articles.get_multi()
            return articles
    """
    return RepositoryFactory(session)