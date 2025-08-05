"""
Central Cortex - Dependencies
Layer 3: Cerebral Cortex

This module provides dependency injection for FastAPI endpoints.
Manages database sessions, authentication, and component access.
"""
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from functools import lru_cache

from fastapi import Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ..shared.config import get_settings, Settings
from ..synaptic_vesicle.database import DatabaseManager, get_db_session
from ..synaptic_vesicle.repositories import RepositoryFactory
from ..signal_relay.task_dispatcher import TaskDispatcher, get_task_dispatcher as _get_task_dispatcher
from ..spinal_cord.fallback_manager import TaskQueueFallbackManager, get_fallback_manager as _get_fallback_manager

logger = logging.getLogger(__name__)

# Global component instances
_database_manager: Optional[DatabaseManager] = None
_task_dispatcher: Optional[TaskDispatcher] = None
_fallback_manager: Optional[TaskQueueFallbackManager] = None


@lru_cache()
def get_settings_cached() -> Settings:
    """Get cached settings instance."""
    return get_settings()


async def get_database_manager() -> DatabaseManager:
    """
    Get database manager instance.
    
    Returns:
        DatabaseManager instance
    """
    global _database_manager
    
    if _database_manager is None:
        _database_manager = DatabaseManager()
        await _database_manager.initialize()
    
    return _database_manager


async def get_task_dispatcher() -> TaskDispatcher:
    """
    Get task dispatcher instance.
    
    Returns:
        TaskDispatcher instance
    """
    global _task_dispatcher
    
    if _task_dispatcher is None:
        _task_dispatcher = await _get_task_dispatcher()
    
    return _task_dispatcher


async def get_fallback_manager() -> TaskQueueFallbackManager:
    """
    Get fallback manager instance.
    
    Returns:
        FallbackManager instance
    """
    global _fallback_manager
    
    if _fallback_manager is None:
        _fallback_manager = await _get_fallback_manager()
    
    return _fallback_manager


async def get_db_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """
    Database session dependency for FastAPI endpoints.
    
    Yields:
        Database session
    """
    async with get_db_session() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


def get_repository_factory(
    session: AsyncSession = Depends(get_db_session_dependency)
) -> RepositoryFactory:
    """
    Repository factory dependency.
    
    Args:
        session: Database session
        
    Returns:
        RepositoryFactory instance
    """
    return RepositoryFactory(session)


def get_current_user(request: Request) -> Dict[str, Any]:
    """
    Get current authenticated user from request state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        User information dictionary
        
    Raises:
        HTTPException: If user is not authenticated
    """
    user = getattr(request.state, "user", None)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    return user


def get_current_user_optional(request: Request) -> Optional[Dict[str, Any]]:
    """
    Get current authenticated user from request state (optional).
    
    Args:
        request: FastAPI request object
        
    Returns:
        User information dictionary or None
    """
    return getattr(request.state, "user", None)


def require_user_tier(required_tier: str):
    """
    Dependency factory for requiring specific user tiers.
    
    Args:
        required_tier: Required user tier (free, premium, enterprise)
        
    Returns:
        Dependency function
    """
    def check_user_tier(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_tier = user.get("tier", "free")
        
        # Define tier hierarchy
        tier_levels = {"free": 0, "premium": 1, "enterprise": 2}
        
        required_level = tier_levels.get(required_tier, 0)
        user_level = tier_levels.get(user_tier, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This endpoint requires {required_tier} tier or higher"
            )
        
        return user
    
    return check_user_tier


def require_active_user(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Require user to be active.
    
    Args:
        user: User information
        
    Returns:
        User information if active
        
    Raises:
        HTTPException: If user is not active
    """
    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is not active"
        )
    
    return user


async def get_system_health() -> Dict[str, Any]:
    """
    Get system health information.
    
    Returns:
        System health dictionary
    """
    try:
        health_info = {
            "status": "healthy",
            "timestamp": logger.info("Getting system health"),
            "components": {}
        }
        
        # Check database health
        try:
            db_manager = await get_database_manager()
            db_health = await db_manager.health_check()
            health_info["components"]["database"] = db_health
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            health_info["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_info["status"] = "degraded"
        
        # Check task dispatcher health
        try:
            task_dispatcher = await get_task_dispatcher()
            dispatcher_stats = task_dispatcher.get_stats()
            health_info["components"]["task_dispatcher"] = {
                "status": "healthy",
                "stats": dispatcher_stats
            }
        except Exception as e:
            logger.error(f"Task dispatcher health check failed: {e}")
            health_info["components"]["task_dispatcher"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_info["status"] = "degraded"
        
        # Check fallback manager health
        try:
            fallback_manager = await get_fallback_manager()
            fallback_stats = fallback_manager.get_stats()
            health_info["components"]["fallback_manager"] = {
                "status": "healthy",
                "stats": fallback_stats
            }
        except Exception as e:
            logger.error(f"Fallback manager health check failed: {e}")
            health_info["components"]["fallback_manager"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_info["status"] = "degraded"
        
        return health_info
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": logger.info("System health check failed")
        }


class PaginationParams:
    """Pagination parameters for list endpoints."""
    
    def __init__(
        self,
        page: int = 1,
        page_size: int = 20,
        max_page_size: int = 100
    ):
        self.page = max(1, page)
        self.page_size = min(max(1, page_size), max_page_size)
        self.offset = (self.page - 1) * self.page_size
        self.limit = self.page_size


def get_pagination_params(
    page: int = 1,
    page_size: int = 20
) -> PaginationParams:
    """
    Get pagination parameters dependency.
    
    Args:
        page: Page number (1-based)
        page_size: Number of items per page
        
    Returns:
        PaginationParams instance
    """
    return PaginationParams(page=page, page_size=page_size)


class SearchParams:
    """Search parameters for search endpoints."""
    
    def __init__(
        self,
        q: Optional[str] = None,
        category: Optional[str] = None,
        source_domain: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sort_by: str = "relevance",
        sort_order: str = "desc"
    ):
        self.query = q
        self.category = category
        self.source_domain = source_domain
        self.date_from = date_from
        self.date_to = date_to
        self.sort_by = sort_by
        self.sort_order = sort_order


def get_search_params(
    q: Optional[str] = None,
    category: Optional[str] = None,
    source_domain: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort_by: str = "relevance",
    sort_order: str = "desc"
) -> SearchParams:
    """
    Get search parameters dependency.
    
    Args:
        q: Search query
        category: Content category filter
        source_domain: Source domain filter
        date_from: Start date filter (ISO format)
        date_to: End date filter (ISO format)
        sort_by: Sort field (relevance, date, title)
        sort_order: Sort order (asc, desc)
        
    Returns:
        SearchParams instance
    """
    return SearchParams(
        q=q,
        category=category,
        source_domain=source_domain,
        date_from=date_from,
        date_to=date_to,
        sort_by=sort_by,
        sort_order=sort_order
    )