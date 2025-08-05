"""
Central Cortex - Authentication Router
Layer 3: Cerebral Cortex

This module implements authentication and API key management endpoints.
Provides user registration, API key generation, and tier management.
"""
import logging
import secrets
import hashlib
from typing import Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import (
    get_db_session_dependency, get_repository_factory, get_current_user,
    require_user_tier, require_active_user
)
from ...shared.schemas import (
    UserCreate, UserResponse, UserUpdate, APIUsageResponse,
    PaginatedResponse, PaginationInfo
)
from ...synaptic_vesicle.repositories import RepositoryFactory

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> UserResponse:
    """
    Register a new user account.
    
    Args:
        user_data: User registration data
        repo_factory: Repository factory
        
    Returns:
        Created user information
        
    Raises:
        HTTPException: If registration fails
    """
    try:
        user_repo = repo_factory.get_user_repository()
        
        # Check if user already exists
        existing_user = await user_repo.get_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists"
            )
        
        existing_username = await user_repo.get_by_username(user_data.username)
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already taken"
            )
        
        # Generate API key
        api_key = _generate_api_key()
        
        # Hash password
        hashed_password = _hash_password(user_data.password)
        
        # Create user with hashed password and API key
        user_create_data = user_data.dict()
        user_create_data['password'] = hashed_password
        user_create_data['api_key'] = api_key
        
        # Create user
        user = await user_repo.create(UserCreate(**user_create_data))
        
        logger.info(f"User registered successfully: {user.email}")
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.get("/profile", response_model=UserResponse)
async def get_user_profile(
    current_user: Dict[str, Any] = Depends(require_active_user),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> UserResponse:
    """
    Get current user profile.
    
    Args:
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        User profile information
    """
    try:
        user_repo = repo_factory.get_user_repository()
        user = await user_repo.get_by_id(current_user["user_id"])
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profile"
        )


@router.put("/profile", response_model=UserResponse)
async def update_user_profile(
    user_update: UserUpdate,
    current_user: Dict[str, Any] = Depends(require_active_user),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> UserResponse:
    """
    Update current user profile.
    
    Args:
        user_update: User update data
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        Updated user information
    """
    try:
        user_repo = repo_factory.get_user_repository()
        
        # Check if email is being changed and is available
        if user_update.email:
            existing_user = await user_repo.get_by_email(user_update.email)
            if existing_user and existing_user.id != current_user["user_id"]:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already in use"
                )
        
        # Check if username is being changed and is available
        if user_update.username:
            existing_user = await user_repo.get_by_username(user_update.username)
            if existing_user and existing_user.id != current_user["user_id"]:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Username already taken"
                )
        
        # Update user
        user = await user_repo.update(current_user["user_id"], user_update)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User profile updated: {user.email}")
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.post("/regenerate-api-key", response_model=Dict[str, str])
async def regenerate_api_key(
    current_user: Dict[str, Any] = Depends(require_active_user),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, str]:
    """
    Regenerate API key for current user.
    
    Args:
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        New API key
    """
    try:
        user_repo = repo_factory.get_user_repository()
        
        # Generate new API key
        new_api_key = _generate_api_key()
        
        # Update user with new API key
        user_update = UserUpdate(api_key=new_api_key)
        user = await user_repo.update(current_user["user_id"], user_update)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"API key regenerated for user: {user.email}")
        
        return {"api_key": new_api_key}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to regenerate API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to regenerate API key"
        )


@router.get("/usage", response_model=PaginatedResponse)
async def get_api_usage(
    page: int = 1,
    page_size: int = 20,
    current_user: Dict[str, Any] = Depends(require_active_user),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> PaginatedResponse:
    """
    Get API usage history for current user.
    
    Args:
        page: Page number
        page_size: Items per page
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        Paginated API usage data
    """
    try:
        api_usage_repo = repo_factory.get_api_usage_repository()
        
        # Get user's API usage
        offset = (page - 1) * page_size
        usage_data, total_count = await api_usage_repo.get_by_user_id(
            current_user["user_id"],
            limit=page_size,
            offset=offset
        )
        
        # Calculate pagination info
        total_pages = (total_count + page_size - 1) // page_size
        
        pagination_info = PaginationInfo(
            page=page,
            page_size=page_size,
            total_results=total_count,
            total_pages=total_pages
        )
        
        return PaginatedResponse(
            pagination=pagination_info,
            data=usage_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get API usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API usage"
        )


@router.get("/usage/summary")
async def get_usage_summary(
    current_user: Dict[str, Any] = Depends(require_active_user),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, Any]:
    """
    Get API usage summary for current user.
    
    Args:
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        Usage summary statistics
    """
    try:
        api_usage_repo = repo_factory.get_api_usage_repository()
        user_repo = repo_factory.get_user_repository()
        
        # Get user info
        user = await user_repo.get_by_id(current_user["user_id"])
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get usage statistics
        usage_stats = await api_usage_repo.get_usage_stats(current_user["user_id"])
        
        return {
            "current_month_calls": user.monthly_api_calls,
            "api_call_limit": user.api_call_limit,
            "remaining_calls": max(0, user.api_call_limit - user.monthly_api_calls),
            "usage_percentage": (user.monthly_api_calls / user.api_call_limit) * 100,
            "tier": user.tier,
            "statistics": usage_stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get usage summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage summary"
        )


@router.post("/upgrade-tier")
async def upgrade_user_tier(
    target_tier: str,
    current_user: Dict[str, Any] = Depends(require_active_user),
    repo_factory: RepositoryFactory = Depends(get_repository_factory)
) -> Dict[str, str]:
    """
    Upgrade user tier (simplified implementation).
    
    Args:
        target_tier: Target tier (premium, enterprise)
        current_user: Current authenticated user
        repo_factory: Repository factory
        
    Returns:
        Upgrade confirmation
    """
    try:
        # Validate target tier
        valid_tiers = ["premium", "enterprise"]
        if target_tier not in valid_tiers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid tier. Must be one of: {valid_tiers}"
            )
        
        user_repo = repo_factory.get_user_repository()
        
        # Set API limits based on tier
        api_limits = {
            "premium": 10000,
            "enterprise": 100000
        }
        
        # Update user tier and limits
        user_update = UserUpdate(
            tier=target_tier,
            api_call_limit=api_limits[target_tier]
        )
        
        user = await user_repo.update(current_user["user_id"], user_update)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User tier upgraded: {user.email} -> {target_tier}")
        
        return {
            "message": f"Successfully upgraded to {target_tier} tier",
            "new_tier": target_tier,
            "new_limit": str(api_limits[target_tier])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upgrade user tier: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upgrade tier"
        )


def _generate_api_key() -> str:
    """
    Generate a secure API key.
    
    Returns:
        Generated API key
    """
    # Generate 32 bytes of random data
    random_bytes = secrets.token_bytes(32)
    
    # Create API key with prefix
    api_key = f"sk_synapse_{secrets.token_urlsafe(32)}"
    
    return api_key


def _hash_password(password: str) -> str:
    """
    Hash password using SHA-256 (simplified implementation).
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    # In production, use bcrypt or similar
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{hashed}"


def _verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify password against hash.
    
    Args:
        password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches
    """
    try:
        salt, hash_value = hashed_password.split(":")
        return hashlib.sha256((password + salt).encode()).hexdigest() == hash_value
    except ValueError:
        return False