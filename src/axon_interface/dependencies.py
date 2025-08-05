"""
Dependencies for Axon Interface API

Provides dependency injection functions for FastAPI endpoints.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..central_cortex.dependencies import get_database_manager
from ..synaptic_vesicle.repositories import RepositoryFactory


async def get_db_session() -> AsyncSession:
    """Get database session for dependency injection."""
    db_manager = get_database_manager()
    async with db_manager.get_session() as session:
        yield session


def get_repository_factory(
    session: AsyncSession = Depends(get_db_session)
) -> RepositoryFactory:
    """Get repository factory for dependency injection."""
    return RepositoryFactory(session)