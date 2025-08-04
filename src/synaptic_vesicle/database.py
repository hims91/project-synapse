"""
Synaptic Vesicle - Database Connection and Session Management
Layer 2: Signal Network

This module handles database connectivity, session management, and health checks.
"""
import os
import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    AsyncEngine, 
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.pool import NullPool
from sqlalchemy import text
import structlog

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """
    Manages database connections and sessions for the Synaptic Vesicle.
    Implements connection pooling, health checks, and retry logic.
    """
    
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._is_connected = False
    
    def get_database_url(self) -> str:
        """Get database URL from environment variables."""
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            # Fallback to individual components
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            database = os.getenv("DB_NAME", "synapse")
            username = os.getenv("DB_USER", "synapse")
            password = os.getenv("DB_PASSWORD", "synapse_dev_password")
            database_url = f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
        
        return database_url
    
    async def initialize(self) -> None:
        """Initialize database connection and session factory."""
        try:
            database_url = self.get_database_url()
            
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                database_url,
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
                pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
                max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "30")),
                pool_pre_ping=True,  # Validate connections before use
                pool_recycle=3600,   # Recycle connections every hour
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            await self.health_check()
            self._is_connected = True
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database connection", error=str(e))
            raise
    
    async def health_check(self) -> bool:
        """
        Perform database health check.
        Returns True if database is accessible, False otherwise.
        """
        if not self.engine:
            return False
        
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False
    
    async def close(self) -> None:
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            self._is_connected = False
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic cleanup.
        
        Usage:
            async with db_manager.get_session() as session:
                # Use session here
                pass
        """
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._is_connected
    
    async def execute_raw_sql(self, sql: str, params: dict = None) -> any:
        """
        Execute raw SQL query.
        Use with caution - prefer using the ORM when possible.
        """
        if not self.engine:
            raise RuntimeError("Database not initialized")
        
        async with self.engine.begin() as conn:
            result = await conn.execute(text(sql), params or {})
            return result


# Global database manager instance
db_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function for FastAPI to get database sessions.
    
    Usage in FastAPI endpoints:
        @app.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_db_session)):
            # Use db session here
            pass
    """
    async with db_manager.get_session() as session:
        yield session


async def init_database() -> None:
    """Initialize database connection. Call this on application startup."""
    await db_manager.initialize()


async def close_database() -> None:
    """Close database connections. Call this on application shutdown."""
    await db_manager.close()


class DatabaseHealthCheck:
    """Database health check utility for monitoring."""
    
    @staticmethod
    async def check_connection() -> dict:
        """
        Comprehensive database health check.
        Returns status information for monitoring.
        """
        status = {
            "status": "unhealthy",
            "connected": False,
            "response_time_ms": None,
            "error": None
        }
        
        try:
            import time
            start_time = time.time()
            
            is_healthy = await db_manager.health_check()
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            status.update({
                "status": "healthy" if is_healthy else "unhealthy",
                "connected": is_healthy,
                "response_time_ms": round(response_time, 2)
            })
            
        except Exception as e:
            status["error"] = str(e)
            logger.error("Database health check failed", error=str(e))
        
        return status
    
    @staticmethod
    async def check_tables() -> dict:
        """
        Check if all required tables exist.
        Returns table status information.
        """
        required_tables = [
            "articles", "scraping_recipes", "task_queue", 
            "monitoring_subscriptions", "api_usage", "feeds", 
            "users", "trends_summary"
        ]
        
        table_status = {}
        
        try:
            async with db_manager.get_session() as session:
                for table in required_tables:
                    try:
                        result = await session.execute(
                            text(f"SELECT 1 FROM {table} LIMIT 1")
                        )
                        table_status[table] = "exists"
                    except Exception:
                        table_status[table] = "missing"
        
        except Exception as e:
            logger.error("Table check failed", error=str(e))
            return {"error": str(e)}
        
        return table_status


# Retry decorator for database operations
def with_db_retry(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry database operations on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Database operation failed, retrying in {delay}s",
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            error=str(e)
                        )
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(
                            "Database operation failed after all retries",
                            attempts=max_retries + 1,
                            error=str(e)
                        )
            
            raise last_exception
        
        return wrapper
    return decorator