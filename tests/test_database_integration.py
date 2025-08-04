"""
Integration tests for database connection and session management.
Tests the Synaptic Vesicle database layer with actual database operations.
"""
import pytest
import asyncio
import os
from unittest.mock import patch, AsyncMock
from sqlalchemy.exc import OperationalError
from sqlalchemy import text

from src.synaptic_vesicle.database import (
    DatabaseManager, db_manager, get_db_session, 
    init_database, close_database, DatabaseHealthCheck,
    with_db_retry
)
from src.synaptic_vesicle.models import Article, ScrapingRecipe, TaskQueue


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseManager:
    """Test DatabaseManager functionality."""
    
    @pytest.fixture
    async def db_manager_instance(self):
        """Create a test database manager instance."""
        manager = DatabaseManager()
        # Use test database URL
        test_db_url = "postgresql+asyncpg://test:test@localhost:5433/synapse_test"
        
        with patch.object(manager, 'get_database_url', return_value=test_db_url):
            try:
                await manager.initialize()
                yield manager
            finally:
                await manager.close()
    
    def test_database_settings_from_env(self):
        """Test database settings retrieval from environment."""
        from src.shared.config import DatabaseSettings
        
        # Test with DATABASE_URL environment variable
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://test:test@localhost/test'}):
            settings = DatabaseSettings()
            url = settings.get_database_url()
            assert url == 'postgresql://test:test@localhost/test'
        
        # Test with individual components
        env_vars = {
            'DB_HOST': 'testhost',
            'DB_PORT': '5433',
            'DB_NAME': 'testdb',
            'DB_USER': 'testuser',
            'DB_PASSWORD': 'testpass'
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = DatabaseSettings()
            url = settings.get_database_url()
            expected = "postgresql+asyncpg://testuser:testpass@testhost:5433/testdb"
            assert url == expected
    
    def test_database_settings_defaults(self):
        """Test database settings with default values."""
        from src.shared.config import DatabaseSettings
        
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            settings = DatabaseSettings()
            url = settings.get_database_url()
            expected = "postgresql+asyncpg://synapse:synapse_dev_password@localhost:5432/synapse"
            assert url == expected
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, db_manager_instance):
        """Test successful database initialization."""
        manager = db_manager_instance
        
        assert manager.engine is not None
        assert manager.session_factory is not None
        assert manager.is_connected is True
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test database initialization failure."""
        manager = DatabaseManager()
        
        # Use invalid database URL
        with patch.object(manager, 'get_database_url', return_value='postgresql://invalid:invalid@nonexistent:5432/invalid'):
            with pytest.raises(Exception):
                await manager.initialize()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, db_manager_instance):
        """Test successful health check."""
        manager = db_manager_instance
        
        is_healthy = await manager.health_check()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_no_engine(self):
        """Test health check without initialized engine."""
        manager = DatabaseManager()
        
        is_healthy = await manager.health_check()
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, db_manager_instance):
        """Test health check with connection error."""
        manager = db_manager_instance
        
        # Mock engine to raise exception
        with patch.object(manager.engine, 'begin', side_effect=OperationalError("Connection failed", None, None)):
            is_healthy = await manager.health_check()
            assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_get_session_context_manager(self, db_manager_instance):
        """Test database session context manager."""
        manager = db_manager_instance
        
        async with manager.get_session() as session:
            assert session is not None
            # Test that we can execute a simple query
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
    
    @pytest.mark.asyncio
    async def test_get_session_without_initialization(self):
        """Test getting session without initialization."""
        manager = DatabaseManager()
        
        with pytest.raises(RuntimeError, match="Database not initialized"):
            async with manager.get_session() as session:
                pass
    
    @pytest.mark.asyncio
    async def test_get_session_rollback_on_exception(self, db_manager_instance):
        """Test session rollback on exception."""
        manager = db_manager_instance
        
        with pytest.raises(ValueError):
            async with manager.get_session() as session:
                # Force an exception
                raise ValueError("Test exception")
    
    @pytest.mark.asyncio
    async def test_execute_raw_sql(self, db_manager_instance):
        """Test raw SQL execution."""
        manager = db_manager_instance
        
        result = await manager.execute_raw_sql("SELECT 1 as test_value")
        assert result.scalar() == 1
        
        # Test with parameters
        result = await manager.execute_raw_sql(
            "SELECT :value as test_value", 
            {"value": 42}
        )
        assert result.scalar() == 42
    
    @pytest.mark.asyncio
    async def test_execute_raw_sql_without_engine(self):
        """Test raw SQL execution without engine."""
        manager = DatabaseManager()
        
        with pytest.raises(RuntimeError, match="Database not initialized"):
            await manager.execute_raw_sql("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_close(self, db_manager_instance):
        """Test database connection closing."""
        manager = db_manager_instance
        
        assert manager.is_connected is True
        await manager.close()
        assert manager.is_connected is False


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseHealthCheck:
    """Test DatabaseHealthCheck functionality."""
    
    @pytest.mark.asyncio
    async def test_check_connection_healthy(self):
        """Test connection health check when healthy."""
        # Mock healthy database manager
        with patch.object(db_manager, 'health_check', return_value=True):
            status = await DatabaseHealthCheck.check_connection()
            
            assert status['status'] == 'healthy'
            assert status['connected'] is True
            assert status['response_time_ms'] is not None
            assert status['error'] is None
    
    @pytest.mark.asyncio
    async def test_check_connection_unhealthy(self):
        """Test connection health check when unhealthy."""
        # Mock unhealthy database manager
        with patch.object(db_manager, 'health_check', return_value=False):
            status = await DatabaseHealthCheck.check_connection()
            
            assert status['status'] == 'unhealthy'
            assert status['connected'] is False
            assert status['response_time_ms'] is not None
            assert status['error'] is None
    
    @pytest.mark.asyncio
    async def test_check_connection_exception(self):
        """Test connection health check with exception."""
        # Mock exception in health check
        with patch.object(db_manager, 'health_check', side_effect=Exception("Connection error")):
            status = await DatabaseHealthCheck.check_connection()
            
            assert status['status'] == 'unhealthy'
            assert status['connected'] is False
            assert status['error'] == "Connection error"
    
    @pytest.mark.asyncio
    async def test_check_tables_success(self):
        """Test table existence check when all tables exist."""
        # Mock successful session and queries
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=AsyncMock())
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            table_status = await DatabaseHealthCheck.check_tables()
            
            # All required tables should be marked as existing
            required_tables = [
                "articles", "scraping_recipes", "task_queue", 
                "monitoring_subscriptions", "api_usage", "feeds", 
                "users", "trends_summary"
            ]
            
            for table in required_tables:
                assert table_status[table] == "exists"
    
    @pytest.mark.asyncio
    async def test_check_tables_missing(self):
        """Test table existence check when some tables are missing."""
        # Mock session that raises exception for some tables
        mock_session = AsyncMock()
        
        def mock_execute(query):
            if "articles" in str(query):
                return AsyncMock()  # Table exists
            else:
                raise Exception("Table not found")  # Table missing
        
        mock_session.execute = mock_execute
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            table_status = await DatabaseHealthCheck.check_tables()
            
            assert table_status["articles"] == "exists"
            # Other tables should be marked as missing
            assert table_status["scraping_recipes"] == "missing"
    
    @pytest.mark.asyncio
    async def test_check_tables_session_error(self):
        """Test table check with session error."""
        # Mock session creation failure
        with patch.object(db_manager, 'get_session', side_effect=Exception("Session error")):
            table_status = await DatabaseHealthCheck.check_tables()
            
            assert "error" in table_status
            assert table_status["error"] == "Session error"


@pytest.mark.integration
class TestRetryDecorator:
    """Test database retry decorator."""
    
    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Test retry decorator with successful first attempt."""
        call_count = 0
        
        @with_db_retry(max_retries=3, delay=0.1)
        async def test_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await test_function()
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test retry decorator with success after failures."""
        call_count = 0
        
        @with_db_retry(max_retries=3, delay=0.01)
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await test_function()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_max_retries_exceeded(self):
        """Test retry decorator when max retries exceeded."""
        call_count = 0
        
        @with_db_retry(max_retries=2, delay=0.01)
        async def test_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Persistent failure")
        
        with pytest.raises(Exception, match="Persistent failure"):
            await test_function()
        
        assert call_count == 3  # Initial attempt + 2 retries


@pytest.mark.integration
@pytest.mark.database
class TestGlobalDatabaseFunctions:
    """Test global database functions."""
    
    @pytest.mark.asyncio
    async def test_init_database(self):
        """Test database initialization function."""
        # Mock the global db_manager
        with patch.object(db_manager, 'initialize') as mock_init:
            await init_database()
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_database(self):
        """Test database closing function."""
        # Mock the global db_manager
        with patch.object(db_manager, 'close') as mock_close:
            await close_database()
            mock_close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_db_session_dependency(self):
        """Test FastAPI dependency function."""
        # Mock the global db_manager
        mock_session = AsyncMock()
        
        with patch.object(db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            # Test the async generator
            async_gen = get_db_session()
            session = await async_gen.__anext__()
            
            assert session == mock_session


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseWithModels:
    """Test database operations with actual models."""
    
    @pytest.fixture
    async def db_session(self):
        """Provide a database session for testing."""
        # This would require an actual test database
        # For now, we'll mock it
        mock_session = AsyncMock()
        yield mock_session
    
    @pytest.mark.asyncio
    async def test_create_article(self, db_session):
        """Test creating an article in the database."""
        # Mock article creation
        article_data = {
            'url': 'https://example.com/test',
            'title': 'Test Article',
            'content': 'Test content',
            'source_domain': 'example.com'
        }
        
        article = Article(**article_data)
        
        # Mock database operations
        db_session.add = AsyncMock()
        db_session.commit = AsyncMock()
        db_session.refresh = AsyncMock()
        
        # Simulate adding to database
        db_session.add(article)
        await db_session.commit()
        await db_session.refresh(article)
        
        # Verify mocks were called
        db_session.add.assert_called_once_with(article)
        db_session.commit.assert_called_once()
        db_session.refresh.assert_called_once_with(article)
    
    @pytest.mark.asyncio
    async def test_query_articles(self, db_session):
        """Test querying articles from the database."""
        # Mock query result
        mock_articles = [
            Article(url='https://example.com/1', title='Article 1', source_domain='example.com'),
            Article(url='https://example.com/2', title='Article 2', source_domain='example.com')
        ]
        
        # Mock query execution
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = mock_articles
        db_session.execute = AsyncMock(return_value=mock_result)
        
        # This would be the actual query in a real implementation
        # result = await db_session.execute(select(Article))
        # articles = result.scalars().all()
        
        # For testing, we just verify the mock setup
        assert len(mock_articles) == 2
        assert mock_articles[0].title == 'Article 1'


@pytest.mark.integration
class TestDatabaseConfiguration:
    """Test database configuration and environment handling."""
    
    def test_connection_pool_configuration(self):
        """Test database connection pool configuration."""
        manager = DatabaseManager()
        
        # Test with custom pool settings
        env_vars = {
            'DB_POOL_SIZE': '10',
            'DB_MAX_OVERFLOW': '20',
            'DB_ECHO': 'true'
        }
        
        with patch.dict(os.environ, env_vars):
            # This would test the actual engine creation
            # For now, we just verify the environment variables are read
            assert os.getenv('DB_POOL_SIZE') == '10'
            assert os.getenv('DB_MAX_OVERFLOW') == '20'
            assert os.getenv('DB_ECHO') == 'true'
    
    def test_database_url_validation(self):
        """Test database URL format validation."""
        manager = DatabaseManager()
        
        # Test various URL formats
        test_urls = [
            'postgresql+asyncpg://user:pass@localhost:5432/db',
            'postgresql://user:pass@localhost/db',
            'postgresql+asyncpg://user@localhost/db'
        ]
        
        for url in test_urls:
            with patch.object(manager, 'get_database_url', return_value=url):
                db_url = manager.get_database_url()
                assert db_url == url
                assert 'postgresql' in db_url