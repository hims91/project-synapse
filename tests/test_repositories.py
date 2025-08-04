"""
Unit tests for repository pattern implementation.
Tests CRUD operations, query optimization, and specialized repository methods.
"""
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.exc import IntegrityError

from src.synaptic_vesicle.repositories import (
    BaseRepository, ArticleRepository, ScrapingRecipeRepository,
    TaskQueueRepository, MonitoringSubscriptionRepository, UserRepository,
    APIUsageRepository, FeedRepository, TrendsSummaryRepository,
    RepositoryFactory
)
from src.synaptic_vesicle.models import Article, ScrapingRecipe, TaskQueue, User
from src.shared.schemas import (
    ArticleCreate, ArticleUpdate, ScrapingRecipeCreate, TaskQueueCreate,
    TaskStatus, UserTier
)


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def sample_article_create():
    """Sample article creation data."""
    return ArticleCreate(
        url="https://example.com/test-article",
        title="Test Article",
        content="This is test content",
        source_domain="example.com"
    )


@pytest.fixture
def sample_article():
    """Sample article model instance."""
    return Article(
        id=uuid.uuid4(),
        url="https://example.com/test-article",
        title="Test Article",
        content="This is test content",
        source_domain="example.com"
    )


class TestBaseRepository:
    """Test BaseRepository functionality."""
    
    @pytest.mark.asyncio
    async def test_create_success(self, mock_session, sample_article_create):
        """Test successful record creation."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock the created object
        created_article = Article(**sample_article_create.model_dump())
        created_article.id = uuid.uuid4()
        
        # Configure session refresh to set the ID
        async def mock_refresh(obj):
            obj.id = created_article.id
        
        mock_session.refresh.side_effect = mock_refresh
        
        result = await repo.create(sample_article_create)
        
        # Verify session operations
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        
        # Verify the created object
        assert result.url == sample_article_create.url
        assert result.title == sample_article_create.title
    
    @pytest.mark.asyncio
    async def test_create_integrity_error(self, mock_session, sample_article_create):
        """Test creation with integrity error."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock integrity error
        mock_session.commit.side_effect = IntegrityError("Duplicate key", None, None)
        
        with pytest.raises(ValueError, match="Data integrity violation"):
            await repo.create(sample_article_create)
        
        mock_session.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_id(self, mock_session, sample_article):
        """Test getting record by ID."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock query result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_article
        mock_session.execute.return_value = mock_result
        
        result = await repo.get(sample_article.id)
        
        assert result == sample_article
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_multi_with_filters(self, mock_session):
        """Test getting multiple records with filters."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock query result
        articles = [Article(id=uuid.uuid4(), url=f"https://example.com/{i}", 
                           title=f"Article {i}", source_domain="example.com") 
                   for i in range(3)]
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = articles
        mock_session.execute.return_value = mock_result
        
        filters = {"source_domain": "example.com"}
        result = await repo.get_multi(skip=0, limit=10, filters=filters)
        
        assert len(result) == 3
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_success(self, mock_session, sample_article):
        """Test successful record update."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock get method
        with patch.object(repo, 'get', return_value=sample_article):
            update_data = ArticleUpdate(title="Updated Title")
            result = await repo.update(sample_article.id, update_data)
            
            assert result.title == "Updated Title"
            mock_session.commit.assert_called_once()
            mock_session.refresh.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_not_found(self, mock_session):
        """Test update when record not found."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock get method to return None
        with patch.object(repo, 'get', return_value=None):
            update_data = ArticleUpdate(title="Updated Title")
            result = await repo.update(uuid.uuid4(), update_data)
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_success(self, mock_session):
        """Test successful record deletion."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock delete result
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        
        result = await repo.delete(uuid.uuid4())
        
        assert result is True
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_not_found(self, mock_session):
        """Test deletion when record not found."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock delete result with no rows affected
        mock_result = AsyncMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result
        
        result = await repo.delete(uuid.uuid4())
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_count_with_filters(self, mock_session):
        """Test counting records with filters."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock count result
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result
        
        filters = {"source_domain": "example.com"}
        result = await repo.count(filters=filters)
        
        assert result == 5
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_exists_true(self, mock_session):
        """Test exists method when record exists."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock exists result
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result
        
        result = await repo.exists(uuid.uuid4())
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_exists_false(self, mock_session):
        """Test exists method when record doesn't exist."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock exists result
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result
        
        result = await repo.exists(uuid.uuid4())
        
        assert result is False


class TestArticleRepository:
    """Test ArticleRepository specialized methods."""
    
    @pytest.mark.asyncio
    async def test_get_by_url(self, mock_session, sample_article):
        """Test getting article by URL."""
        repo = ArticleRepository(mock_session)
        
        # Mock query result
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = sample_article
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_by_url("https://example.com/test-article")
        
        assert result == sample_article
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_full_text(self, mock_session):
        """Test full-text search functionality."""
        repo = ArticleRepository(mock_session)
        
        # Mock search results
        articles = [Article(id=uuid.uuid4(), url=f"https://example.com/{i}", 
                           title=f"Article {i}", source_domain="example.com") 
                   for i in range(2)]
        
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = articles
        mock_session.execute.return_value = mock_result
        
        result = await repo.search_full_text("test query")
        
        assert len(result) == 2
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_domain(self, mock_session):
        """Test getting articles by domain."""
        repo = ArticleRepository(mock_session)
        
        # Mock domain results
        articles = [Article(id=uuid.uuid4(), url=f"https://example.com/{i}", 
                           title=f"Article {i}", source_domain="example.com") 
                   for i in range(3)]
        
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = articles
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_by_domain("example.com")
        
        assert len(result) == 3
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_recent(self, mock_session):
        """Test getting recent articles."""
        repo = ArticleRepository(mock_session)
        
        # Mock recent articles
        articles = [Article(id=uuid.uuid4(), url=f"https://example.com/{i}", 
                           title=f"Article {i}", source_domain="example.com") 
                   for i in range(2)]
        
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = articles
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_recent(hours=24)
        
        assert len(result) == 2
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_sentiment_range(self, mock_session):
        """Test getting articles by sentiment range."""
        repo = ArticleRepository(mock_session)
        
        # Mock sentiment results
        articles = [Article(id=uuid.uuid4(), url=f"https://example.com/{i}", 
                           title=f"Article {i}", source_domain="example.com") 
                   for i in range(2)]
        
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = articles
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_by_sentiment_range(min_sentiment=0.0, max_sentiment=1.0)
        
        assert len(result) == 2
        mock_session.execute.assert_called_once()


class TestScrapingRecipeRepository:
    """Test ScrapingRecipeRepository specialized methods."""
    
    @pytest.mark.asyncio
    async def test_get_by_domain(self, mock_session):
        """Test getting recipe by domain."""
        repo = ScrapingRecipeRepository(mock_session)
        
        recipe = ScrapingRecipe(
            id=uuid.uuid4(),
            domain="example.com",
            selectors={"title": "h1", "content": ".content"},
            success_rate=0.95
        )
        
        # Mock query result
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = recipe
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_by_domain("example.com")
        
        assert result == recipe
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_best_recipes(self, mock_session):
        """Test getting best recipes by success rate."""
        repo = ScrapingRecipeRepository(mock_session)
        
        recipes = [
            ScrapingRecipe(id=uuid.uuid4(), domain=f"example{i}.com", 
                          selectors={"title": "h1"}, success_rate=0.9 + i*0.01)
            for i in range(3)
        ]
        
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = recipes
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_best_recipes(limit=10)
        
        assert len(result) == 3
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_increment_usage(self, mock_session):
        """Test incrementing recipe usage count."""
        repo = ScrapingRecipeRepository(mock_session)
        
        # Mock update result
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        
        result = await repo.increment_usage("example.com")
        
        assert result is True
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_success_rate(self, mock_session):
        """Test updating recipe success rate."""
        repo = ScrapingRecipeRepository(mock_session)
        
        # Mock update result
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        
        result = await repo.update_success_rate("example.com", 0.95)
        
        assert result is True
        mock_session.commit.assert_called_once()


class TestTaskQueueRepository:
    """Test TaskQueueRepository specialized methods."""
    
    @pytest.mark.asyncio
    async def test_get_next_task(self, mock_session):
        """Test getting next task to process."""
        repo = TaskQueueRepository(mock_session)
        
        task = TaskQueue(
            id=uuid.uuid4(),
            task_type="scrape_url",
            payload={"url": "https://example.com"},
            priority=1,
            status=TaskStatus.PENDING
        )
        
        # Mock query result
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = task
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_next_task()
        
        assert result == task
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_status(self, mock_session):
        """Test getting tasks by status."""
        repo = TaskQueueRepository(mock_session)
        
        tasks = [
            TaskQueue(id=uuid.uuid4(), task_type="scrape_url", 
                     payload={"url": f"https://example{i}.com"}, 
                     status=TaskStatus.PENDING)
            for i in range(2)
        ]
        
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = tasks
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_by_status(TaskStatus.PENDING)
        
        assert len(result) == 2
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mark_as_processing(self, mock_session):
        """Test marking task as processing."""
        repo = TaskQueueRepository(mock_session)
        
        # Mock update result
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        
        task_id = uuid.uuid4()
        result = await repo.mark_as_processing(task_id)
        
        assert result is True
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mark_as_completed(self, mock_session):
        """Test marking task as completed."""
        repo = TaskQueueRepository(mock_session)
        
        # Mock update result
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        
        task_id = uuid.uuid4()
        result = await repo.mark_as_completed(task_id)
        
        assert result is True
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mark_as_failed(self, mock_session):
        """Test marking task as failed."""
        repo = TaskQueueRepository(mock_session)
        
        # Mock update result
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        
        task_id = uuid.uuid4()
        result = await repo.mark_as_failed(task_id, "Test error")
        
        assert result is True
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_failed_tasks_for_retry(self, mock_session):
        """Test getting failed tasks that can be retried."""
        repo = TaskQueueRepository(mock_session)
        
        tasks = [
            TaskQueue(id=uuid.uuid4(), task_type="scrape_url", 
                     payload={"url": f"https://example{i}.com"}, 
                     status=TaskStatus.FAILED, retry_count=1, max_retries=3)
            for i in range(2)
        ]
        
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = tasks
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_failed_tasks_for_retry()
        
        assert len(result) == 2
        mock_session.execute.assert_called_once()


class TestUserRepository:
    """Test UserRepository specialized methods."""
    
    @pytest.mark.asyncio
    async def test_get_by_email(self, mock_session):
        """Test getting user by email."""
        repo = UserRepository(mock_session)
        
        user = User(
            id=uuid.uuid4(),
            email="test@example.com",
            username="testuser",
            hashed_password="hashed",
            api_key="test-key",
            tier=UserTier.FREE
        )
        
        # Mock query result
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_by_email("test@example.com")
        
        assert result == user
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_api_key(self, mock_session):
        """Test getting user by API key."""
        repo = UserRepository(mock_session)
        
        user = User(
            id=uuid.uuid4(),
            email="test@example.com",
            username="testuser",
            hashed_password="hashed",
            api_key="test-key",
            tier=UserTier.FREE
        )
        
        # Mock query result
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_by_api_key("test-key")
        
        assert result == user
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_increment_api_calls(self, mock_session):
        """Test incrementing API call count."""
        repo = UserRepository(mock_session)
        
        # Mock update result
        mock_result = AsyncMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        
        user_id = uuid.uuid4()
        result = await repo.increment_api_calls(user_id)
        
        assert result is True
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reset_monthly_api_calls(self, mock_session):
        """Test resetting monthly API calls."""
        repo = UserRepository(mock_session)
        
        # Mock update result
        mock_result = AsyncMock()
        mock_result.rowcount = 5  # 5 users updated
        mock_session.execute.return_value = mock_result
        
        result = await repo.reset_monthly_api_calls()
        
        assert result == 5
        mock_session.commit.assert_called_once()


class TestRepositoryFactory:
    """Test RepositoryFactory functionality."""
    
    def test_factory_properties(self, mock_session):
        """Test that factory creates correct repository instances."""
        factory = RepositoryFactory(mock_session)
        
        assert isinstance(factory.articles, ArticleRepository)
        assert isinstance(factory.scraping_recipes, ScrapingRecipeRepository)
        assert isinstance(factory.task_queue, TaskQueueRepository)
        assert isinstance(factory.users, UserRepository)
        assert isinstance(factory.api_usage, APIUsageRepository)
        assert isinstance(factory.feeds, FeedRepository)
        assert isinstance(factory.trends_summary, TrendsSummaryRepository)
        
        # Verify all repositories use the same session
        assert factory.articles.session == mock_session
        assert factory.users.session == mock_session


class TestRepositoryErrorHandling:
    """Test error handling in repositories."""
    
    @pytest.mark.asyncio
    async def test_create_with_exception(self, mock_session, sample_article_create):
        """Test creation with unexpected exception."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock unexpected exception
        mock_session.commit.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Database error"):
            await repo.create(sample_article_create)
        
        mock_session.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_with_exception(self, mock_session):
        """Test get operation with exception."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock exception during query
        mock_session.execute.side_effect = Exception("Query error")
        
        with pytest.raises(Exception, match="Query error"):
            await repo.get(uuid.uuid4())
    
    @pytest.mark.asyncio
    async def test_update_with_exception(self, mock_session, sample_article):
        """Test update with exception."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock get method to return article
        with patch.object(repo, 'get', return_value=sample_article):
            # Mock commit exception
            mock_session.commit.side_effect = Exception("Update error")
            
            update_data = ArticleUpdate(title="Updated Title")
            
            with pytest.raises(Exception, match="Update error"):
                await repo.update(sample_article.id, update_data)
            
            mock_session.rollback.assert_called_once()


class TestRepositoryQueryOptimization:
    """Test query optimization features."""
    
    @pytest.mark.asyncio
    async def test_get_multi_with_ordering(self, mock_session):
        """Test get_multi with ordering."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock query result
        articles = [Article(id=uuid.uuid4(), url=f"https://example.com/{i}", 
                           title=f"Article {i}", source_domain="example.com") 
                   for i in range(3)]
        
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = articles
        mock_session.execute.return_value = mock_result
        
        # Test ascending order
        result = await repo.get_multi(order_by="title")
        assert len(result) == 3
        
        # Test descending order
        result = await repo.get_multi(order_by="-title")
        assert len(result) == 3
    
    @pytest.mark.asyncio
    async def test_get_multi_with_list_filters(self, mock_session):
        """Test get_multi with list-based filters."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock query result
        articles = [Article(id=uuid.uuid4(), url=f"https://example.com/{i}", 
                           title=f"Article {i}", source_domain="example.com") 
                   for i in range(2)]
        
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = articles
        mock_session.execute.return_value = mock_result
        
        # Test with list filter
        filters = {"source_domain": ["example.com", "test.com"]}
        result = await repo.get_multi(filters=filters)
        
        assert len(result) == 2
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_count_with_complex_filters(self, mock_session):
        """Test count with complex filtering."""
        repo = BaseRepository(mock_session, Article)
        
        # Mock count result
        mock_result = AsyncMock()
        mock_result.scalar.return_value = 10
        mock_session.execute.return_value = mock_result
        
        filters = {
            "source_domain": "example.com",
            "title": ["Article 1", "Article 2"]
        }
        result = await repo.count(filters=filters)
        
        assert result == 10
        mock_session.execute.assert_called_once()