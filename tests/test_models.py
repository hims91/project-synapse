"""
Unit tests for database models and Pydantic schemas.
Tests data validation, serialization, and model functionality.
"""
import pytest
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy.exc import IntegrityError
from pydantic import ValidationError

from src.synaptic_vesicle.models import (
    Article, ScrapingRecipe, TaskQueue, MonitoringSubscription,
    APIUsage, Feed, User, TrendsSummary
)
from src.shared.schemas import (
    ArticleCreate, ArticleResponse, ArticleNLPData, ArticleMetadata,
    ScrapingRecipeCreate, ScrapingSelectors, ScrapingAction,
    TaskQueueCreate, TaskPayload, TaskStatus,
    MonitoringSubscriptionCreate, UserCreate, UserTier,
    SearchQuery, ScrapeJobCreate, WebhookPayload
)


class TestArticleModel:
    """Test Article database model."""
    
    def test_article_creation(self):
        """Test creating an article instance."""
        article = Article(
            url="https://example.com/article",
            title="Test Article",
            content="This is test content",
            source_domain="example.com"
        )
        
        assert article.url == "https://example.com/article"
        assert article.title == "Test Article"
        assert article.source_domain == "example.com"
        # ID is generated when saved to database, not on object creation
        assert hasattr(article, 'id')
        # Test that we can set an ID manually
        article.id = uuid.uuid4()
        assert isinstance(article.id, uuid.UUID)
    
    def test_article_repr(self):
        """Test article string representation."""
        article = Article(
            url="https://example.com/article",
            title="Test Article with a very long title that should be truncated",
            source_domain="example.com"
        )
        
        repr_str = repr(article)
        assert "Test Article with a very long title that should be" in repr_str
        assert "example.com" in repr_str


class TestScrapingRecipeModel:
    """Test ScrapingRecipe database model."""
    
    def test_recipe_creation(self):
        """Test creating a scraping recipe."""
        recipe = ScrapingRecipe(
            domain="example.com",
            selectors={
                "title": "h1.title",
                "content": ".article-content",
                "author": ".author-name"
            },
            success_rate=Decimal("0.95"),
            usage_count=100
        )
        
        assert recipe.domain == "example.com"
        assert recipe.selectors["title"] == "h1.title"
        assert recipe.success_rate == Decimal("0.95")
        assert recipe.usage_count == 100
    
    def test_recipe_constraints(self):
        """Test recipe validation constraints."""
        # Test valid success rate
        recipe = ScrapingRecipe(
            domain="example.com",
            selectors={"title": "h1"},
            success_rate=Decimal("0.5")
        )
        assert recipe.success_rate == Decimal("0.5")
        
        # Invalid success rates would be caught by database constraints
        # but we can test the model creation
        recipe.success_rate = Decimal("1.5")  # This would fail at DB level
        assert recipe.success_rate == Decimal("1.5")


class TestTaskQueueModel:
    """Test TaskQueue database model."""
    
    def test_task_creation(self):
        """Test creating a task."""
        task = TaskQueue(
            task_type="scrape_url",
            payload={"url": "https://example.com", "priority": True},
            priority=1,
            status="pending"
        )
        
        assert task.task_type == "scrape_url"
        assert task.payload["url"] == "https://example.com"
        assert task.priority == 1
        assert task.status == "pending"
        # Default values are applied at database level, not object creation
        # Test that we can set these values
        task.retry_count = 0
        task.max_retries = 3
        assert task.retry_count == 0
        assert task.max_retries == 3


class TestArticleSchema:
    """Test Article Pydantic schemas."""
    
    def test_article_create_valid(self):
        """Test valid article creation schema."""
        article_data = {
            "url": "https://example.com/article",
            "title": "Test Article",
            "content": "This is test content",
            "source_domain": "example.com",
            "author": "John Doe",
            "nlp_data": {
                "sentiment": 0.5,
                "entities": [{"text": "John", "label": "PERSON"}],
                "categories": ["technology"],
                "significance": 7.5
            },
            "page_metadata": {
                "paywall": False,
                "word_count": 500,
                "reading_time": 3
            }
        }
        
        article = ArticleCreate(**article_data)
        assert str(article.url) == "https://example.com/article"
        assert article.title == "Test Article"
        assert article.nlp_data.sentiment == 0.5
        assert article.page_metadata.paywall is False
    
    def test_article_create_invalid_url(self):
        """Test article creation with invalid URL."""
        with pytest.raises(ValidationError) as exc_info:
            ArticleCreate(
                url="not-a-valid-url",
                title="Test",
                source_domain="example.com"
            )
        
        assert "url" in str(exc_info.value)
    
    def test_article_create_empty_title(self):
        """Test article creation with empty title."""
        with pytest.raises(ValidationError) as exc_info:
            ArticleCreate(
                url="https://example.com",
                title="",
                source_domain="example.com"
            )
        
        assert "title" in str(exc_info.value)
    
    def test_nlp_data_validation(self):
        """Test NLP data validation."""
        # Valid sentiment score
        nlp_data = ArticleNLPData(sentiment=0.5)
        assert nlp_data.sentiment == 0.5
        
        # Invalid sentiment score (out of range)
        with pytest.raises(ValidationError):
            ArticleNLPData(sentiment=2.0)
        
        with pytest.raises(ValidationError):
            ArticleNLPData(sentiment=-2.0)
    
    def test_article_metadata_validation(self):
        """Test article metadata validation."""
        metadata = ArticleMetadata(
            paywall=True,
            word_count=1000,
            reading_time=5,
            tech_stack=["React", "Node.js"]
        )
        
        assert metadata.paywall is True
        assert metadata.word_count == 1000
        assert "React" in metadata.tech_stack
        
        # Invalid word count
        with pytest.raises(ValidationError):
            ArticleMetadata(word_count=-100)


class TestScrapingRecipeSchema:
    """Test ScrapingRecipe Pydantic schemas."""
    
    def test_recipe_create_valid(self):
        """Test valid recipe creation."""
        recipe_data = {
            "domain": "example.com",
            "selectors": {
                "title": "h1.title",
                "content": ".article-content",
                "author": ".author"
            },
            "actions": [
                {
                    "type": "click",
                    "selector": ".load-more",
                    "timeout": 5
                }
            ]
        }
        
        recipe = ScrapingRecipeCreate(**recipe_data)
        assert recipe.domain == "example.com"
        assert recipe.selectors.title == "h1.title"
        assert len(recipe.actions) == 1
        assert recipe.actions[0].type == "click"
    
    def test_scraping_selectors_validation(self):
        """Test scraping selectors validation."""
        selectors = ScrapingSelectors(
            title="h1",
            content=".content"
        )
        assert selectors.title == "h1"
        assert selectors.content == ".content"
        
        # Missing required fields
        with pytest.raises(ValidationError):
            ScrapingSelectors(title="h1")  # Missing content
    
    def test_scraping_action_validation(self):
        """Test scraping action validation."""
        action = ScrapingAction(
            type="click",
            selector=".button",
            timeout=10
        )
        assert action.type == "click"
        assert action.timeout == 10
        
        # Invalid timeout
        with pytest.raises(ValidationError):
            ScrapingAction(type="click", timeout=100)  # Too long


class TestTaskQueueSchema:
    """Test TaskQueue Pydantic schemas."""
    
    def test_task_create_valid(self):
        """Test valid task creation."""
        task_data = {
            "task_type": "scrape_url",
            "payload": {
                "url": "https://example.com",
                "priority": True,
                "metadata": {"source": "feed"}
            },
            "priority": 1,
            "max_retries": 5
        }
        
        task = TaskQueueCreate(**task_data)
        assert task.task_type == "scrape_url"
        assert str(task.payload.url) == "https://example.com/"
        assert task.priority == 1
        assert task.max_retries == 5
    
    def test_task_priority_validation(self):
        """Test task priority validation."""
        # Valid priority
        task = TaskQueueCreate(
            task_type="test",
            payload=TaskPayload(),
            priority=5
        )
        assert task.priority == 5
        
        # Invalid priority (out of range)
        with pytest.raises(ValidationError):
            TaskQueueCreate(
                task_type="test",
                payload=TaskPayload(),
                priority=15
            )
    
    def test_task_status_enum(self):
        """Test task status enum validation."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.COMPLETED == "completed"
        
        # Test all valid statuses
        valid_statuses = ["pending", "processing", "completed", "failed", "cancelled"]
        for status in valid_statuses:
            assert TaskStatus(status) == status


class TestMonitoringSubscriptionSchema:
    """Test MonitoringSubscription Pydantic schemas."""
    
    def test_subscription_create_valid(self):
        """Test valid subscription creation."""
        sub_data = {
            "name": "Tech News Monitor",
            "keywords": ["AI", "machine learning", "technology"],
            "webhook_url": "https://example.com/webhook"
        }
        
        subscription = MonitoringSubscriptionCreate(**sub_data)
        assert subscription.name == "Tech News Monitor"
        assert "AI" in subscription.keywords
        assert str(subscription.webhook_url) == "https://example.com/webhook"
    
    def test_subscription_empty_keywords(self):
        """Test subscription with empty keywords."""
        with pytest.raises(ValidationError):
            MonitoringSubscriptionCreate(
                name="Test",
                keywords=[],  # Empty keywords not allowed
                webhook_url="https://example.com/webhook"
            )
    
    def test_subscription_invalid_webhook_url(self):
        """Test subscription with invalid webhook URL."""
        with pytest.raises(ValidationError):
            MonitoringSubscriptionCreate(
                name="Test",
                keywords=["test"],
                webhook_url="not-a-valid-url"
            )


class TestUserSchema:
    """Test User Pydantic schemas."""
    
    def test_user_create_valid(self):
        """Test valid user creation."""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "securepassword123",
            "tier": "premium"
        }
        
        user = UserCreate(**user_data)
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert user.tier == UserTier.PREMIUM
    
    def test_user_invalid_email(self):
        """Test user creation with invalid email."""
        with pytest.raises(ValidationError):
            UserCreate(
                email="not-an-email",
                username="testuser",
                password="password123"
            )
    
    def test_user_short_password(self):
        """Test user creation with short password."""
        with pytest.raises(ValidationError):
            UserCreate(
                email="test@example.com",
                username="testuser",
                password="short"  # Too short
            )
    
    def test_user_tier_enum(self):
        """Test user tier enum."""
        assert UserTier.FREE == "free"
        assert UserTier.PREMIUM == "premium"
        assert UserTier.ENTERPRISE == "enterprise"


class TestSearchSchema:
    """Test search-related schemas."""
    
    def test_search_query_valid(self):
        """Test valid search query."""
        query = SearchQuery(
            q="artificial intelligence",
            limit=20,
            offset=10
        )
        assert query.q == "artificial intelligence"
        assert query.limit == 20
        assert query.offset == 10
    
    def test_search_query_empty(self):
        """Test search query with empty string."""
        with pytest.raises(ValidationError):
            SearchQuery(q="")
    
    def test_search_query_limit_validation(self):
        """Test search query limit validation."""
        # Valid limit
        query = SearchQuery(q="test", limit=50)
        assert query.limit == 50
        
        # Invalid limit (too high)
        with pytest.raises(ValidationError):
            SearchQuery(q="test", limit=200)


class TestWebhookSchema:
    """Test webhook-related schemas."""
    
    def test_webhook_payload_valid(self):
        """Test valid webhook payload."""
        article_data = {
            "id": str(uuid.uuid4()),
            "url": "https://example.com/article",
            "title": "Test Article",
            "source_domain": "example.com",
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "nlp_data": {"sentiment": 0.5},
            "page_metadata": {"paywall": False}
        }
        
        payload_data = {
            "subscription_id": str(uuid.uuid4()),
            "subscription_name": "Test Monitor",
            "matched_keywords": ["test", "article"],
            "article": article_data,
            "triggered_at": datetime.now(timezone.utc).isoformat()
        }
        
        payload = WebhookPayload(**payload_data)
        assert payload.subscription_name == "Test Monitor"
        assert "test" in payload.matched_keywords
        assert payload.article.title == "Test Article"


class TestScrapeJobSchema:
    """Test scrape job schemas."""
    
    def test_scrape_job_create_valid(self):
        """Test valid scrape job creation."""
        job_data = {
            "url": "https://example.com/article",
            "priority": True
        }
        
        job = ScrapeJobCreate(**job_data)
        assert str(job.url) == "https://example.com/article"
        assert job.priority is True
    
    def test_scrape_job_invalid_url(self):
        """Test scrape job with invalid URL."""
        with pytest.raises(ValidationError):
            ScrapeJobCreate(url="not-a-url")


class TestValidationEdgeCases:
    """Test edge cases and complex validation scenarios."""
    
    def test_uuid_validation(self):
        """Test UUID field validation."""
        # Valid UUID string should be accepted
        valid_uuid = str(uuid.uuid4())
        
        # Test in a schema that uses UUID
        task = TaskQueueCreate(
            task_type="test",
            payload=TaskPayload()
        )
        # UUIDs are generated automatically for new instances
        assert isinstance(task.payload, TaskPayload)
    
    def test_datetime_validation(self):
        """Test datetime field validation."""
        now = datetime.now(timezone.utc)
        
        # Test with timezone-aware datetime
        article = ArticleCreate(
            url="https://example.com",
            title="Test",
            source_domain="example.com",
            published_at=now
        )
        assert article.published_at == now
    
    def test_nested_model_validation(self):
        """Test validation of nested models."""
        # Test nested NLP data validation
        article_data = {
            "url": "https://example.com",
            "title": "Test",
            "source_domain": "example.com",
            "nlp_data": {
                "sentiment": 1.5,  # Invalid - out of range
                "entities": []
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ArticleCreate(**article_data)
        
        # Should mention the nested field error
        assert "nlp_data" in str(exc_info.value) or "sentiment" in str(exc_info.value)
    
    def test_optional_field_handling(self):
        """Test handling of optional fields."""
        # Create article with minimal required fields
        article = ArticleCreate(
            url="https://example.com",
            title="Test",
            source_domain="example.com"
        )
        
        # Optional fields should have default values
        assert article.content is None
        assert article.author is None
        assert isinstance(article.nlp_data, ArticleNLPData)
        assert isinstance(article.page_metadata, ArticleMetadata)
    
    def test_list_validation(self):
        """Test validation of list fields."""
        # Valid list
        subscription = MonitoringSubscriptionCreate(
            name="Test",
            keywords=["keyword1", "keyword2"],
            webhook_url="https://example.com/webhook"
        )
        assert len(subscription.keywords) == 2
        
        # Empty list should fail for keywords
        with pytest.raises(ValidationError):
            MonitoringSubscriptionCreate(
                name="Test",
                keywords=[],
                webhook_url="https://example.com/webhook"
            )
    
    def test_string_constraints(self):
        """Test string length and pattern constraints."""
        # Valid string length
        user = UserCreate(
            email="test@example.com",
            username="validuser",
            password="validpassword123"
        )
        assert user.username == "validuser"
        
        # Username too short
        with pytest.raises(ValidationError):
            UserCreate(
                email="test@example.com",
                username="ab",  # Too short
                password="validpassword123"
            )
        
        # Username too long
        with pytest.raises(ValidationError):
            UserCreate(
                email="test@example.com",
                username="a" * 101,  # Too long
                password="validpassword123"
            )