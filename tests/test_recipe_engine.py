"""
Unit tests for the Scraping Recipe Engine.
Tests recipe creation, validation, execution, and caching functionality.
"""
import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal

from src.neurons.recipe_engine import (
    ScrapingRecipeEngine, RecipeValidationError, RecipeExecutionError
)
from src.shared.schemas import (
    ScrapingRecipeResponse, ScrapingSelectors, ScrapingAction,
    ArticleCreate, RecipeCreatedBy
)


@pytest.fixture
def recipe_engine():
    """Create a recipe engine instance for testing."""
    return ScrapingRecipeEngine()


@pytest.fixture
def sample_selectors():
    """Sample CSS selectors for testing."""
    return ScrapingSelectors(
        title="h1.article-title",
        content=".article-content",
        author=".author-name",
        publish_date=".publish-date",
        summary=".article-summary"
    )


@pytest.fixture
def sample_actions():
    """Sample scraping actions for testing."""
    return [
        ScrapingAction(type="wait", timeout=2),
        ScrapingAction(type="click", selector=".load-more", timeout=5)
    ]


@pytest.fixture
def sample_recipe():
    """Sample recipe response for testing."""
    return ScrapingRecipeResponse(
        id=uuid.uuid4(),
        domain="example.com",
        selectors=ScrapingSelectors(
            title="h1",
            content=".content"
        ),
        actions=[],
        success_rate=Decimal("0.95"),
        usage_count=100,
        last_updated=datetime.utcnow(),
        created_by=RecipeCreatedBy.LEARNING
    )


@pytest.fixture
def sample_html():
    """Sample HTML content for testing."""
    return """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Test Article Title</h1>
            <div class="content">
                <p>This is the main content of the article.</p>
                <p>It contains multiple paragraphs with useful information.</p>
            </div>
            <div class="author-name">John Doe</div>
            <div class="publish-date">2024-01-15</div>
            <div class="article-summary">This is a summary of the article.</div>
        </body>
    </html>
    """


class TestScrapingRecipeEngine:
    """Test cases for ScrapingRecipeEngine."""
    
    @pytest.mark.asyncio
    async def test_get_recipe_cache_hit(self, recipe_engine, sample_recipe):
        """Test getting recipe from cache."""
        # Setup cache
        recipe_engine.recipe_cache["example.com"] = sample_recipe
        recipe_engine.last_cache_update["example.com"] = datetime.utcnow()
        
        # Test cache hit
        result = await recipe_engine.get_recipe("example.com")
        
        assert result == sample_recipe
        assert result.domain == "example.com"
    
    @pytest.mark.asyncio
    async def test_get_recipe_cache_miss(self, recipe_engine):
        """Test getting recipe when not in cache."""
        with patch('src.synaptic_vesicle.database.get_db_session') as mock_session:
            mock_repo = AsyncMock()
            mock_repo.get_by_domain.return_value = None
            
            mock_session.return_value.__aenter__.return_value = Mock()
            
            with patch('src.synaptic_vesicle.repositories.ScrapingRecipeRepository', return_value=mock_repo):
                result = await recipe_engine.get_recipe("nonexistent.com")
                
                assert result is None
                mock_repo.get_by_domain.assert_called_once_with("nonexistent.com")
    
    @pytest.mark.asyncio
    async def test_get_recipe_database_fetch(self, recipe_engine, sample_recipe):
        """Test fetching recipe from database and caching."""
        with patch('src.synaptic_vesicle.database.get_db_session') as mock_session:
            mock_repo = AsyncMock()
            mock_repo.get_by_domain.return_value = sample_recipe
            
            mock_session.return_value.__aenter__.return_value = Mock()
            
            with patch('src.synaptic_vesicle.repositories.ScrapingRecipeRepository', return_value=mock_repo):
                result = await recipe_engine.get_recipe("example.com")
                
                assert result == sample_recipe
                assert "example.com" in recipe_engine.recipe_cache
                assert recipe_engine.recipe_cache["example.com"] == sample_recipe
    
    @pytest.mark.asyncio
    async def test_create_recipe_success(self, recipe_engine, sample_selectors, sample_actions):
        """Test successful recipe creation."""
        with patch('src.synaptic_vesicle.database.get_db_session') as mock_session:
            mock_repo = AsyncMock()
            created_recipe = ScrapingRecipeResponse(
                id=uuid.uuid4(),
                domain="newsite.com",
                selectors=sample_selectors,
                actions=sample_actions,
                success_rate=Decimal("0.0"),
                usage_count=0,
                last_updated=datetime.utcnow(),
                created_by=RecipeCreatedBy.LEARNING
            )
            mock_repo.create.return_value = created_recipe
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            with patch('src.synaptic_vesicle.repositories.ScrapingRecipeRepository', return_value=mock_repo):
                result = await recipe_engine.create_recipe(
                    "newsite.com", sample_selectors, sample_actions, "learning"
                )
                
                assert result == created_recipe
                assert "newsite.com" in recipe_engine.recipe_cache
                mock_session_instance.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_recipe_validation_error(self, recipe_engine):
        """Test recipe creation with validation error."""
        invalid_selectors = ScrapingSelectors(
            title="",  # Empty title selector
            content=".content"
        )
        
        with pytest.raises(RecipeValidationError):
            await recipe_engine.create_recipe("test.com", invalid_selectors)
    
    @pytest.mark.asyncio
    async def test_update_recipe_success(self, recipe_engine, sample_recipe):
        """Test updating recipe success rate."""
        with patch('src.synaptic_vesicle.database.get_db_session') as mock_session:
            mock_repo = AsyncMock()
            mock_repo.get_by_domain.return_value = sample_recipe
            
            updated_recipe = ScrapingRecipeResponse(
                id=sample_recipe.id,
                domain=sample_recipe.domain,
                selectors=sample_recipe.selectors,
                actions=sample_recipe.actions,
                success_rate=Decimal("0.96"),  # Increased success rate
                usage_count=101,  # Increased usage count
                last_updated=datetime.utcnow(),
                created_by=sample_recipe.created_by
            )
            mock_repo.update.return_value = updated_recipe
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            with patch('src.synaptic_vesicle.repositories.ScrapingRecipeRepository', return_value=mock_repo):
                await recipe_engine.update_recipe_success("example.com", True)
                
                # Verify update was called with correct success rate calculation
                mock_repo.update.assert_called_once()
                update_call_args = mock_repo.update.call_args[0][1]
                assert update_call_args.usage_count == 101
                mock_session_instance.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_recipe_success(self, recipe_engine, sample_recipe, sample_html):
        """Test successful recipe execution."""
        result = await recipe_engine.execute_recipe(
            sample_recipe, sample_html, "https://example.com/article"
        )
        
        assert isinstance(result, ArticleCreate)
        assert result.title == "Test Article Title"
        assert "main content of the article" in result.content
        assert result.source_domain == "example.com"
        assert result.url == "https://example.com/article"
    
    @pytest.mark.asyncio
    async def test_execute_recipe_missing_title(self, recipe_engine, sample_recipe):
        """Test recipe execution with missing title element."""
        html_without_title = "<html><body><div class='content'>Content</div></body></html>"
        
        with pytest.raises(RecipeExecutionError) as exc_info:
            await recipe_engine.execute_recipe(
                sample_recipe, html_without_title, "https://example.com/article"
            )
        
        assert "Title selector" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execute_recipe_missing_content(self, recipe_engine, sample_recipe):
        """Test recipe execution with missing content element."""
        html_without_content = "<html><body><h1>Title</h1></body></html>"
        
        with pytest.raises(RecipeExecutionError) as exc_info:
            await recipe_engine.execute_recipe(
                sample_recipe, html_without_content, "https://example.com/article"
            )
        
        assert "Content selector" in str(exc_info.value)
    
    def test_validate_recipe_success(self, recipe_engine, sample_selectors, sample_actions):
        """Test successful recipe validation."""
        is_valid, errors = recipe_engine.validate_recipe(sample_selectors, sample_actions)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_recipe_missing_title(self, recipe_engine, sample_actions):
        """Test recipe validation with missing title selector."""
        invalid_selectors = ScrapingSelectors(
            title="",
            content=".content"
        )
        
        is_valid, errors = recipe_engine.validate_recipe(invalid_selectors, sample_actions)
        
        assert is_valid is False
        assert len(errors) > 0
        assert "Title selector is required" in errors[0]
    
    def test_validate_recipe_missing_content(self, recipe_engine, sample_actions):
        """Test recipe validation with missing content selector."""
        invalid_selectors = ScrapingSelectors(
            title="h1",
            content=""
        )
        
        is_valid, errors = recipe_engine.validate_recipe(invalid_selectors, sample_actions)
        
        assert is_valid is False
        assert len(errors) > 0
        assert "Content selector is required" in errors[0]
    
    def test_validate_recipe_invalid_css_selector(self, recipe_engine, sample_actions):
        """Test recipe validation with invalid CSS selector."""
        invalid_selectors = ScrapingSelectors(
            title="h1[[[invalid",  # Invalid CSS selector
            content=".content"
        )
        
        is_valid, errors = recipe_engine.validate_recipe(invalid_selectors, sample_actions)
        
        assert is_valid is False
        assert len(errors) > 0
        assert "Invalid CSS selector" in errors[0]
    
    def test_validate_recipe_invalid_action_type(self, recipe_engine, sample_selectors):
        """Test recipe validation with invalid action type."""
        invalid_actions = [
            ScrapingAction(type="invalid_action", timeout=5)
        ]
        
        is_valid, errors = recipe_engine.validate_recipe(sample_selectors, invalid_actions)
        
        assert is_valid is False
        assert len(errors) > 0
        assert "invalid type" in errors[0]
    
    def test_validate_recipe_invalid_timeout(self, recipe_engine, sample_selectors):
        """Test recipe validation with invalid timeout."""
        # Create action with invalid timeout manually to bypass Pydantic validation
        invalid_actions = []
        action = ScrapingAction(type="wait", timeout=5)
        action.timeout = 100  # Set invalid timeout after creation
        invalid_actions.append(action)
        
        is_valid, errors = recipe_engine.validate_recipe(sample_selectors, invalid_actions)
        
        assert is_valid is False
        assert len(errors) > 0
        assert "timeout must be between" in errors[0]
    
    def test_is_cache_valid_fresh(self, recipe_engine, sample_recipe):
        """Test cache validity with fresh entry."""
        recipe_engine.recipe_cache["example.com"] = sample_recipe
        recipe_engine.last_cache_update["example.com"] = datetime.utcnow()
        
        assert recipe_engine._is_cache_valid("example.com") is True
    
    def test_is_cache_valid_expired(self, recipe_engine, sample_recipe):
        """Test cache validity with expired entry."""
        recipe_engine.recipe_cache["example.com"] = sample_recipe
        recipe_engine.last_cache_update["example.com"] = datetime.utcnow() - timedelta(hours=2)
        
        assert recipe_engine._is_cache_valid("example.com") is False
    
    def test_is_cache_valid_missing(self, recipe_engine):
        """Test cache validity with missing entry."""
        assert recipe_engine._is_cache_valid("nonexistent.com") is False
    
    def test_is_valid_css_selector(self, recipe_engine):
        """Test CSS selector validation."""
        assert recipe_engine._is_valid_css_selector("h1") is True
        assert recipe_engine._is_valid_css_selector(".class") is True
        assert recipe_engine._is_valid_css_selector("#id") is True
        assert recipe_engine._is_valid_css_selector("div > p") is True
        assert recipe_engine._is_valid_css_selector("h1[[[invalid") is False
    
    def test_clean_text(self, recipe_engine):
        """Test text cleaning functionality."""
        assert recipe_engine._clean_text("  Hello   World  ") == "Hello World"
        assert recipe_engine._clean_text("\n\t  Text  \n\t") == "Text"
        assert recipe_engine._clean_text("") == ""
        assert recipe_engine._clean_text(None) == ""
    
    def test_parse_date(self, recipe_engine):
        """Test date parsing functionality."""
        # This test requires dateutil to be installed
        try:
            result = recipe_engine._parse_date("2024-01-15")
            assert result is not None
            assert result.year == 2024
            assert result.month == 1
            assert result.day == 15
        except ImportError:
            # dateutil not available, should return None
            result = recipe_engine._parse_date("2024-01-15")
            assert result is None


class TestRecipeEngineIntegration:
    """Integration tests for recipe engine."""
    
    @pytest.mark.asyncio
    async def test_full_recipe_workflow(self, recipe_engine, sample_selectors, sample_html):
        """Test complete recipe workflow: create, execute, update."""
        with patch('src.synaptic_vesicle.database.get_db_session') as mock_session:
            # Mock repository
            mock_repo = AsyncMock()
            
            # Mock recipe creation
            created_recipe = ScrapingRecipeResponse(
                id=uuid.uuid4(),
                domain="workflow.com",
                selectors=sample_selectors,
                actions=[],
                success_rate=Decimal("0.0"),
                usage_count=0,
                last_updated=datetime.utcnow(),
                created_by=RecipeCreatedBy.LEARNING
            )
            mock_repo.create.return_value = created_recipe
            
            # Mock recipe retrieval for update
            mock_repo.get_by_domain.return_value = created_recipe
            
            # Mock recipe update
            updated_recipe = ScrapingRecipeResponse(
                id=created_recipe.id,
                domain=created_recipe.domain,
                selectors=created_recipe.selectors,
                actions=created_recipe.actions,
                success_rate=Decimal("1.0"),
                usage_count=1,
                last_updated=datetime.utcnow(),
                created_by=created_recipe.created_by
            )
            mock_repo.update.return_value = updated_recipe
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            with patch('src.synaptic_vesicle.repositories.ScrapingRecipeRepository', return_value=mock_repo):
                # Step 1: Create recipe
                recipe = await recipe_engine.create_recipe(
                    "workflow.com", sample_selectors, [], "learning"
                )
                assert recipe.domain == "workflow.com"
                
                # Step 2: Execute recipe
                article = await recipe_engine.execute_recipe(
                    recipe, sample_html, "https://workflow.com/article"
                )
                assert article.title == "Test Article Title"
                
                # Step 3: Update success rate
                await recipe_engine.update_recipe_success("workflow.com", True)
                
                # Verify all operations were called
                mock_repo.create.assert_called_once()
                mock_repo.get_by_domain.assert_called_once()
                mock_repo.update.assert_called_once()
                assert mock_session_instance.commit.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])