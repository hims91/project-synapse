"""
Unit tests for the Recipe Testing Framework.
Tests recipe testing, validation, and benchmarking functionality.
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal

import httpx

from src.neurons.recipe_tester import (
    RecipeTester, RecipeTestResult, RecipeValidationResult,
    create_recipe_tester
)
from src.neurons.recipe_engine import ScrapingRecipeEngine
from src.shared.schemas import (
    ScrapingRecipeResponse, ScrapingSelectors, ScrapingAction,
    ArticleCreate, RecipeCreatedBy
)


@pytest.fixture
def mock_recipe_engine():
    """Create a mock recipe engine for testing."""
    return Mock(spec=ScrapingRecipeEngine)


@pytest.fixture
def recipe_tester(mock_recipe_engine):
    """Create a recipe tester instance for testing."""
    return RecipeTester(mock_recipe_engine)


@pytest.fixture
def sample_recipe():
    """Sample recipe for testing."""
    return ScrapingRecipeResponse(
        id="test-recipe-id",
        domain="example.com",
        selectors=ScrapingSelectors(
            title="h1.title",
            content=".content",
            author=".author",
            publish_date=".date"
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
            <h1 class="title">Test Article Title</h1>
            <div class="content">
                <p>This is the main content of the article with sufficient length.</p>
                <p>It contains multiple paragraphs to ensure good content quality.</p>
                <p>The content is detailed and informative for testing purposes.</p>
            </div>
            <div class="author">Jane Smith</div>
            <div class="date">2024-01-15</div>
        </body>
    </html>
    """


@pytest.fixture
def sample_article():
    """Sample article data for testing."""
    return ArticleCreate(
        url="https://example.com/article",
        title="Test Article Title",
        content="This is the main content of the article with sufficient length. It contains multiple paragraphs to ensure good content quality. The content is detailed and informative for testing purposes.",
        author="Jane Smith",
        published_at=datetime(2024, 1, 15),
        source_domain="example.com"
    )


class TestRecipeTester:
    """Test cases for RecipeTester."""
    
    @pytest.mark.asyncio
    async def test_test_recipe_on_url_success(self, recipe_tester, sample_recipe, sample_article):
        """Test successful recipe testing on URL."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = "<html><body><h1>Test</h1></body></html>"
        mock_response.raise_for_status = Mock()
        
        recipe_tester.http_client.get = AsyncMock(return_value=mock_response)
        
        # Mock recipe engine execution
        recipe_tester.recipe_engine.execute_recipe = AsyncMock(return_value=sample_article)
        
        # Test recipe execution
        result = await recipe_tester.test_recipe_on_url(sample_recipe, "https://example.com/test")
        
        assert result.success is True
        assert result.extracted_data is not None
        assert result.extracted_data['title'] == "Test Article Title"
        assert result.extracted_data['author'] == "Jane Smith"
        assert result.execution_time_ms is not None
        assert result.content_quality_score is not None
        
        # Verify HTTP client was called
        recipe_tester.http_client.get.assert_called_once_with("https://example.com/test")
        
        # Verify recipe engine was called
        recipe_tester.recipe_engine.execute_recipe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_test_recipe_on_url_http_error(self, recipe_tester, sample_recipe):
        """Test recipe testing with HTTP error."""
        # Mock HTTP error
        recipe_tester.http_client.get = AsyncMock(
            side_effect=httpx.RequestError("Connection failed")
        )
        
        result = await recipe_tester.test_recipe_on_url(sample_recipe, "https://example.com/test")
        
        assert result.success is False
        assert "HTTP request failed" in result.error_message
        assert result.extracted_data is None
    
    @pytest.mark.asyncio
    async def test_test_recipe_on_url_recipe_execution_error(self, recipe_tester, sample_recipe):
        """Test recipe testing with recipe execution error."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = "<html><body></body></html>"
        mock_response.raise_for_status = Mock()
        
        recipe_tester.http_client.get = AsyncMock(return_value=mock_response)
        
        # Mock recipe engine execution error
        from src.neurons.recipe_engine import RecipeExecutionError
        recipe_tester.recipe_engine.execute_recipe = AsyncMock(
            side_effect=RecipeExecutionError("Title selector not found")
        )
        
        result = await recipe_tester.test_recipe_on_url(sample_recipe, "https://example.com/test")
        
        assert result.success is False
        assert "Title selector not found" in result.error_message
        assert result.execution_time_ms is not None
        assert result.selectors_found is not None
    
    @pytest.mark.asyncio
    async def test_validate_recipe_basic_validation(self, recipe_tester):
        """Test basic recipe validation."""
        selectors = ScrapingSelectors(
            title="h1",
            content=".content"
        )
        actions = []
        
        # Mock recipe engine validation
        recipe_tester.recipe_engine.validate_recipe = Mock(return_value=(True, []))
        
        result = await recipe_tester.validate_recipe(selectors, actions)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        
        # Verify recipe engine was called
        recipe_tester.recipe_engine.validate_recipe.assert_called_once_with(selectors, actions)
    
    @pytest.mark.asyncio
    async def test_validate_recipe_with_errors(self, recipe_tester):
        """Test recipe validation with errors."""
        selectors = ScrapingSelectors(
            title="",  # Invalid empty selector
            content=".content"
        )
        actions = []
        
        # Mock recipe engine validation with errors
        recipe_tester.recipe_engine.validate_recipe = Mock(
            return_value=(False, ["Title selector is required"])
        )
        
        result = await recipe_tester.validate_recipe(selectors, actions)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Title selector is required" in result.errors[0]
    
    @pytest.mark.asyncio
    async def test_validate_recipe_with_test_urls(self, recipe_tester, sample_article):
        """Test recipe validation with test URLs."""
        selectors = ScrapingSelectors(
            title="h1",
            content=".content"
        )
        actions = []
        test_urls = ["https://example.com/test1", "https://example.com/test2"]
        
        # Mock recipe engine validation
        recipe_tester.recipe_engine.validate_recipe = Mock(return_value=(True, []))
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = "<html><body><h1>Test</h1></body></html>"
        mock_response.raise_for_status = Mock()
        
        recipe_tester.http_client.get = AsyncMock(return_value=mock_response)
        
        # Mock recipe engine execution
        recipe_tester.recipe_engine.execute_recipe = AsyncMock(return_value=sample_article)
        
        result = await recipe_tester.validate_recipe(selectors, actions, test_urls)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        
        # Verify HTTP client was called for test URLs
        assert recipe_tester.http_client.get.call_count == 2
    
    @pytest.mark.asyncio
    async def test_benchmark_recipe(self, recipe_tester, sample_recipe, sample_article):
        """Test recipe benchmarking."""
        test_urls = ["https://example.com/test1", "https://example.com/test2"]
        iterations = 2
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = "<html><body><h1>Test</h1></body></html>"
        mock_response.raise_for_status = Mock()
        
        recipe_tester.http_client.get = AsyncMock(return_value=mock_response)
        
        # Mock recipe engine execution
        recipe_tester.recipe_engine.execute_recipe = AsyncMock(return_value=sample_article)
        
        # Mock test_recipe_on_url to return successful results
        successful_result = RecipeTestResult(
            success=True,
            extracted_data={'title': 'Test'},
            execution_time_ms=100,
            content_quality_score=0.8
        )
        
        recipe_tester.test_recipe_on_url = AsyncMock(return_value=successful_result)
        
        result = await recipe_tester.benchmark_recipe(sample_recipe, test_urls, iterations)
        
        assert result['total_tests'] == 4  # 2 URLs * 2 iterations
        assert result['successful_tests'] == 4
        assert result['failed_tests'] == 0
        assert result['success_rate'] == 1.0
        assert result['average_execution_time_ms'] == 100
        assert len(result['url_results']) == 2
        
        # Verify test_recipe_on_url was called correct number of times
        assert recipe_tester.test_recipe_on_url.call_count == 4
    
    @pytest.mark.asyncio
    async def test_benchmark_recipe_with_failures(self, recipe_tester, sample_recipe):
        """Test recipe benchmarking with some failures."""
        test_urls = ["https://example.com/test1"]
        iterations = 2
        
        # Mock test_recipe_on_url to return mixed results
        successful_result = RecipeTestResult(
            success=True,
            execution_time_ms=100,
            content_quality_score=0.8
        )
        
        failed_result = RecipeTestResult(
            success=False,
            error_message="Selector not found",
            execution_time_ms=50
        )
        
        recipe_tester.test_recipe_on_url = AsyncMock(
            side_effect=[successful_result, failed_result]
        )
        
        result = await recipe_tester.benchmark_recipe(sample_recipe, test_urls, iterations)
        
        assert result['total_tests'] == 2
        assert result['successful_tests'] == 1
        assert result['failed_tests'] == 1
        assert result['success_rate'] == 0.5
        assert result['average_execution_time_ms'] == 75  # (100 + 50) / 2
        assert 'Selector not found' in result['error_summary']
    
    @pytest.mark.asyncio
    async def test_analyze_selectors(self, recipe_tester, sample_html):
        """Test selector analysis functionality."""
        selectors = ScrapingSelectors(
            title="h1.title",
            content=".content",
            author=".author",
            publish_date=".nonexistent"  # This selector won't find anything
        )
        
        result = await recipe_tester._analyze_selectors(selectors, sample_html)
        
        assert result['title'] is True
        assert result['content'] is True
        assert result['author'] is True
        assert result['publish_date'] is False  # Selector not found
    
    def test_analyze_selector_patterns(self, recipe_tester):
        """Test selector pattern analysis."""
        # Test overly specific selector
        selectors = ScrapingSelectors(
            title="html body div.container div.content div.article h1.title.main",  # Very specific
            content=".content"
        )
        
        result = recipe_tester._analyze_selector_patterns(selectors)
        
        assert len(result['warnings']) > 0
        assert any("very specific" in warning.lower() for warning in result['warnings'])
        assert len(result['suggestions']) > 0
    
    def test_analyze_selector_patterns_nth_child(self, recipe_tester):
        """Test selector pattern analysis with nth-child selectors."""
        selectors = ScrapingSelectors(
            title="div:nth-child(2) h1",  # Position-based selector
            content=".content"
        )
        
        result = recipe_tester._analyze_selector_patterns(selectors)
        
        assert len(result['warnings']) > 0
        assert any("position-based" in warning.lower() for warning in result['warnings'])
    
    def test_analyze_actions(self, recipe_tester):
        """Test action analysis functionality."""
        # Test many actions
        actions = [ScrapingAction(type="wait", timeout=1) for _ in range(15)]
        
        result = recipe_tester._analyze_actions(actions)
        
        assert len(result['warnings']) > 0
        assert any("many actions" in warning.lower() for warning in result['warnings'])
    
    def test_analyze_actions_long_wait(self, recipe_tester):
        """Test action analysis with long wait times."""
        actions = [
            ScrapingAction(type="wait", timeout=15)  # Long wait
        ]
        
        result = recipe_tester._analyze_actions(actions)
        
        assert len(result['warnings']) > 0
        assert any("long wait time" in warning.lower() for warning in result['warnings'])
    
    def test_calculate_content_quality_high_quality(self, recipe_tester):
        """Test content quality calculation for high-quality content."""
        article = ArticleCreate(
            url="https://example.com/test",
            title="This is a good title with appropriate length",
            content="This is a comprehensive article with substantial content. " * 20,  # Long content
            author="John Doe",
            published_at=datetime.utcnow(),
            source_domain="example.com"
        )
        
        score = recipe_tester._calculate_content_quality(article)
        
        assert score >= 0.9  # Should be high quality
    
    def test_calculate_content_quality_low_quality(self, recipe_tester):
        """Test content quality calculation for low-quality content."""
        article = ArticleCreate(
            url="https://example.com/test",
            title="Short",  # Very short title
            content="Short content.",  # Very short content
            source_domain="example.com"
        )
        
        score = recipe_tester._calculate_content_quality(article)
        
        assert score < 0.5  # Should be low quality
    
    @pytest.mark.asyncio
    async def test_close(self, recipe_tester):
        """Test closing the recipe tester."""
        recipe_tester.http_client.aclose = AsyncMock()
        
        await recipe_tester.close()
        
        recipe_tester.http_client.aclose.assert_called_once()


class TestRecipeTesterFactory:
    """Test cases for recipe tester factory function."""
    
    def test_create_recipe_tester(self, mock_recipe_engine):
        """Test recipe tester factory function."""
        tester = create_recipe_tester(mock_recipe_engine)
        
        assert isinstance(tester, RecipeTester)
        assert tester.recipe_engine == mock_recipe_engine


class TestRecipeTestResult:
    """Test cases for RecipeTestResult dataclass."""
    
    def test_recipe_test_result_creation(self):
        """Test creating RecipeTestResult."""
        result = RecipeTestResult(
            success=True,
            extracted_data={'title': 'Test'},
            execution_time_ms=100,
            content_quality_score=0.8
        )
        
        assert result.success is True
        assert result.extracted_data['title'] == 'Test'
        assert result.execution_time_ms == 100
        assert result.content_quality_score == 0.8


class TestRecipeValidationResult:
    """Test cases for RecipeValidationResult dataclass."""
    
    def test_recipe_validation_result_creation(self):
        """Test creating RecipeValidationResult."""
        result = RecipeValidationResult(
            is_valid=True,
            errors=[],
            warnings=['Minor issue'],
            suggestions=['Consider improvement']
        )
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert len(result.suggestions) == 1


if __name__ == "__main__":
    pytest.main([__file__])