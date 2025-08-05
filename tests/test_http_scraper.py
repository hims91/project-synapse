"""
Integration tests for the HTTP Scraper.
Tests HTTP scraping functionality with various website types and error scenarios.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

import httpx

from src.neurons.http_scraper import (
    HTTPScraper, NetworkError, ContentExtractionError
)
from src.neurons.recipe_engine import ScrapingRecipeEngine
from src.shared.schemas import ArticleCreate, ScrapingRecipeResponse, ScrapingSelectors


@pytest.fixture
def mock_recipe_engine():
    """Create a mock recipe engine for testing."""
    return Mock(spec=ScrapingRecipeEngine)


@pytest.fixture
def http_scraper(mock_recipe_engine):
    """Create HTTP scraper instance for testing."""
    return HTTPScraper(recipe_engine=mock_recipe_engine, timeout=5.0)


@pytest.fixture
def sample_html():
    """Sample HTML content for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Article - Sample News Site</title>
        <meta name="description" content="This is a test article for scraping validation.">
        <meta name="author" content="John Doe">
        <meta property="article:published_time" content="2024-01-15T10:30:00Z">
    </head>
    <body>
        <header>
            <nav>Navigation menu</nav>
        </header>
        <main>
            <article>
                <h1 class="article-title">Test Article Title</h1>
                <div class="byline">By John Doe</div>
                <time datetime="2024-01-15T10:30:00Z">January 15, 2024</time>
                <div class="article-content">
                    <p>This is the first paragraph of the test article. It contains substantial content to test the scraping functionality.</p>
                    <p>This is the second paragraph with more detailed information about the topic being discussed.</p>
                    <p>The third paragraph continues the discussion and provides additional context for the article content.</p>
                </div>
            </article>
        </main>
        <footer>Footer content</footer>
        <script>console.log('test');</script>
    </body>
    </html>
    """


@pytest.fixture
def minimal_html():
    """Minimal HTML content for testing."""
    return """
    <html>
    <head><title>Minimal Page</title></head>
    <body>
        <h1>Simple Title</h1>
        <p>Short content that might not meet minimum requirements.</p>
    </body>
    </html>
    """


class TestHTTPScraper:
    """Test cases for HTTPScraper."""
    
    @pytest.mark.asyncio
    async def test_scrape_url_with_recipe_success(self, http_scraper, sample_html):
        """Test successful URL scraping using cached recipe."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = sample_html
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.raise_for_status = Mock()
        
        http_scraper.client.get = AsyncMock(return_value=mock_response)
        
        # Mock recipe engine
        sample_recipe = Mock(spec=ScrapingRecipeResponse)
        http_scraper.recipe_engine.get_recipe = AsyncMock(return_value=sample_recipe)
        
        sample_article = ArticleCreate(
            url="https://example.com/article",
            title="Test Article Title",
            content="Article content here",
            source_domain="example.com"
        )
        http_scraper.recipe_engine.execute_recipe = AsyncMock(return_value=sample_article)
        http_scraper.recipe_engine.update_recipe_success = AsyncMock()
        
        # Test scraping
        result = await http_scraper.scrape_url("https://example.com/article")
        
        assert result.title == "Test Article Title"
        assert result.content == "Article content here"
        assert result.source_domain == "example.com"
        
        # Verify recipe engine calls
        http_scraper.recipe_engine.get_recipe.assert_called_once_with("example.com")
        http_scraper.recipe_engine.execute_recipe.assert_called_once()
        http_scraper.recipe_engine.update_recipe_success.assert_called_once_with("example.com", True)
    
    @pytest.mark.asyncio
    async def test_scrape_url_recipe_failure_fallback(self, http_scraper, sample_html):
        """Test fallback to generic extraction when recipe fails."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = sample_html
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.raise_for_status = Mock()
        
        http_scraper.client.get = AsyncMock(return_value=mock_response)
        
        # Mock recipe engine with failure
        sample_recipe = Mock(spec=ScrapingRecipeResponse)
        http_scraper.recipe_engine.get_recipe = AsyncMock(return_value=sample_recipe)
        
        from src.neurons.recipe_engine import RecipeExecutionError
        http_scraper.recipe_engine.execute_recipe = AsyncMock(
            side_effect=RecipeExecutionError("Recipe failed")
        )
        http_scraper.recipe_engine.update_recipe_success = AsyncMock()
        
        # Test scraping
        result = await http_scraper.scrape_url("https://example.com/article")
        
        assert result.title == "Test Article Title"
        assert "first paragraph of the test article" in result.content
        assert result.source_domain == "example.com"
        assert result.author == "John Doe"
        
        # Verify recipe failure was recorded
        http_scraper.recipe_engine.update_recipe_success.assert_called_once_with("example.com", False)
    
    @pytest.mark.asyncio
    async def test_scrape_url_no_recipe_fallback(self, http_scraper, sample_html):
        """Test fallback extraction when no recipe exists."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = sample_html
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.raise_for_status = Mock()
        
        http_scraper.client.get = AsyncMock(return_value=mock_response)
        
        # Mock recipe engine with no recipe
        http_scraper.recipe_engine.get_recipe = AsyncMock(return_value=None)
        
        # Test scraping
        result = await http_scraper.scrape_url("https://example.com/article")
        
        assert result.title == "Test Article Title"
        assert "first paragraph of the test article" in result.content
        assert result.source_domain == "example.com"
        assert result.author == "John Doe"
        assert result.published_at is not None
        assert result.summary == "This is a test article for scraping validation."
    
    @pytest.mark.asyncio
    async def test_scrape_url_network_error(self, http_scraper):
        """Test handling of network errors."""
        # Mock network error
        http_scraper.client.get = AsyncMock(
            side_effect=httpx.RequestError("Connection failed")
        )
        
        with pytest.raises(NetworkError) as exc_info:
            await http_scraper.scrape_url("https://example.com/article")
        
        assert "Failed to fetch" in str(exc_info.value)
        assert "Connection failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_scrape_url_http_error(self, http_scraper):
        """Test handling of HTTP errors."""
        # Mock HTTP error response
        mock_response = Mock()
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError("404 Not Found", request=Mock(), response=Mock())
        )
        
        http_scraper.client.get = AsyncMock(return_value=mock_response)
        
        with pytest.raises(NetworkError):
            await http_scraper.scrape_url("https://example.com/nonexistent")
    
    @pytest.mark.asyncio
    async def test_scrape_url_content_extraction_error(self, http_scraper):
        """Test handling of content extraction errors."""
        # Mock response with no extractable content
        empty_html = "<html><head><title></title></head><body></body></html>"
        mock_response = Mock()
        mock_response.text = empty_html
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.raise_for_status = Mock()
        
        http_scraper.client.get = AsyncMock(return_value=mock_response)
        http_scraper.recipe_engine.get_recipe = AsyncMock(return_value=None)
        
        with pytest.raises(ContentExtractionError) as exc_info:
            await http_scraper.scrape_url("https://example.com/empty")
        
        assert "Could not extract title" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_scrape_url_without_recipe(self, http_scraper, sample_html):
        """Test scraping without using recipes."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = sample_html
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.raise_for_status = Mock()
        
        http_scraper.client.get = AsyncMock(return_value=mock_response)
        
        # Test scraping without recipe
        result = await http_scraper.scrape_url("https://example.com/article", use_recipe=False)
        
        assert result.title == "Test Article Title"
        assert "first paragraph of the test article" in result.content
        assert result.source_domain == "example.com"
        
        # Verify recipe engine was not called
        http_scraper.recipe_engine.get_recipe.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_scrape_multiple_urls(self, http_scraper, sample_html):
        """Test concurrent scraping of multiple URLs."""
        urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3"
        ]
        
        # Mock HTTP responses
        mock_response = Mock()
        mock_response.text = sample_html
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.raise_for_status = Mock()
        
        http_scraper.client.get = AsyncMock(return_value=mock_response)
        http_scraper.recipe_engine.get_recipe = AsyncMock(return_value=None)
        
        # Test multiple scraping
        results = await http_scraper.scrape_multiple(urls, max_concurrent=2)
        
        assert len(results) == 3
        
        for url, article, error in results:
            assert url in urls
            assert article is not None
            assert error is None
            assert article.title == "Test Article Title"
    
    @pytest.mark.asyncio
    async def test_scrape_multiple_with_failures(self, http_scraper, sample_html):
        """Test concurrent scraping with some failures."""
        urls = [
            "https://example.com/success",
            "https://example.com/failure",
            "https://example.com/success2"
        ]
        
        async def mock_get(url, **kwargs):
            if "failure" in url:
                raise httpx.RequestError("Connection failed")
            
            mock_response = Mock()
            mock_response.text = sample_html
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.raise_for_status = Mock()
            return mock_response
        
        http_scraper.client.get = AsyncMock(side_effect=mock_get)
        http_scraper.recipe_engine.get_recipe = AsyncMock(return_value=None)
        
        # Test multiple scraping with failures
        results = await http_scraper.scrape_multiple(urls)
        
        assert len(results) == 3
        
        success_count = sum(1 for _, article, _ in results if article is not None)
        failure_count = sum(1 for _, _, error in results if error is not None)
        
        assert success_count == 2
        assert failure_count == 1
    
    @pytest.mark.asyncio
    async def test_test_scraping_capability_success(self, http_scraper, sample_html):
        """Test scraping capability testing."""
        # Mock HEAD request
        head_response = Mock()
        head_response.status_code = 200
        head_response.headers = {'content-type': 'text/html'}
        head_response.raise_for_status = Mock()
        
        # Mock GET request
        get_response = Mock()
        get_response.text = sample_html
        get_response.status_code = 200
        get_response.headers = {'content-type': 'text/html'}
        get_response.raise_for_status = Mock()
        
        http_scraper.client.head = AsyncMock(return_value=head_response)
        http_scraper.client.get = AsyncMock(return_value=get_response)
        
        # Test capability
        result = await http_scraper.test_scraping_capability("https://example.com/test")
        
        assert result['success'] is True
        assert result['url'] == "https://example.com/test"
        assert result['status_code'] == 200
        assert result['content_length'] > 0
        assert result['title_candidates'] > 0
        assert result['content_candidates'] > 0
        assert result['has_javascript'] is True
        assert 'execution_time_seconds' in result
    
    @pytest.mark.asyncio
    async def test_test_scraping_capability_failure(self, http_scraper):
        """Test scraping capability testing with failure."""
        # Mock network error
        http_scraper.client.head = AsyncMock(
            side_effect=httpx.RequestError("Connection failed")
        )
        
        # Test capability
        result = await http_scraper.test_scraping_capability("https://example.com/test")
        
        assert result['success'] is False
        assert result['url'] == "https://example.com/test"
        assert 'error' in result
        assert result['error_type'] == 'RequestError'
    
    @pytest.mark.asyncio
    async def test_fetch_html_with_retries(self, http_scraper, sample_html):
        """Test HTML fetching with retry logic."""
        # Mock first two requests fail, third succeeds
        call_count = 0
        
        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count <= 2:
                raise httpx.RequestError("Temporary failure")
            
            mock_response = Mock()
            mock_response.text = sample_html
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.raise_for_status = Mock()
            return mock_response
        
        http_scraper.client.get = AsyncMock(side_effect=mock_get)
        
        # Test with retries
        result = await http_scraper._fetch_html("https://example.com/test")
        
        assert result == sample_html
        assert call_count == 3  # Two failures + one success
    
    @pytest.mark.asyncio
    async def test_fetch_html_all_retries_fail(self, http_scraper):
        """Test HTML fetching when all retries fail."""
        # Mock all requests fail
        http_scraper.client.get = AsyncMock(
            side_effect=httpx.RequestError("Persistent failure")
        )
        
        with pytest.raises(httpx.RequestError):
            await http_scraper._fetch_html("https://example.com/test")
        
        # Verify all retry attempts were made
        assert http_scraper.client.get.call_count == http_scraper.max_retries + 1
    
    def test_extract_title_various_selectors(self, http_scraper):
        """Test title extraction with various HTML structures."""
        test_cases = [
            ('<h1 class="entry-title">Entry Title</h1>', "Entry Title"),
            ('<h1 class="post-title">Post Title</h1>', "Post Title"),
            ('<h1 class="article-title">Article Title</h1>', "Article Title"),
            ('<h1>Simple H1 Title</h1>', "Simple H1 Title"),
            ('<title>Page Title</title>', "Page Title"),
            ('<div class="entry-title">Div Title</div>', "Div Title"),
        ]
        
        for html, expected_title in test_cases:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            result = http_scraper._extract_title(soup)
            assert result == expected_title
    
    def test_extract_content_various_selectors(self, http_scraper):
        """Test content extraction with various HTML structures."""
        test_cases = [
            (
                '<div class="entry-content"><p>This is a long enough content paragraph that should be extracted successfully by the content extraction algorithm.</p></div>',
                "This is a long enough content paragraph that should be extracted successfully by the content extraction algorithm."
            ),
            (
                '<div class="post-content"><p>Another long content paragraph that meets the minimum length requirements for successful extraction.</p></div>',
                "Another long content paragraph that meets the minimum length requirements for successful extraction."
            ),
            (
                '<main><p>Main content paragraph that is sufficiently long to be considered valid content for extraction purposes.</p></main>',
                "Main content paragraph that is sufficiently long to be considered valid content for extraction purposes."
            ),
        ]
        
        for html, expected_content in test_cases:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            result = http_scraper._extract_content(soup)
            assert expected_content in result
    
    def test_extract_author_various_selectors(self, http_scraper):
        """Test author extraction with various HTML structures."""
        test_cases = [
            ('<div class="author">John Doe</div>', "John Doe"),
            ('<div class="byline">By Jane Smith</div>', "By Jane Smith"),
            ('<span class="post-author">Author Name</span>', "Author Name"),
            ('<a rel="author">Link Author</a>', "Link Author"),
            ('<meta name="author" content="Meta Author">', "Meta Author"),
        ]
        
        for html, expected_author in test_cases:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            result = http_scraper._extract_author(soup)
            assert result == expected_author
    
    def test_extract_publish_date_various_formats(self, http_scraper):
        """Test publish date extraction with various formats."""
        test_cases = [
            '<time datetime="2024-01-15T10:30:00Z">January 15, 2024</time>',
            '<meta property="article:published_time" content="2024-01-15T10:30:00Z">',
            '<meta name="publishdate" content="2024-01-15">',
            '<meta name="date" content="January 15, 2024">',
        ]
        
        for html in test_cases:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            result = http_scraper._extract_publish_date(soup)
            
            # Should extract some date (exact format may vary)
            if result:  # Only check if dateutil is available
                assert result.year == 2024
                assert result.month == 1
                assert result.day == 15
    
    def test_extract_summary_various_sources(self, http_scraper):
        """Test summary extraction from various sources."""
        test_cases = [
            ('<meta name="description" content="Meta description summary">', "Meta description summary"),
            ('<meta property="og:description" content="OpenGraph description">', "OpenGraph description"),
        ]
        
        for html, expected_summary in test_cases:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            result = http_scraper._extract_summary(soup, "")
            assert result == expected_summary
    
    def test_clean_text(self, http_scraper):
        """Test text cleaning functionality."""
        test_cases = [
            ("  Hello   World  ", "Hello World"),
            ("\n\t  Text  \n\t", "Text"),
            ("Multiple\n\nLine\n\nText", "Multiple Line Text"),
            ("", ""),
        ]
        
        for input_text, expected_output in test_cases:
            result = http_scraper._clean_text(input_text)
            assert result == expected_output
    
    def test_get_headers_user_agent_rotation(self, http_scraper):
        """Test user agent rotation in headers."""
        headers1 = http_scraper._get_headers()
        headers2 = http_scraper._get_headers()
        
        # Headers should contain User-Agent
        assert 'User-Agent' in headers1
        assert 'User-Agent' in headers2
        
        # User agents should be from the predefined list
        assert headers1['User-Agent'] in http_scraper.USER_AGENTS
        assert headers2['User-Agent'] in http_scraper.USER_AGENTS
    
    @pytest.mark.asyncio
    async def test_close(self, http_scraper):
        """Test closing the HTTP scraper."""
        http_scraper.client.aclose = AsyncMock()
        
        await http_scraper.close()
        
        http_scraper.client.aclose.assert_called_once()


class TestHTTPScraperIntegration:
    """Integration tests for HTTP scraper."""
    
    @pytest.mark.asyncio
    async def test_full_scraping_workflow(self, http_scraper, sample_html):
        """Test complete scraping workflow from URL to article."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = sample_html
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.raise_for_status = Mock()
        
        http_scraper.client.get = AsyncMock(return_value=mock_response)
        http_scraper.recipe_engine.get_recipe = AsyncMock(return_value=None)
        
        # Test complete workflow
        result = await http_scraper.scrape_url("https://example.com/article")
        
        # Verify all fields were extracted
        assert result.url == "https://example.com/article"
        assert result.title == "Test Article Title"
        assert "first paragraph of the test article" in result.content
        assert result.author == "John Doe"
        assert result.source_domain == "example.com"
        assert result.summary == "This is a test article for scraping validation."
        
        # Verify publish date was extracted (if dateutil available)
        if result.published_at:
            assert result.published_at.year == 2024
            assert result.published_at.month == 1
            assert result.published_at.day == 15


if __name__ == "__main__":
    pytest.main([__file__])