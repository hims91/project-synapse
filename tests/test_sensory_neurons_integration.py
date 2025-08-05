"""
Integration tests for Sensory Neurons (Learning Scrapers).
Tests the complete learning scraper system with all components.
"""
import pytest
import asyncio
import tempfile
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from pathlib import Path

from src.sensory_neurons.playwright_scraper import PlaywrightScraper
from src.sensory_neurons.recipe_learner import RecipeLearner, LearningExample
from src.sensory_neurons.chameleon_network import ChameleonNetwork, ProxyInfo, ProxyType
from src.shared.schemas import ArticleCreate, ScrapingSelectors


@pytest.fixture
def sample_html():
    """Sample HTML for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Article - News Site</title>
        <meta name="description" content="This is a test article for learning validation.">
        <meta name="author" content="Test Author">
    </head>
    <body>
        <header>
            <nav>Navigation</nav>
        </header>
        <main>
            <article>
                <h1 class="article-title">Learning Test Article</h1>
                <div class="byline">By Test Author</div>
                <time datetime="2024-01-15T10:30:00Z">January 15, 2024</time>
                <div class="article-content">
                    <p>This is the first paragraph of the learning test article.</p>
                    <p>This paragraph contains substantial content for testing the learning algorithms.</p>
                    <p>The learning system should be able to extract this content effectively.</p>
                </div>
            </article>
        </main>
        <script>console.log('test');</script>
    </body>
    </html>
    """


class TestPlaywrightScraperIntegration:
    """Integration tests for Playwright scraper."""
    
    @pytest.mark.asyncio
    async def test_playwright_scraper_basic_functionality(self, sample_html):
        """Test basic Playwright scraper functionality."""
        # Mock Playwright components
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            # Setup mocks
            mock_playwright_instance = AsyncMock()
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()
            
            mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
            mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)
            
            # Mock page methods
            mock_page.goto = AsyncMock()
            mock_page.content = AsyncMock(return_value=sample_html)
            mock_page.wait_for_selector = AsyncMock()
            mock_page.wait_for_load_state = AsyncMock()
            mock_page.evaluate = AsyncMock(return_value={
                'totalElements': 10,
                'headings': {'h1': 1, 'h2': 0},
                'paragraphs': 3
            })
            mock_page.screenshot = AsyncMock(return_value=b'fake_screenshot_data')
            mock_page.close = AsyncMock()
            
            # Mock browser cleanup
            mock_browser.close = AsyncMock()
            mock_playwright_instance.stop = AsyncMock()
            
            # Test scraper
            scraper = PlaywrightScraper()
            
            try:
                await scraper.start()
                result = await scraper.scrape_url("https://example.com/test")
                
                # Verify result
                assert result is not None
                assert result.title == "Learning Test Article"
                assert "first paragraph of the learning test article" in result.content
                assert result.source_domain == "example.com"
                
                # Verify Playwright calls
                mock_page.goto.assert_called_once_with("https://example.com/test", wait_until="networkidle")
                mock_page.content.assert_called_once()
                
            finally:
                await scraper.close()
    
    @pytest.mark.asyncio
    async def test_playwright_scraper_with_actions(self, sample_html):
        """Test Playwright scraper with JavaScript actions."""
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            # Setup mocks
            mock_playwright_instance = AsyncMock()
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()
            
            mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
            mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)
            
            # Mock page methods
            mock_page.goto = AsyncMock()
            mock_page.content = AsyncMock(return_value=sample_html)
            mock_page.click = AsyncMock()
            mock_page.wait_for_timeout = AsyncMock()
            mock_page.scroll = AsyncMock()
            mock_page.fill = AsyncMock()
            mock_page.close = AsyncMock()
            mock_browser.close = AsyncMock()
            mock_playwright_instance.stop = AsyncMock()
            
            # Test scraper with actions
            scraper = PlaywrightScraper()
            
            from src.shared.schemas import ScrapingAction
            actions = [
                ScrapingAction(type="click", selector=".load-more", timeout=5),
                ScrapingAction(type="wait", timeout=2),
                ScrapingAction(type="scroll", value="bottom", timeout=3)
            ]
            
            try:
                await scraper.start()
                result = await scraper.scrape_url("https://example.com/test")
                
                # Verify actions were executed
                mock_page.click.assert_called_once_with(".load-more")
                mock_page.wait_for_timeout.assert_called_with(2000)
                
                assert result is not None
                
            finally:
                await scraper.close()


class TestRecipeLearnerIntegration:
    """Integration tests for recipe learning system."""
    
    @pytest.mark.asyncio
    async def test_recipe_learning_workflow(self, sample_html):
        """Test complete recipe learning workflow."""
        learner = RecipeLearner()
        
        # Create learning examples
        examples = [
            LearningExample(
                url="https://example.com/article1",
                html_content=sample_html,
                expected_title="Learning Test Article",
                expected_content="This is the first paragraph of the learning test article.",
                expected_author="Test Author"
            ),
            LearningExample(
                url="https://example.com/article2", 
                html_content=sample_html.replace("Learning Test Article", "Another Test Article"),
                expected_title="Another Test Article",
                expected_content="This is the first paragraph of the learning test article.",
                expected_author="Test Author"
            )
        ]
        
        # Test learning process
        learned_recipe = await learner.learn_from_examples("example.com", examples)
        
        # Verify learned recipe
        assert learned_recipe is not None
        assert learned_recipe.domain == "example.com"
        assert learned_recipe.selectors.title is not None
        assert learned_recipe.selectors.content is not None
        
        # Test recipe validation
        is_valid, errors = learner.validate_learned_recipe(learned_recipe, examples)
        assert is_valid is True
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_recipe_optimization(self, sample_html):
        """Test recipe optimization process."""
        learner = RecipeLearner()
        
        # Create initial recipe
        from src.shared.schemas import ScrapingSelectors
        initial_selectors = ScrapingSelectors(
            title="h1",  # Generic selector
            content=".content"  # Generic selector
        )
        
        # Create learning examples
        examples = [
            LearningExample(
                url="https://example.com/article1",
                html_content=sample_html,
                expected_title="Learning Test Article",
                expected_content="This is the first paragraph of the learning test article."
            )
        ]
        
        # Test optimization
        optimized_recipe = await learner.optimize_recipe("example.com", initial_selectors, examples)
        
        # Verify optimization improved specificity
        assert optimized_recipe is not None
        assert optimized_recipe.selectors.title != "h1"  # Should be more specific
        assert ".article-title" in optimized_recipe.selectors.title


class TestChameleonNetworkIntegration:
    """Integration tests for Chameleon Network proxy system."""
    
    @pytest.mark.asyncio
    async def test_proxy_rotation_workflow(self):
        """Test complete proxy rotation workflow."""
        # Mock proxy providers
        with patch('src.sensory_neurons.chameleon_network.aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                'ip': '192.168.1.100',
                'country': 'US',
                'region': 'California'
            })
            mock_response.text = AsyncMock(return_value='{"ip": "192.168.1.100"}')
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            network = ChameleonNetwork()
            
            # Add test proxies
            test_proxies = [
                ProxyInfo(
                    host="proxy1.example.com",
                    port=8080,
                    proxy_type=ProxyType.HTTP,
                    username="user1",
                    password="pass1"
                ),
                ProxyInfo(
                    host="proxy2.example.com", 
                    port=8080,
                    proxy_type=ProxyType.SOCKS5,
                    username="user2",
                    password="pass2"
                )
            ]
            
            for proxy in test_proxies:
                network.add_proxy(proxy)
            
            # Test proxy rotation
            proxy1 = await network.get_next_proxy()
            proxy2 = await network.get_next_proxy()
            
            assert proxy1 != proxy2  # Should rotate
            assert proxy1 in test_proxies
            assert proxy2 in test_proxies
            
            # Test proxy validation
            is_valid = await network.validate_proxy(proxy1)
            assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_proxy_failure_handling(self):
        """Test proxy failure detection and handling."""
        with patch('src.sensory_neurons.chameleon_network.aiohttp.ClientSession') as mock_session:
            # Mock failed proxy response
            mock_session.return_value.__aenter__.return_value.get.side_effect = Exception("Connection failed")
            
            network = ChameleonNetwork()
            
            # Add test proxy
            test_proxy = ProxyInfo(
                host="failing-proxy.example.com",
                port=8080,
                proxy_type=ProxyType.HTTP
            )
            network.add_proxy(test_proxy)
            
            # Test failure handling
            is_valid = await network.validate_proxy(test_proxy)
            assert is_valid is False
            
            # Verify proxy is marked as failed
            assert test_proxy.failure_count > 0


class TestSensoryNeuronsFullIntegration:
    """Full integration tests combining all sensory neuron components."""
    
    @pytest.mark.asyncio
    async def test_complete_learning_scraper_workflow(self, sample_html):
        """Test complete workflow from scraping to learning to recipe generation."""
        # Mock all external dependencies
        with patch('playwright.async_api.async_playwright') as mock_playwright, \
             patch('src.sensory_neurons.chameleon_network.aiohttp.ClientSession') as mock_session:
            
            # Setup Playwright mocks
            mock_playwright_instance = AsyncMock()
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()
            
            mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
            mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)
            
            mock_page.goto = AsyncMock()
            mock_page.content = AsyncMock(return_value=sample_html)
            mock_page.close = AsyncMock()
            mock_browser.close = AsyncMock()
            mock_playwright_instance.stop = AsyncMock()
            
            # Setup proxy mocks
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'ip': '192.168.1.100'})
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # Initialize components
            scraper = PlaywrightScraper()
            learner = RecipeLearner()
            network = ChameleonNetwork()
            
            try:
                # Step 1: Scrape with browser automation
                await scraper.start()
                article = await scraper.scrape_url("https://example.com/test")
                assert article is not None
                assert article.title == "Learning Test Article"
                
                # Step 2: Create learning example from scraped data
                learning_example = LearningExample(
                    url="https://example.com/test",
                    html_content=sample_html,
                    expected_title=article.title,
                    expected_content=article.content,
                    expected_author=article.author
                )
                
                # Step 3: Learn recipe from example
                learned_recipe = await learner.learn_from_examples("example.com", [learning_example])
                assert learned_recipe is not None
                assert learned_recipe.domain == "example.com"
                
                # Step 4: Validate learned recipe
                is_valid, errors = learner.validate_learned_recipe(learned_recipe, [learning_example])
                assert is_valid is True
                
                # Step 5: Test proxy integration
                test_proxy = ProxyInfo(
                    host="proxy.example.com",
                    port=8080,
                    proxy_type=ProxyType.HTTP
                )
                network.add_proxy(test_proxy)
                
                proxy = await network.get_next_proxy()
                assert proxy is not None
                
            finally:
                await scraper.close()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery across all components."""
        # Test Playwright scraper error handling
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright.return_value.start.side_effect = Exception("Playwright failed to start")
            
            scraper = PlaywrightScraper()
            
            with pytest.raises(Exception):
                await scraper.start()
                await scraper.scrape_url("https://example.com/test")
        
        # Test recipe learner error handling
        learner = RecipeLearner()
        
        # Test with invalid examples
        invalid_examples = [
            LearningExample(
                url="https://example.com/test",
                html_content="<html><body></body></html>",  # No content
                expected_title="Title",
                expected_content="Content"
            )
        ]
        
        learned_recipe = await learner.learn_from_examples("example.com", invalid_examples)
        # Should handle gracefully and return basic recipe or None
        
        # Test proxy network error handling
        network = ChameleonNetwork()
        
        # Test with no proxies
        proxy = await network.get_next_proxy()
        assert proxy is None  # Should handle gracefully
    
    @pytest.mark.asyncio
    async def test_performance_and_concurrency(self, sample_html):
        """Test performance and concurrent operations."""
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            # Setup mocks for concurrent operations
            mock_playwright_instance = AsyncMock()
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            
            mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
            mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            
            # Mock multiple pages for concurrent scraping
            mock_pages = []
            for i in range(5):
                mock_page = AsyncMock()
                mock_page.goto = AsyncMock()
                mock_page.content = AsyncMock(return_value=sample_html)
                mock_page.close = AsyncMock()
                mock_pages.append(mock_page)
            
            mock_context.new_page = AsyncMock(side_effect=mock_pages)
            mock_browser.close = AsyncMock()
            mock_playwright_instance.stop = AsyncMock()
            
            scraper = PlaywrightScraper()
            
            try:
                # Test concurrent scraping
                await scraper.start()
                urls = [f"https://example.com/test{i}" for i in range(5)]
                
                tasks = [scraper.scrape_url(url) for url in urls]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Verify all succeeded
                successful_results = [r for r in results if not isinstance(r, Exception)]
                assert len(successful_results) == 5
                
                # Verify all have expected content
                for result in successful_results:
                    assert result.title == "Learning Test Article"
                
            finally:
                await scraper.close()


if __name__ == "__main__":
    pytest.main([__file__])