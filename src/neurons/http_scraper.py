"""
Neurons - HTTP Scraper with httpx
Layer 1: Perception Layer

This module implements the HTTP scraper using httpx for async content retrieval.
It integrates with the recipe engine for content extraction and provides
comprehensive error handling for network failures and parsing errors.
"""
import asyncio
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from urllib.parse import urlparse, urljoin
import random

import httpx
from bs4 import BeautifulSoup

from ..shared.schemas import ArticleCreate, ScrapingRecipeResponse
from .recipe_engine import ScrapingRecipeEngine, RecipeExecutionError, recipe_engine

logger = logging.getLogger(__name__)


class ScrapingError(Exception):
    """Base exception for scraping errors."""
    pass


class NetworkError(ScrapingError):
    """Raised when network requests fail."""
    pass


class ContentExtractionError(ScrapingError):
    """Raised when content extraction fails."""
    pass


class HTTPScraper:
    """
    HTTP scraper using httpx for async content retrieval.
    
    Integrates with the recipe engine for intelligent content extraction
    and provides comprehensive error handling for network and parsing failures.
    """
    
    # User agents for rotation to avoid detection
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59'
    ]
    
    def __init__(
        self, 
        recipe_engine: Optional[ScrapingRecipeEngine] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize HTTP scraper.
        
        Args:
            recipe_engine: Recipe engine for content extraction
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Base delay between retries in seconds
        """
        self.recipe_engine = recipe_engine or recipe_engine
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # HTTP client configuration
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        )
    
    async def scrape_url(self, url: str, use_recipe: bool = True) -> ArticleCreate:
        """
        Scrape content from a URL.
        
        Args:
            url: URL to scrape
            use_recipe: Whether to use cached recipes for extraction
            
        Returns:
            Extracted article data
            
        Raises:
            NetworkError: If network request fails
            ContentExtractionError: If content extraction fails
        """
        try:
            logger.info(f"Starting scrape for URL: {url}")
            
            # Fetch HTML content
            html_content = await self._fetch_html(url)
            
            # Extract content using recipe or fallback
            if use_recipe:
                article_data = await self._extract_with_recipe(url, html_content)
            else:
                article_data = await self._extract_fallback(url, html_content)
            
            logger.info(f"Successfully scraped URL: {url}")
            return article_data
            
        except httpx.RequestError as e:
            logger.error(f"Network error scraping {url}: {e}")
            raise NetworkError(f"Failed to fetch {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {e}")
            raise ContentExtractionError(f"Failed to extract content from {url}: {e}")
    
    async def scrape_multiple(
        self, 
        urls: List[str], 
        use_recipe: bool = True,
        max_concurrent: int = 10
    ) -> List[Tuple[str, Optional[ArticleCreate], Optional[Exception]]]:
        """
        Scrape multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            use_recipe: Whether to use cached recipes
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of tuples (url, article_data, error)
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_single(url: str) -> Tuple[str, Optional[ArticleCreate], Optional[Exception]]:
            async with semaphore:
                try:
                    article_data = await self.scrape_url(url, use_recipe)
                    return url, article_data, None
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
                    return url, None, e
        
        logger.info(f"Starting concurrent scrape of {len(urls)} URLs")
        results = await asyncio.gather(*[scrape_single(url) for url in urls])
        
        successful = sum(1 for _, article, _ in results if article is not None)
        logger.info(f"Completed scraping: {successful}/{len(urls)} successful")
        
        return results
    
    async def test_scraping_capability(self, url: str) -> Dict[str, Any]:
        """
        Test scraping capability for a URL without full extraction.
        
        Args:
            url: URL to test
            
        Returns:
            Dictionary with test results
        """
        try:
            start_time = datetime.utcnow()
            
            # Test basic connectivity
            response = await self.client.head(url, headers=self._get_headers())
            response.raise_for_status()
            
            # Test content retrieval
            html_content = await self._fetch_html(url)
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Analyze content
            title_candidates = self._find_title_candidates(soup)
            content_candidates = self._find_content_candidates(soup)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                'success': True,
                'url': url,
                'status_code': response.status_code,
                'content_length': len(html_content),
                'title_candidates': len(title_candidates),
                'content_candidates': len(content_candidates),
                'execution_time_seconds': execution_time,
                'has_javascript': 'script' in html_content.lower(),
                'has_forms': bool(soup.find('form')),
                'meta_robots': soup.find('meta', attrs={'name': 'robots'}),
                'content_type': response.headers.get('content-type', ''),
            }
            
        except Exception as e:
            return {
                'success': False,
                'url': url,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    async def _fetch_html(self, url: str) -> str:
        """
        Fetch HTML content from URL with retries.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content as string
            
        Raises:
            httpx.RequestError: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                headers = self._get_headers()
                
                logger.debug(f"Fetching {url} (attempt {attempt + 1}/{self.max_retries + 1})")
                
                response = await self.client.get(url, headers=headers)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                    logger.warning(f"Unexpected content type for {url}: {content_type}")
                
                return response.text
                
            except httpx.RequestError as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Request failed for {url}, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed for {url}: {e}")
        
        raise last_exception
    
    async def _extract_with_recipe(self, url: str, html_content: str) -> ArticleCreate:
        """
        Extract content using cached recipe.
        
        Args:
            url: Source URL
            html_content: HTML content
            
        Returns:
            Extracted article data
        """
        domain = urlparse(url).netloc
        
        # Try to get cached recipe
        recipe = await self.recipe_engine.get_recipe(domain)
        
        if recipe:
            try:
                # Execute recipe
                article_data = await self.recipe_engine.execute_recipe(
                    recipe, html_content, url
                )
                
                # Update recipe success
                await self.recipe_engine.update_recipe_success(domain, True)
                
                logger.info(f"Successfully used recipe for {domain}")
                return article_data
                
            except RecipeExecutionError as e:
                logger.warning(f"Recipe execution failed for {domain}: {e}")
                
                # Update recipe failure
                await self.recipe_engine.update_recipe_success(domain, False)
                
                # Fall back to generic extraction
                return await self._extract_fallback(url, html_content)
        else:
            logger.info(f"No recipe found for {domain}, using fallback extraction")
            return await self._extract_fallback(url, html_content)
    
    async def _extract_fallback(self, url: str, html_content: str) -> ArticleCreate:
        """
        Extract content using fallback heuristics.
        
        Args:
            url: Source URL
            html_content: HTML content
            
        Returns:
            Extracted article data
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        domain = urlparse(url).netloc
        
        # Extract title
        title = self._extract_title(soup)
        if not title:
            raise ContentExtractionError("Could not extract title")
        
        # Extract content
        content = self._extract_content(soup)
        if not content:
            raise ContentExtractionError("Could not extract content")
        
        # Extract optional fields
        author = self._extract_author(soup)
        published_at = self._extract_publish_date(soup)
        summary = self._extract_summary(soup, content)
        
        return ArticleCreate(
            url=url,
            title=title,
            content=content,
            summary=summary,
            author=author,
            published_at=published_at,
            source_domain=domain
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with random user agent."""
        headers = {
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        return headers
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract title using multiple strategies."""
        # Try common title selectors in order of preference
        selectors = [
            'h1.entry-title',
            'h1.post-title',
            'h1.article-title',
            'h1[class*="title"]',
            '.entry-title',
            '.post-title',
            '.article-title',
            'h1',
            'title'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                title = self._clean_text(element.get_text())
                if title and len(title) > 5:  # Minimum title length
                    return title
        
        return None
    
    def _extract_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract main content using multiple strategies."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()
        
        # Try common content selectors
        selectors = [
            '.entry-content',
            '.post-content',
            '.article-content',
            '.content',
            '[class*="content"]',
            '.post-body',
            '.article-body',
            'main',
            'article'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                content = self._clean_text(element.get_text())
                if content and len(content) > 100:  # Minimum content length
                    return content
        
        # Fallback: try to find the largest text block
        text_blocks = []
        for p in soup.find_all('p'):
            text = self._clean_text(p.get_text())
            if text and len(text) > 50:
                text_blocks.append(text)
        
        if text_blocks:
            return '\n\n'.join(text_blocks)
        
        return None
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author using multiple strategies."""
        selectors = [
            '.author',
            '.byline',
            '.post-author',
            '.article-author',
            '[rel="author"]',
            '[class*="author"]',
            '.by-author'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                author = self._clean_text(element.get_text())
                if author and len(author) < 100:  # Reasonable author length
                    # Clean up common prefixes
                    author = author.replace('By ', '').replace('by ', '').strip()
                    return author
        
        # Try meta tags
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author and meta_author.get('content'):
            return meta_author['content'].strip()
        
        return None
    
    def _extract_publish_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract publish date using multiple strategies."""
        # Try time elements
        time_element = soup.find('time')
        if time_element:
            datetime_attr = time_element.get('datetime')
            if datetime_attr:
                try:
                    from dateutil import parser
                    return parser.parse(datetime_attr)
                except Exception:
                    pass
        
        # Try meta tags
        meta_selectors = [
            {'property': 'article:published_time'},
            {'name': 'publishdate'},
            {'name': 'date'},
            {'name': 'DC.date.issued'},
            {'property': 'og:published_time'}
        ]
        
        for meta_attrs in meta_selectors:
            meta_element = soup.find('meta', attrs=meta_attrs)
            if meta_element and meta_element.get('content'):
                try:
                    from dateutil import parser
                    return parser.parse(meta_element['content'])
                except Exception:
                    continue
        
        return None
    
    def _extract_summary(self, soup: BeautifulSoup, content: str) -> Optional[str]:
        """Extract summary using multiple strategies."""
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            summary = meta_desc['content'].strip()
            if summary and len(summary) > 20:
                return summary
        
        # Try Open Graph description
        og_desc = soup.find('meta', attrs={'property': 'og:description'})
        if og_desc and og_desc.get('content'):
            summary = og_desc['content'].strip()
            if summary and len(summary) > 20:
                return summary
        
        # Fallback: use first paragraph of content
        if content:
            sentences = content.split('.')[:3]  # First 3 sentences
            summary = '.'.join(sentences).strip()
            if len(summary) > 20:
                return summary + '.'
        
        return None
    
    def _find_title_candidates(self, soup: BeautifulSoup) -> List[str]:
        """Find potential title elements."""
        candidates = []
        
        for tag in ['h1', 'h2', 'title']:
            elements = soup.find_all(tag)
            for element in elements:
                text = self._clean_text(element.get_text())
                if text and len(text) > 5:
                    candidates.append(text)
        
        return candidates
    
    def _find_content_candidates(self, soup: BeautifulSoup) -> List[str]:
        """Find potential content elements."""
        candidates = []
        
        # Look for common content containers
        for selector in ['.content', '.post', '.article', 'main', 'article']:
            elements = soup.select(selector)
            for element in elements:
                text = self._clean_text(element.get_text())
                if text and len(text) > 100:
                    candidates.append(text[:200] + '...')  # Truncate for analysis
        
        return candidates
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        cleaned = " ".join(text.split())
        return cleaned.strip()
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# Global scraper instance
http_scraper = HTTPScraper()