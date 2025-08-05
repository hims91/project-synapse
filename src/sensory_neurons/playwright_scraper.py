"""
Sensory Neurons - Playwright Browser Automation
Layer 1: Perception Layer

This module implements advanced browser automation using Playwright for JavaScript-heavy sites.
It provides screenshot capabilities, DOM analysis, and sophisticated anti-bot detection evasion.
"""
import asyncio
import logging
import random
import base64
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from urllib.parse import urlparse
from pathlib import Path

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError

from ..shared.schemas import ArticleCreate, ScrapingRecipeResponse
from ..neurons.recipe_engine import ScrapingRecipeEngine, recipe_engine

logger = logging.getLogger(__name__)


class BrowserAutomationError(Exception):
    """Base exception for browser automation errors."""
    pass


class AntiDetectionError(BrowserAutomationError):
    """Raised when anti-bot detection is triggered."""
    pass


class JavaScriptError(BrowserAutomationError):
    """Raised when JavaScript execution fails."""
    pass


class PlaywrightScraper:
    """
    Advanced browser automation scraper using Playwright.
    
    Provides JavaScript execution, screenshot capabilities, DOM analysis,
    and sophisticated anti-bot detection evasion techniques.
    """
    
    # User agents for rotation
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36'
    ]
    
    # Screen resolutions for rotation
    SCREEN_SIZES = [
        {'width': 1920, 'height': 1080},
        {'width': 1366, 'height': 768},
        {'width': 1536, 'height': 864},
        {'width': 1440, 'height': 900},
        {'width': 1280, 'height': 720},
        {'width': 1600, 'height': 900},
        {'width': 2560, 'height': 1440},
        {'width': 1920, 'height': 1200}
    ]
    
    def __init__(
        self,
        recipe_engine: Optional[ScrapingRecipeEngine] = None,
        headless: bool = True,
        timeout: float = 30000,  # Playwright uses milliseconds
        screenshot_dir: Optional[str] = None,
        enable_stealth: bool = True
    ):
        """
        Initialize Playwright scraper.
        
        Args:
            recipe_engine: Recipe engine for content extraction
            headless: Whether to run browser in headless mode
            timeout: Page load timeout in milliseconds
            screenshot_dir: Directory to save screenshots
            enable_stealth: Whether to enable stealth mode
        """
        self.recipe_engine = recipe_engine or recipe_engine
        self.headless = headless
        self.timeout = timeout
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else Path("screenshots")
        self.enable_stealth = enable_stealth
        
        # Browser instances
        self.playwright = None
        self.browser = None
        self.context = None
        
        # Anti-detection settings
        self.stealth_settings = {
            'disable_blink_features': ['AutomationControlled'],
            'exclude_switches': ['--enable-automation', '--enable-blink-features=AutomationControlled'],
            'use_automation_extension': False,
            'disable_dev_shm_usage': True,
            'disable_extensions': False,
            'disable_plugins': False,
            'disable_default_apps': True,
            'disable_background_timer_throttling': True,
            'disable_backgrounding_occluded_windows': True,
            'disable_renderer_backgrounding': True,
            'disable_features': ['TranslateUI', 'BlinkGenPropertyTrees'],
            'disable_ipc_flooding_protection': True
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def start(self):
        """Start Playwright browser."""
        try:
            logger.info("Starting Playwright browser...")
            
            self.playwright = await async_playwright().start()
            
            # Choose browser type (Chrome for better compatibility)
            browser_args = []
            
            if self.enable_stealth:
                browser_args.extend([
                    '--no-first-run',
                    '--no-default-browser-check',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--disable-extensions',
                    '--disable-plugins',
                    '--disable-default-apps',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--disable-features=TranslateUI,BlinkGenPropertyTrees',
                    '--disable-ipc-flooding-protection'
                ])
            
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=browser_args
            )
            
            # Create context with anti-detection settings
            context_options = await self._get_context_options()
            self.context = await self.browser.new_context(**context_options)
            
            # Add stealth scripts
            if self.enable_stealth:
                await self._add_stealth_scripts()
            
            logger.info("Playwright browser started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Playwright browser: {e}")
            raise BrowserAutomationError(f"Browser startup failed: {e}")
    
    async def close(self):
        """Close Playwright browser."""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            
            logger.info("Playwright browser closed")
            
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
    
    async def scrape_url(
        self, 
        url: str, 
        use_recipe: bool = True,
        take_screenshot: bool = False,
        wait_for_selector: Optional[str] = None,
        execute_js: Optional[str] = None
    ) -> Tuple[ArticleCreate, Optional[Dict[str, Any]]]:
        """
        Scrape content from a URL using browser automation.
        
        Args:
            url: URL to scrape
            use_recipe: Whether to use cached recipes
            take_screenshot: Whether to take a screenshot
            wait_for_selector: CSS selector to wait for before scraping
            execute_js: JavaScript code to execute before scraping
            
        Returns:
            Tuple of (article_data, metadata)
        """
        page = None
        try:
            logger.info(f"Starting browser scrape for URL: {url}")
            
            # Create new page
            page = await self.context.new_page()
            
            # Set up page with anti-detection
            await self._setup_page(page)
            
            # Navigate to URL
            await self._navigate_to_url(page, url)
            
            # Wait for specific selector if provided
            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=self.timeout)
            
            # Execute custom JavaScript if provided
            if execute_js:
                await page.evaluate(execute_js)
            
            # Wait for page to be fully loaded
            await self._wait_for_page_load(page)
            
            # Take screenshot if requested
            screenshot_data = None
            if take_screenshot:
                screenshot_data = await self._take_screenshot(page, url)
            
            # Get page content
            html_content = await page.content()
            
            # Analyze DOM structure
            dom_analysis = await self._analyze_dom(page)
            
            # Extract content using recipe or fallback
            if use_recipe:
                article_data = await self._extract_with_recipe(url, html_content, page)
            else:
                article_data = await self._extract_fallback(url, html_content, page)
            
            # Prepare metadata
            metadata = {
                'dom_analysis': dom_analysis,
                'screenshot': screenshot_data,
                'page_metrics': await self._get_page_metrics(page),
                'network_requests': await self._get_network_info(page),
                'javascript_errors': await self._get_js_errors(page)
            }
            
            logger.info(f"Successfully scraped URL with browser: {url}")
            return article_data, metadata
            
        except PlaywrightTimeoutError as e:
            logger.error(f"Timeout error scraping {url}: {e}")
            raise BrowserAutomationError(f"Page load timeout for {url}: {e}")
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            raise BrowserAutomationError(f"Browser scraping failed for {url}: {e}")
        finally:
            if page:
                await page.close()
    
    async def scrape_multiple(
        self,
        urls: List[str],
        use_recipe: bool = True,
        max_concurrent: int = 3,
        take_screenshots: bool = False
    ) -> List[Tuple[str, Optional[ArticleCreate], Optional[Dict[str, Any]], Optional[Exception]]]:
        """
        Scrape multiple URLs concurrently with browser automation.
        
        Args:
            urls: List of URLs to scrape
            use_recipe: Whether to use cached recipes
            max_concurrent: Maximum concurrent browser tabs
            take_screenshots: Whether to take screenshots
            
        Returns:
            List of tuples (url, article_data, metadata, error)
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_single(url: str):
            async with semaphore:
                try:
                    article_data, metadata = await self.scrape_url(
                        url, use_recipe, take_screenshots
                    )
                    return url, article_data, metadata, None
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
                    return url, None, None, e
        
        logger.info(f"Starting concurrent browser scrape of {len(urls)} URLs")
        results = await asyncio.gather(*[scrape_single(url) for url in urls])
        
        successful = sum(1 for _, article, _, _ in results if article is not None)
        logger.info(f"Completed browser scraping: {successful}/{len(urls)} successful")
        
        return results
    
    async def test_anti_detection(self, test_url: str = "https://bot.sannysoft.com/") -> Dict[str, Any]:
        """
        Test anti-bot detection capabilities.
        
        Args:
            test_url: URL to test against (default is bot detection test site)
            
        Returns:
            Dictionary with detection test results
        """
        page = None
        try:
            logger.info(f"Testing anti-detection capabilities on: {test_url}")
            
            page = await self.context.new_page()
            await self._setup_page(page)
            
            # Navigate to test URL
            await page.goto(test_url, wait_until='networkidle', timeout=self.timeout)
            
            # Wait for page to load completely
            await asyncio.sleep(3)
            
            # Take screenshot for analysis
            screenshot_path = self.screenshot_dir / f"anti_detection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            
            # Analyze detection results
            detection_results = await page.evaluate("""
                () => {
                    const results = {};
                    
                    // Check for common detection indicators
                    results.webdriver = navigator.webdriver;
                    results.plugins = navigator.plugins.length;
                    results.languages = navigator.languages;
                    results.platform = navigator.platform;
                    results.userAgent = navigator.userAgent;
                    results.cookieEnabled = navigator.cookieEnabled;
                    results.doNotTrack = navigator.doNotTrack;
                    results.hardwareConcurrency = navigator.hardwareConcurrency;
                    results.maxTouchPoints = navigator.maxTouchPoints;
                    
                    // Check for automation indicators
                    results.automationControlled = window.chrome && window.chrome.runtime && window.chrome.runtime.onConnect;
                    results.phantomJS = window.callPhantom || window._phantom;
                    results.selenium = window.document.$cdc_asdjflasutopfhvcZLmcfl_;
                    
                    // Check screen properties
                    results.screenWidth = screen.width;
                    results.screenHeight = screen.height;
                    results.availWidth = screen.availWidth;
                    results.availHeight = screen.availHeight;
                    results.colorDepth = screen.colorDepth;
                    results.pixelDepth = screen.pixelDepth;
                    
                    return results;
                }
            """)
            
            return {
                'success': True,
                'url': test_url,
                'screenshot_path': str(screenshot_path),
                'detection_results': detection_results,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'url': test_url,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
        finally:
            if page:
                await page.close()
    
    async def _get_context_options(self) -> Dict[str, Any]:
        """Get browser context options with anti-detection settings."""
        screen_size = random.choice(self.SCREEN_SIZES)
        user_agent = random.choice(self.USER_AGENTS)
        
        return {
            'user_agent': user_agent,
            'viewport': screen_size,
            'device_scale_factor': random.choice([1, 1.25, 1.5, 2]),
            'is_mobile': False,
            'has_touch': False,
            'locale': random.choice(['en-US', 'en-GB', 'en-CA']),
            'timezone_id': random.choice(['America/New_York', 'America/Los_Angeles', 'Europe/London']),
            'permissions': ['geolocation'],
            'extra_http_headers': {
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }
        }
    
    async def _add_stealth_scripts(self):
        """Add stealth scripts to bypass detection."""
        stealth_script = """
        // Override webdriver property
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });
        
        // Override plugins
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });
        
        // Override languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });
        
        // Override chrome property
        window.chrome = {
            runtime: {},
        };
        
        // Override permissions
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );
        
        // Override getParameter
        const getParameter = WebGLRenderingContext.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {
            if (parameter === 37445) {
                return 'Intel Inc.';
            }
            if (parameter === 37446) {
                return 'Intel Iris OpenGL Engine';
            }
            return getParameter(parameter);
        };
        """
        
        await self.context.add_init_script(stealth_script)
    
    async def _setup_page(self, page: Page):
        """Set up page with anti-detection measures."""
        # Set extra headers
        await page.set_extra_http_headers({
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Block unnecessary resources for faster loading
        await page.route("**/*", lambda route: (
            route.abort() if route.request.resource_type in ["image", "stylesheet", "font", "media"]
            else route.continue_()
        ))
        
        # Add random mouse movements
        await page.evaluate("""
            () => {
                let mouseX = 0;
                let mouseY = 0;
                
                document.addEventListener('mousemove', (e) => {
                    mouseX = e.clientX;
                    mouseY = e.clientY;
                });
                
                // Simulate random mouse movements
                setInterval(() => {
                    const event = new MouseEvent('mousemove', {
                        clientX: mouseX + Math.random() * 10 - 5,
                        clientY: mouseY + Math.random() * 10 - 5
                    });
                    document.dispatchEvent(event);
                }, 1000 + Math.random() * 2000);
            }
        """)
    
    async def _navigate_to_url(self, page: Page, url: str):
        """Navigate to URL with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await page.goto(
                    url,
                    wait_until='networkidle',
                    timeout=self.timeout
                )
                return
            except PlaywrightTimeoutError:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Navigation timeout for {url}, retrying... (attempt {attempt + 1})")
                await asyncio.sleep(2)
    
    async def _wait_for_page_load(self, page: Page):
        """Wait for page to be fully loaded."""
        try:
            # Wait for network to be idle
            await page.wait_for_load_state('networkidle', timeout=10000)
            
            # Wait for any dynamic content
            await asyncio.sleep(2)
            
            # Check if page is still loading
            is_loading = await page.evaluate("""
                () => document.readyState !== 'complete' || 
                     document.querySelector('[data-loading]') !== null ||
                     document.querySelector('.loading') !== null
            """)
            
            if is_loading:
                await asyncio.sleep(3)
                
        except PlaywrightTimeoutError:
            logger.warning("Page load wait timeout, continuing...")
    
    async def _take_screenshot(self, page: Page, url: str) -> Optional[Dict[str, Any]]:
        """Take screenshot of the page."""
        try:
            # Ensure screenshot directory exists
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            domain = urlparse(url).netloc.replace('.', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{domain}_{timestamp}.png"
            filepath = self.screenshot_dir / filename
            
            # Take screenshot
            await page.screenshot(
                path=filepath,
                full_page=True,
                type='png'
            )
            
            # Get screenshot as base64 for embedding
            screenshot_bytes = await page.screenshot(full_page=True, type='png')
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode()
            
            return {
                'filepath': str(filepath),
                'filename': filename,
                'base64': screenshot_base64,
                'size': len(screenshot_bytes),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Failed to take screenshot: {e}")
            return None
    
    async def _analyze_dom(self, page: Page) -> Dict[str, Any]:
        """Analyze DOM structure for content extraction insights."""
        try:
            analysis = await page.evaluate("""
                () => {
                    const analysis = {};
                    
                    // Count elements
                    analysis.totalElements = document.querySelectorAll('*').length;
                    analysis.headings = {
                        h1: document.querySelectorAll('h1').length,
                        h2: document.querySelectorAll('h2').length,
                        h3: document.querySelectorAll('h3').length,
                        h4: document.querySelectorAll('h4').length,
                        h5: document.querySelectorAll('h5').length,
                        h6: document.querySelectorAll('h6').length
                    };
                    analysis.paragraphs = document.querySelectorAll('p').length;
                    analysis.links = document.querySelectorAll('a').length;
                    analysis.images = document.querySelectorAll('img').length;
                    analysis.forms = document.querySelectorAll('form').length;
                    analysis.scripts = document.querySelectorAll('script').length;
                    
                    // Analyze content structure
                    const articles = document.querySelectorAll('article');
                    const mains = document.querySelectorAll('main');
                    const contentDivs = document.querySelectorAll('div[class*="content"], div[class*="article"], div[class*="post"]');
                    
                    analysis.contentContainers = {
                        articles: articles.length,
                        mains: mains.length,
                        contentDivs: contentDivs.length
                    };
                    
                    // Find potential title elements
                    const titleCandidates = [];
                    document.querySelectorAll('h1, h2, .title, .headline, [class*="title"]').forEach(el => {
                        if (el.textContent.trim().length > 10) {
                            titleCandidates.push({
                                tag: el.tagName.toLowerCase(),
                                class: el.className,
                                text: el.textContent.trim().substring(0, 100)
                            });
                        }
                    });
                    analysis.titleCandidates = titleCandidates.slice(0, 5);
                    
                    // Check for JavaScript frameworks
                    analysis.frameworks = {
                        react: !!window.React || document.querySelector('[data-reactroot]') !== null,
                        vue: !!window.Vue || document.querySelector('[data-v-]') !== null,
                        angular: !!window.angular || document.querySelector('[ng-app]') !== null,
                        jquery: !!window.jQuery || !!window.$
                    };
                    
                    // Check page performance
                    if (window.performance && window.performance.timing) {
                        const timing = window.performance.timing;
                        analysis.loadTime = timing.loadEventEnd - timing.navigationStart;
                        analysis.domContentLoaded = timing.domContentLoadedEventEnd - timing.navigationStart;
                    }
                    
                    return analysis;
                }
            """)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"DOM analysis failed: {e}")
            return {}
    
    async def _get_page_metrics(self, page: Page) -> Dict[str, Any]:
        """Get page performance metrics."""
        try:
            metrics = await page.evaluate("""
                () => {
                    const metrics = {};
                    
                    if (window.performance) {
                        const navigation = performance.getEntriesByType('navigation')[0];
                        if (navigation) {
                            metrics.loadTime = navigation.loadEventEnd - navigation.fetchStart;
                            metrics.domContentLoaded = navigation.domContentLoadedEventEnd - navigation.fetchStart;
                            metrics.firstPaint = navigation.responseEnd - navigation.fetchStart;
                        }
                        
                        // Get resource timing
                        const resources = performance.getEntriesByType('resource');
                        metrics.resourceCount = resources.length;
                        metrics.totalResourceSize = resources.reduce((sum, r) => sum + (r.transferSize || 0), 0);
                    }
                    
                    // Get viewport info
                    metrics.viewport = {
                        width: window.innerWidth,
                        height: window.innerHeight,
                        devicePixelRatio: window.devicePixelRatio
                    };
                    
                    return metrics;
                }
            """)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to get page metrics: {e}")
            return {}
    
    async def _get_network_info(self, page: Page) -> Dict[str, Any]:
        """Get network request information."""
        try:
            # This would require setting up request/response listeners
            # For now, return basic info
            return {
                'requests_made': True,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.warning(f"Failed to get network info: {e}")
            return {}
    
    async def _get_js_errors(self, page: Page) -> List[str]:
        """Get JavaScript errors from the page."""
        try:
            errors = await page.evaluate("""
                () => {
                    const errors = [];
                    const originalError = window.onerror;
                    
                    window.onerror = function(message, source, lineno, colno, error) {
                        errors.push({
                            message: message,
                            source: source,
                            line: lineno,
                            column: colno
                        });
                        if (originalError) originalError.apply(this, arguments);
                    };
                    
                    return errors;
                }
            """)
            
            return errors
            
        except Exception as e:
            logger.warning(f"Failed to get JS errors: {e}")
            return []
    
    async def _extract_with_recipe(self, url: str, html_content: str, page: Page) -> ArticleCreate:
        """Extract content using cached recipe with browser context."""
        domain = urlparse(url).netloc
        
        # Try to get cached recipe
        recipe = await self.recipe_engine.get_recipe(domain)
        
        if recipe:
            try:
                # Execute recipe with browser context for dynamic content
                article_data = await self._execute_recipe_with_browser(recipe, page, url)
                
                # Update recipe success
                await self.recipe_engine.update_recipe_success(domain, True)
                
                logger.info(f"Successfully used recipe for {domain} with browser")
                return article_data
                
            except Exception as e:
                logger.warning(f"Recipe execution failed for {domain}: {e}")
                
                # Update recipe failure
                await self.recipe_engine.update_recipe_success(domain, False)
                
                # Fall back to generic extraction
                return await self._extract_fallback(url, html_content, page)
        else:
            logger.info(f"No recipe found for {domain}, using browser fallback extraction")
            return await self._extract_fallback(url, html_content, page)
    
    async def _execute_recipe_with_browser(
        self, 
        recipe: ScrapingRecipeResponse, 
        page: Page, 
        url: str
    ) -> ArticleCreate:
        """Execute recipe using browser for dynamic content."""
        domain = urlparse(url).netloc
        
        # Execute recipe actions if any
        for action in recipe.actions:
            await self._execute_action(page, action)
        
        # Extract content using selectors
        extracted_data = {}
        
        # Title (required)
        title_element = await page.query_selector(recipe.selectors.title)
        if not title_element:
            raise Exception(f"Title selector '{recipe.selectors.title}' found no elements")
        extracted_data['title'] = await title_element.text_content()
        
        # Content (required)
        content_element = await page.query_selector(recipe.selectors.content)
        if not content_element:
            raise Exception(f"Content selector '{recipe.selectors.content}' found no elements")
        extracted_data['content'] = await content_element.text_content()
        
        # Author (optional)
        if recipe.selectors.author:
            author_element = await page.query_selector(recipe.selectors.author)
            if author_element:
                extracted_data['author'] = await author_element.text_content()
        
        # Publish date (optional)
        if recipe.selectors.publish_date:
            date_element = await page.query_selector(recipe.selectors.publish_date)
            if date_element:
                date_text = await date_element.text_content()
                extracted_data['published_at'] = self._parse_date(date_text)
        
        # Summary (optional)
        if recipe.selectors.summary:
            summary_element = await page.query_selector(recipe.selectors.summary)
            if summary_element:
                extracted_data['summary'] = await summary_element.text_content()
        
        return ArticleCreate(
            url=url,
            title=self._clean_text(extracted_data['title']),
            content=self._clean_text(extracted_data.get('content')),
            summary=self._clean_text(extracted_data.get('summary')),
            author=self._clean_text(extracted_data.get('author')),
            published_at=extracted_data.get('published_at'),
            source_domain=domain
        )
    
    async def _execute_action(self, page: Page, action):
        """Execute a scraping action on the page."""
        try:
            if action.type == 'click':
                if action.selector:
                    await page.click(action.selector, timeout=action.timeout * 1000)
            elif action.type == 'wait':
                await asyncio.sleep(action.timeout)
            elif action.type == 'scroll':
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            elif action.type == 'input':
                if action.selector and action.value:
                    await page.fill(action.selector, action.value)
            elif action.type == 'hover':
                if action.selector:
                    await page.hover(action.selector)
                    
        except Exception as e:
            logger.warning(f"Action execution failed: {action.type} - {e}")
    
    async def _extract_fallback(self, url: str, html_content: str, page: Page) -> ArticleCreate:
        """Extract content using fallback heuristics with browser context."""
        domain = urlparse(url).netloc
        
        # Use browser to extract content dynamically
        extracted_data = await page.evaluate("""
            () => {
                const data = {};
                
                // Extract title
                const titleSelectors = [
                    'h1.entry-title', 'h1.post-title', 'h1.article-title',
                    'h1[class*="title"]', '.entry-title', '.post-title',
                    '.article-title', 'h1', 'title'
                ];
                
                for (const selector of titleSelectors) {
                    const element = document.querySelector(selector);
                    if (element && element.textContent.trim().length > 5) {
                        data.title = element.textContent.trim();
                        break;
                    }
                }
                
                // Extract content
                const contentSelectors = [
                    '.entry-content', '.post-content', '.article-content',
                    '.content', '[class*="content"]', '.post-body',
                    '.article-body', 'main', 'article'
                ];
                
                for (const selector of contentSelectors) {
                    const element = document.querySelector(selector);
                    if (element && element.textContent.trim().length > 100) {
                        data.content = element.textContent.trim();
                        break;
                    }
                }
                
                // Extract author
                const authorSelectors = [
                    '.author', '.byline', '.post-author', '.article-author',
                    '[rel="author"]', '[class*="author"]', '.by-author'
                ];
                
                for (const selector of authorSelectors) {
                    const element = document.querySelector(selector);
                    if (element && element.textContent.trim().length < 100) {
                        data.author = element.textContent.trim().replace(/^By\\s+/i, '');
                        break;
                    }
                }
                
                // Extract summary from meta description
                const metaDesc = document.querySelector('meta[name="description"]');
                if (metaDesc && metaDesc.content) {
                    data.summary = metaDesc.content.trim();
                }
                
                return data;
            }
        """)
        
        if not extracted_data.get('title'):
            raise BrowserAutomationError("Could not extract title")
        
        if not extracted_data.get('content'):
            raise BrowserAutomationError("Could not extract content")
        
        return ArticleCreate(
            url=url,
            title=extracted_data['title'],
            content=extracted_data['content'],
            summary=extracted_data.get('summary'),
            author=extracted_data.get('author'),
            published_at=None,  # Would need more sophisticated date parsing
            source_domain=domain
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        cleaned = " ".join(text.split())
        return cleaned.strip()
    
    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse date from text."""
        try:
            from dateutil import parser
            return parser.parse(date_text)
        except Exception:
            logger.warning(f"Could not parse date: {date_text}")
            return None


# Global scraper instance
playwright_scraper = PlaywrightScraper()