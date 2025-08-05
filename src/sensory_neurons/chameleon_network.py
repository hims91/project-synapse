"""
Sensory Neurons - Chameleon Network (Proxy System)
Layer 1: Perception Layer

This module implements advanced proxy rotation with multiple providers,
Tor network integration, and IP reputation management for maximum anonymity.
"""
import asyncio
import logging
import random
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse
import json

import httpx
import aiohttp
from aiohttp_socks import ProxyConnector

logger = logging.getLogger(__name__)


class ProxyType(Enum):
    """Proxy connection types."""
    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS5 = "socks5"
    TOR = "tor"


class ProxyStatus(Enum):
    """Proxy status states."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BANNED = "banned"
    TESTING = "testing"
    FAILED = "failed"


@dataclass
class ProxyInfo:
    """Information about a proxy server."""
    host: str
    port: int
    proxy_type: ProxyType
    username: Optional[str] = None
    password: Optional[str] = None
    country: Optional[str] = None
    provider: Optional[str] = None
    status: ProxyStatus = ProxyStatus.INACTIVE
    last_used: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    response_time: float = 0.0
    reputation_score: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def proxy_url(self) -> str:
        """Get proxy URL."""
        if self.username and self.password:
            return f"{self.proxy_type.value}://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"{self.proxy_type.value}://{self.host}:{self.port}"
    
    def update_stats(self, success: bool, response_time: float = 0.0):
        """Update proxy statistics."""
        if success:
            self.success_count += 1
            self.status = ProxyStatus.ACTIVE
        else:
            self.failure_count += 1
            if self.failure_count > 5:
                self.status = ProxyStatus.FAILED
        
        self.response_time = response_time
        self.last_used = datetime.utcnow()
        
        # Update reputation score
        self._update_reputation()
    
    def _update_reputation(self):
        """Update reputation score based on performance."""
        base_score = self.success_rate
        
        # Penalize high response times
        if self.response_time > 10:
            base_score *= 0.8
        elif self.response_time > 5:
            base_score *= 0.9
        
        # Penalize old proxies
        age_days = (datetime.utcnow() - self.created_at).days
        if age_days > 30:
            base_score *= 0.9
        
        # Penalize recently failed proxies
        if self.status == ProxyStatus.FAILED:
            base_score *= 0.5
        
        self.reputation_score = max(0.0, min(1.0, base_score))


@dataclass
class ProxyProvider:
    """Proxy provider configuration."""
    name: str
    api_url: str
    api_key: Optional[str] = None
    proxy_types: List[ProxyType] = field(default_factory=list)
    rate_limit: int = 100  # requests per hour
    cost_per_request: float = 0.0
    enabled: bool = True
    last_fetch: Optional[datetime] = None
    fetch_interval: timedelta = field(default_factory=lambda: timedelta(hours=1))


class ChameleonNetwork:
    """
    Advanced proxy rotation and anonymization network.
    
    Provides intelligent proxy selection, rotation, and management
    with support for multiple providers and Tor integration.
    """
    
    def __init__(
        self,
        tor_enabled: bool = True,
        tor_port: int = 9050,
        max_retries: int = 3,
        timeout: float = 30.0,
        reputation_threshold: float = 0.5
    ):
        """
        Initialize Chameleon Network.
        
        Args:
            tor_enabled: Whether to enable Tor integration
            tor_port: Tor SOCKS proxy port
            max_retries: Maximum retry attempts per request
            timeout: Request timeout in seconds
            reputation_threshold: Minimum reputation score for proxy selection
        """
        self.tor_enabled = tor_enabled
        self.tor_port = tor_port
        self.max_retries = max_retries
        self.timeout = timeout
        self.reputation_threshold = reputation_threshold
        
        # Proxy management
        self.proxies: Dict[str, ProxyInfo] = {}
        self.providers: Dict[str, ProxyProvider] = {}
        self.active_proxies: List[str] = []
        self.banned_ips: set = set()
        
        # Usage tracking
        self.usage_stats: Dict[str, Any] = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'proxy_rotations': 0,
            'tor_requests': 0,
            'start_time': datetime.utcnow()
        }
        
        # Current proxy selection
        self.current_proxy: Optional[str] = None
        self.proxy_rotation_count = 0
        
        # Initialize default providers
        self._initialize_default_providers()
        
        # Add Tor proxy if enabled
        if self.tor_enabled:
            self._add_tor_proxy()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self):
        """Start the Chameleon Network."""
        logger.info("Starting Chameleon Network...")
        
        # Fetch proxies from providers
        await self.refresh_proxy_list()
        
        # Test initial proxy connectivity
        await self._test_proxy_connectivity()
        
        logger.info(f"Chameleon Network started with {len(self.active_proxies)} active proxies")
    
    async def stop(self):
        """Stop the Chameleon Network."""
        logger.info("Stopping Chameleon Network...")
        
        # Save proxy statistics
        await self._save_proxy_stats()
        
        logger.info("Chameleon Network stopped")
    
    async def make_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """
        Make an HTTP request through the proxy network.
        
        Args:
            method: HTTP method
            url: Target URL
            **kwargs: Additional request parameters
            
        Returns:
            HTTP response
            
        Raises:
            Exception: If all proxy attempts fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Select best proxy
                proxy_id = await self.select_proxy(url)
                
                if not proxy_id:
                    raise Exception("No available proxies")
                
                proxy = self.proxies[proxy_id]
                
                # Make request through proxy
                start_time = time.time()
                response = await self._make_proxied_request(
                    method, url, proxy, **kwargs
                )
                response_time = time.time() - start_time
                
                # Update proxy stats
                proxy.update_stats(True, response_time)
                
                # Update usage stats
                self.usage_stats['total_requests'] += 1
                self.usage_stats['successful_requests'] += 1
                
                if proxy.proxy_type == ProxyType.TOR:
                    self.usage_stats['tor_requests'] += 1
                
                logger.debug(f"Request successful via {proxy.host}:{proxy.port}")
                return response
                
            except Exception as e:
                last_exception = e
                
                if proxy_id and proxy_id in self.proxies:
                    proxy = self.proxies[proxy_id]
                    proxy.update_stats(False)
                    
                    # Check if proxy should be banned
                    if proxy.failure_count > 10 or proxy.success_rate < 0.1:
                        await self._ban_proxy(proxy_id)
                
                # Update usage stats
                self.usage_stats['total_requests'] += 1
                self.usage_stats['failed_requests'] += 1
                
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                
                # Rotate proxy for next attempt
                await self.rotate_proxy()
                
                # Wait before retry
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        raise Exception(f"All proxy attempts failed. Last error: {last_exception}")
    
    async def select_proxy(self, url: Optional[str] = None) -> Optional[str]:
        """
        Select the best proxy for a request.
        
        Args:
            url: Target URL (for geo-targeting)
            
        Returns:
            Proxy ID or None if no suitable proxy available
        """
        try:
            # Filter available proxies
            available_proxies = [
                proxy_id for proxy_id in self.active_proxies
                if (proxy_id in self.proxies and 
                    self.proxies[proxy_id].status == ProxyStatus.ACTIVE and
                    self.proxies[proxy_id].reputation_score >= self.reputation_threshold)
            ]
            
            if not available_proxies:
                # Try to refresh proxy list
                await self.refresh_proxy_list()
                available_proxies = [
                    proxy_id for proxy_id in self.active_proxies
                    if (proxy_id in self.proxies and 
                        self.proxies[proxy_id].status == ProxyStatus.ACTIVE)
                ]
            
            if not available_proxies:
                return None
            
            # Select proxy based on strategy
            return await self._select_proxy_by_strategy(available_proxies, url)
            
        except Exception as e:
            logger.error(f"Error selecting proxy: {e}")
            return None
    
    async def rotate_proxy(self):
        """Rotate to a different proxy."""
        try:
            new_proxy = await self.select_proxy()
            
            if new_proxy and new_proxy != self.current_proxy:
                self.current_proxy = new_proxy
                self.proxy_rotation_count += 1
                self.usage_stats['proxy_rotations'] += 1
                
                logger.debug(f"Rotated to proxy: {self.proxies[new_proxy].host}")
            
        except Exception as e:
            logger.error(f"Error rotating proxy: {e}")
    
    async def refresh_proxy_list(self):
        """Refresh proxy list from all providers."""
        logger.info("Refreshing proxy list from providers...")
        
        for provider_name, provider in self.providers.items():
            if not provider.enabled:
                continue
            
            # Check rate limiting
            if (provider.last_fetch and 
                datetime.utcnow() - provider.last_fetch < provider.fetch_interval):
                continue
            
            try:
                await self._fetch_proxies_from_provider(provider)
                provider.last_fetch = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Error fetching proxies from {provider_name}: {e}")
        
        # Update active proxy list
        self._update_active_proxy_list()
        
        logger.info(f"Proxy list refreshed: {len(self.active_proxies)} active proxies")
    
    async def test_proxy(self, proxy_id: str) -> bool:
        """
        Test a specific proxy.
        
        Args:
            proxy_id: Proxy identifier
            
        Returns:
            True if proxy is working
        """
        if proxy_id not in self.proxies:
            return False
        
        proxy = self.proxies[proxy_id]
        proxy.status = ProxyStatus.TESTING
        
        try:
            # Test with a simple HTTP request
            start_time = time.time()
            response = await self._make_proxied_request(
                "GET", "http://httpbin.org/ip", proxy, timeout=10
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                proxy.update_stats(True, response_time)
                logger.debug(f"Proxy test successful: {proxy.host}:{proxy.port}")
                return True
            else:
                proxy.update_stats(False)
                return False
                
        except Exception as e:
            proxy.update_stats(False)
            logger.debug(f"Proxy test failed: {proxy.host}:{proxy.port} - {e}")
            return False
    
    async def get_current_ip(self) -> Optional[str]:
        """Get current external IP address."""
        try:
            response = await self.make_request("GET", "http://httpbin.org/ip")
            data = response.json()
            return data.get('origin')
            
        except Exception as e:
            logger.error(f"Error getting current IP: {e}")
            return None
    
    async def get_network_stats(self) -> Dict[str, Any]:
        """Get network usage statistics."""
        stats = self.usage_stats.copy()
        
        # Add current status
        stats.update({
            'total_proxies': len(self.proxies),
            'active_proxies': len(self.active_proxies),
            'current_proxy': self.current_proxy,
            'banned_ips': len(self.banned_ips),
            'providers': len(self.providers),
            'uptime_seconds': (datetime.utcnow() - stats['start_time']).total_seconds()
        })
        
        # Add proxy performance stats
        if self.proxies:
            response_times = [p.response_time for p in self.proxies.values() if p.response_time > 0]
            reputation_scores = [p.reputation_score for p in self.proxies.values()]
            
            stats.update({
                'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                'avg_reputation_score': sum(reputation_scores) / len(reputation_scores) if reputation_scores else 0,
                'success_rate': stats['successful_requests'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
            })
        
        return stats
    
    def add_provider(self, provider: ProxyProvider):
        """Add a proxy provider."""
        self.providers[provider.name] = provider
        logger.info(f"Added proxy provider: {provider.name}")
    
    def remove_provider(self, provider_name: str):
        """Remove a proxy provider."""
        if provider_name in self.providers:
            del self.providers[provider_name]
            logger.info(f"Removed proxy provider: {provider_name}")
    
    async def _make_proxied_request(
        self,
        method: str,
        url: str,
        proxy: ProxyInfo,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request through a specific proxy."""
        # Set timeout
        kwargs.setdefault('timeout', self.timeout)
        
        if proxy.proxy_type == ProxyType.TOR:
            # Use aiohttp with SOCKS proxy for Tor
            connector = ProxyConnector.from_url(proxy.proxy_url)
            
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.request(method, url, **kwargs) as response:
                    # Convert aiohttp response to httpx-like response
                    content = await response.read()
                    
                    # Create a mock httpx response
                    class MockResponse:
                        def __init__(self, status_code, content, headers):
                            self.status_code = status_code
                            self.content = content
                            self.headers = headers
                        
                        def json(self):
                            import json
                            return json.loads(self.content)
                        
                        @property
                        def text(self):
                            return self.content.decode('utf-8')
                    
                    return MockResponse(response.status, content, dict(response.headers))
        
        else:
            # Use httpx with HTTP/HTTPS proxy
            proxies = {
                'http://': proxy.proxy_url,
                'https://': proxy.proxy_url
            }
            
            async with httpx.AsyncClient(proxies=proxies) as client:
                return await client.request(method, url, **kwargs)
    
    async def _select_proxy_by_strategy(
        self, 
        available_proxies: List[str], 
        url: Optional[str] = None
    ) -> str:
        """Select proxy using intelligent strategy."""
        # Strategy 1: Round-robin with reputation weighting
        if len(available_proxies) == 1:
            return available_proxies[0]
        
        # Calculate weights based on reputation and performance
        weights = []
        for proxy_id in available_proxies:
            proxy = self.proxies[proxy_id]
            
            # Base weight from reputation
            weight = proxy.reputation_score
            
            # Boost weight for faster proxies
            if proxy.response_time > 0:
                weight *= (10 / max(proxy.response_time, 1))
            
            # Boost weight for less recently used proxies
            if proxy.last_used:
                hours_since_use = (datetime.utcnow() - proxy.last_used).total_seconds() / 3600
                weight *= (1 + min(hours_since_use / 24, 1))  # Up to 2x boost for unused proxies
            
            weights.append(weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(available_proxies)
        
        r = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if r <= cumulative_weight:
                return available_proxies[i]
        
        return available_proxies[-1]  # Fallback
    
    def _initialize_default_providers(self):
        """Initialize default proxy providers."""
        # Free proxy providers (for testing)
        free_provider = ProxyProvider(
            name="free_proxy_list",
            api_url="https://api.proxyscrape.com/v2/",
            proxy_types=[ProxyType.HTTP, ProxyType.SOCKS4, ProxyType.SOCKS5],
            rate_limit=50,
            cost_per_request=0.0
        )
        self.providers["free_proxy_list"] = free_provider
        
        # Add more providers as needed
        # Note: In production, you would add paid proxy providers here
    
    def _add_tor_proxy(self):
        """Add Tor proxy to the network."""
        if not self.tor_enabled:
            return
        
        tor_proxy = ProxyInfo(
            host="127.0.0.1",
            port=self.tor_port,
            proxy_type=ProxyType.TOR,
            provider="tor",
            status=ProxyStatus.ACTIVE,
            reputation_score=0.8  # Tor is reliable but slower
        )
        
        proxy_id = f"tor_{self.tor_port}"
        self.proxies[proxy_id] = tor_proxy
        self.active_proxies.append(proxy_id)
        
        logger.info(f"Added Tor proxy: 127.0.0.1:{self.tor_port}")
    
    async def _fetch_proxies_from_provider(self, provider: ProxyProvider):
        """Fetch proxies from a specific provider."""
        # This is a simplified implementation
        # In production, you would implement specific API calls for each provider
        
        if provider.name == "free_proxy_list":
            await self._fetch_free_proxies()
    
    async def _fetch_free_proxies(self):
        """Fetch free proxies (simplified implementation)."""
        # This is a placeholder - in production you would call actual APIs
        sample_proxies = [
            ("8.8.8.8", 8080, ProxyType.HTTP),
            ("1.1.1.1", 3128, ProxyType.HTTP),
            ("9.9.9.9", 1080, ProxyType.SOCKS5),
        ]
        
        for host, port, proxy_type in sample_proxies:
            proxy_id = f"{host}_{port}"
            
            if proxy_id not in self.proxies:
                proxy = ProxyInfo(
                    host=host,
                    port=port,
                    proxy_type=proxy_type,
                    provider="free_proxy_list"
                )
                self.proxies[proxy_id] = proxy
    
    def _update_active_proxy_list(self):
        """Update the list of active proxies."""
        self.active_proxies = [
            proxy_id for proxy_id, proxy in self.proxies.items()
            if proxy.status in [ProxyStatus.ACTIVE, ProxyStatus.INACTIVE]
        ]
    
    async def _test_proxy_connectivity(self):
        """Test connectivity for all proxies."""
        logger.info("Testing proxy connectivity...")
        
        # Test a sample of proxies
        test_proxies = list(self.proxies.keys())[:10]  # Test first 10
        
        tasks = [self.test_proxy(proxy_id) for proxy_id in test_proxies]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_tests = sum(1 for result in results if result is True)
        logger.info(f"Proxy connectivity test: {successful_tests}/{len(test_proxies)} successful")
    
    async def _ban_proxy(self, proxy_id: str):
        """Ban a proxy due to poor performance."""
        if proxy_id in self.proxies:
            proxy = self.proxies[proxy_id]
            proxy.status = ProxyStatus.BANNED
            
            if proxy_id in self.active_proxies:
                self.active_proxies.remove(proxy_id)
            
            self.banned_ips.add(proxy.host)
            
            logger.warning(f"Banned proxy: {proxy.host}:{proxy.port}")
    
    async def _save_proxy_stats(self):
        """Save proxy statistics to file."""
        try:
            stats_file = "proxy_stats.json"
            stats = await self.get_network_stats()
            
            # Convert datetime objects to strings
            stats['start_time'] = stats['start_time'].isoformat()
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Saved proxy statistics to {stats_file}")
            
        except Exception as e:
            logger.error(f"Error saving proxy stats: {e}")


# Global chameleon network instance
chameleon_network = ChameleonNetwork()