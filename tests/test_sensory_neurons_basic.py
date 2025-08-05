"""
Basic tests for Sensory Neurons components.
Tests basic functionality without complex mocking.
"""
import pytest
from unittest.mock import Mock, AsyncMock

from src.sensory_neurons.recipe_learner import RecipeLearner, LearningExample
from src.sensory_neurons.chameleon_network import ChameleonNetwork, ProxyInfo, ProxyType


class TestRecipeLearnerBasic:
    """Basic tests for RecipeLearner."""
    
    def test_recipe_learner_creation(self):
        """Test that RecipeLearner can be created."""
        learner = RecipeLearner()
        assert learner is not None
        assert hasattr(learner, 'learn_from_examples')
        assert hasattr(learner, 'validate_learned_recipe')
    
    def test_learning_example_creation(self):
        """Test that LearningExample can be created."""
        example = LearningExample(
            url="https://example.com/test",
            html_content="<html><body><h1>Test</h1></body></html>",
            expected_title="Test",
            expected_content="Test content"
        )
        
        assert example.url == "https://example.com/test"
        assert example.expected_title == "Test"
        assert "Test" in example.html_content


class TestChameleonNetworkBasic:
    """Basic tests for ChameleonNetwork."""
    
    def test_chameleon_network_creation(self):
        """Test that ChameleonNetwork can be created."""
        network = ChameleonNetwork()
        assert network is not None
        assert hasattr(network, 'add_proxy')
        assert hasattr(network, 'get_next_proxy')
    
    def test_proxy_info_creation(self):
        """Test that ProxyInfo can be created."""
        proxy = ProxyInfo(
            host="proxy.example.com",
            port=8080,
            proxy_type=ProxyType.HTTP,
            username="user",
            password="pass"
        )
        
        assert proxy.host == "proxy.example.com"
        assert proxy.port == 8080
        assert proxy.proxy_type == ProxyType.HTTP
        assert proxy.username == "user"
    
    def test_proxy_addition(self):
        """Test adding proxies to the network."""
        network = ChameleonNetwork()
        
        proxy = ProxyInfo(
            host="proxy.example.com",
            port=8080,
            proxy_type=ProxyType.HTTP
        )
        
        network.add_proxy(proxy)
        assert len(network.proxies) == 1
        assert network.proxies[0] == proxy
    
    @pytest.mark.asyncio
    async def test_proxy_rotation_basic(self):
        """Test basic proxy rotation."""
        network = ChameleonNetwork()
        
        # Add test proxies
        proxy1 = ProxyInfo(host="proxy1.example.com", port=8080, proxy_type=ProxyType.HTTP)
        proxy2 = ProxyInfo(host="proxy2.example.com", port=8080, proxy_type=ProxyType.SOCKS5)
        
        network.add_proxy(proxy1)
        network.add_proxy(proxy2)
        
        # Test rotation
        first_proxy = await network.get_next_proxy()
        second_proxy = await network.get_next_proxy()
        
        assert first_proxy is not None
        assert second_proxy is not None
        assert first_proxy != second_proxy  # Should rotate


class TestSensoryNeuronsIntegration:
    """Basic integration tests."""
    
    def test_all_components_importable(self):
        """Test that all sensory neuron components can be imported."""
        from src.sensory_neurons import (
            PlaywrightScraper, RecipeLearner, ChameleonNetwork
        )
        
        # Test that classes can be instantiated
        scraper = PlaywrightScraper()
        learner = RecipeLearner()
        network = ChameleonNetwork()
        
        assert scraper is not None
        assert learner is not None
        assert network is not None
    
    def test_component_methods_exist(self):
        """Test that expected methods exist on components."""
        from src.sensory_neurons import (
            PlaywrightScraper, RecipeLearner, ChameleonNetwork
        )
        
        scraper = PlaywrightScraper()
        learner = RecipeLearner()
        network = ChameleonNetwork()
        
        # Test PlaywrightScraper methods
        assert hasattr(scraper, 'start')
        assert hasattr(scraper, 'close')
        assert hasattr(scraper, 'scrape_url')
        assert hasattr(scraper, 'scrape_multiple')
        
        # Test RecipeLearner methods
        assert hasattr(learner, 'learn_from_examples')
        assert hasattr(learner, 'validate_learned_recipe')
        assert hasattr(learner, 'optimize_recipe')
        
        # Test ChameleonNetwork methods
        assert hasattr(network, 'add_proxy')
        assert hasattr(network, 'get_next_proxy')
        assert hasattr(network, 'validate_proxy')


if __name__ == "__main__":
    pytest.main([__file__])