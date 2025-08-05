"""
Minimal Configuration for Neurons Deployment
Simplified config for scraping workers.
"""
import os
from typing import Optional


class SimpleSettings:
    """Simple settings class for Neurons."""
    
    def __init__(self):
        # Application settings
        self.environment = os.getenv("ENVIRONMENT", "production")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.app_name = os.getenv("APP_NAME", "Project Synapse - Neurons")
        self.app_version = os.getenv("APP_VERSION", "1.0.0")
        
        # Hub connection
        self.hub_url = os.getenv("HUB_URL", "https://synapse-central-hub.onrender.com")
        self.neurons_api_key = os.getenv("NEURONS_API_KEY")
        
        # Scraping settings
        self.user_agent = os.getenv("USER_AGENT", "Project Synapse Neurons Scraper 1.0")
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.concurrent_jobs = int(os.getenv("CONCURRENT_JOBS", "5"))
        
        # Rate limiting
        self.requests_per_minute = int(os.getenv("REQUESTS_PER_MINUTE", "60"))
        self.burst_limit = int(os.getenv("BURST_LIMIT", "10"))
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
    
    def is_development(self) -> bool:
        return self.environment == "development"
    
    def is_production(self) -> bool:
        return self.environment == "production"


# Global settings instance
settings = SimpleSettings()


def get_settings() -> SimpleSettings:
    """Get application settings."""
    return settings